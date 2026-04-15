# Chapter 27: MPI with C++20 jthread and packaged_task

## 27.1 Why C++20 Threading with MPI?

C++20 adds two facilities that interact directly with MPI's threading model:

- **`std::jthread`**: a joining thread with cooperative cancellation via
  `std::stop_token`. RAII-safe — automatically joins on destruction, no `join()`
  needed. Solves the "thread outlives MPI_Finalize" class of bugs.
- **`std::packaged_task` / `std::future`**: decouple async MPI operations from
  the thread that eventually waits for them. Useful for building completion pipelines
  where the initiator and waiter are different threads.

The binding constraint remains the MPI thread level (Chapter 21). C++20 threads
are just OS threads from MPI's perspective — the same `MPI_THREAD_MULTIPLE` rules
apply. The advantage is safer lifetime management and cleaner async patterns.

---

## 27.2 Initialization with MPI_Init_thread

Always use `MPI_Init_thread` — never `MPI_Init` — in any C++20 threaded program:

```cpp
#include <mpi.h>
#include <thread>
#include <stdexcept>
#include <print>   /* C++23; use printf/cout for C++20-only */

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
        throw std::runtime_error("MPI_THREAD_MULTIPLE not supported");

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* ... thread-based work ... */

    MPI_Finalize();
}
```

---

## 27.3 jthread: RAII Thread Ownership

`std::jthread` joins automatically on destruction. This eliminates the most common
threading bug in MPI programs — threads outliving `MPI_Finalize`:

```cpp
#include <mpi.h>
#include <thread>
#include <vector>
#include <atomic>

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int NTHREADS = 4;
    std::atomic<int> completed{0};

    {   /* jthreads are destroyed (joined) before MPI_Finalize */
        std::vector<std::jthread> workers;
        workers.reserve(NTHREADS);

        for (int t = 0; t < NTHREADS; t++) {
            workers.emplace_back([&, t]() {
                /* Each thread performs independent MPI sends */
                double data = static_cast<double>(rank * NTHREADS + t);
                int dest = (rank + 1) % size;
                int tag  = t;
                MPI_Send(&data, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
                ++completed;
            });
        }
        /* workers destructor: joins all jthreads here */
    }

    /* Now safe: all threads done before MPI_Finalize */
    MPI_Finalize();
}
```

Compare with `std::thread`: forgetting `join()` before `MPI_Finalize` causes
`std::terminate` or undefined behavior. `std::jthread` eliminates this by design.

---

## 27.4 stop_token: Graceful Thread Shutdown

`std::jthread` provides a `std::stop_token` that threads can poll for cancellation.
This is the correct mechanism for a background MPI progress thread:

```cpp
#include <mpi.h>
#include <thread>
#include <chrono>
using namespace std::chrono_literals;

/* Background thread: drives MPI progress and handles incoming requests */
void progress_thread(std::stop_token stop, MPI_Comm comm)
{
    while (!stop.stop_requested()) {
        int flag;
        /* Poll for incoming control messages */
        MPI_Iprobe(MPI_ANY_SOURCE, CONTROL_TAG, comm, &flag,
                   MPI_STATUS_IGNORE);
        if (flag) {
            int msg;
            MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, CONTROL_TAG,
                     comm, MPI_STATUS_IGNORE);
            handle_control_message(msg);
        }
        /* Yield to avoid spinning 100% */
        std::this_thread::sleep_for(10us);
    }
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    {
        /* Launch progress thread; it receives a stop_token automatically */
        std::jthread prog(progress_thread, MPI_COMM_WORLD);

        /* Main thread does work */
        run_computation();

        /* prog.request_stop() is called automatically by jthread destructor,
           then it joins — clean shutdown before MPI_Finalize */
    }

    MPI_Finalize();
}
```

The `std::stop_callback` variant lets the shutdown signal interrupt a blocking wait.
The thread above uses `sleep_for` to yield between polls, but a thread blocking inside
`MPI_Recv` cannot poll `stop_requested()` at all — it will hang indefinitely even after
`request_stop()` is called.

`std::stop_callback` solves this. It is a **RAII registration object**: its constructor
registers a callable with the `stop_token`; the callable fires automatically (on the
thread that calls `request_stop()`) the moment stop is requested. You never call it
directly — the `jthread` destructor triggers it.

```cpp
void progress_thread(std::stop_token stop, MPI_Comm comm)
{
    /* std::stop_callback constructor registers the lambda NOW.
       The lambda does NOT run here — it runs later, on whichever
       thread calls request_stop() (i.e. the jthread destructor in main).
       When that happens, MPI_Send delivers a sentinel that unblocks
       the MPI_Recv below, allowing the loop to exit cleanly. */
    std::stop_callback on_stop{stop, [&]() {
        int rank;
        MPI_Comm_rank(comm, &rank);
        int sentinel = -1;
        MPI_Send(&sentinel, 1, MPI_INT, rank, CONTROL_TAG, comm);
    }};

    int msg;
    MPI_Status status;
    while (true) {
        MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, CONTROL_TAG, comm, &status);
        if (msg == -1) break;   /* sentinel arrived → stop_callback fired → exit */
        handle_control_message(msg);
    }
    /* on_stop destructor unregisters the callback here (stop already fired; no-op) */
}
```

The full call chain when `main`'s scope closes:

```
prog (jthread) destructor
  └─► request_stop() flips stop_token to "stop requested"
        └─► stop_callback machinery invokes on_stop's lambda
              └─► MPI_Send(-1 sentinel to self)
                    └─► MPI_Recv unblocks, msg == -1, loop breaks
  └─► jthread joins the now-returning progress_thread
        └─► MPI_Finalize() is safe
```

`on_stop`'s lambda runs on **the destructor's thread** (the main thread), not the
progress thread. Both threads call MPI simultaneously — this requires
`MPI_THREAD_MULTIPLE`.


---

## 27.5 packaged_task: Fire-and-Forget MPI Operations

`std::packaged_task` wraps a callable as an async task whose result is retrieved
via `std::future`. This is useful when MPI sends/receives are initiated by one
thread but results are consumed by another.

### Async MPI_Recv Pattern

```cpp
#include <mpi.h>
#include <future>
#include <thread>
#include <vector>

/* Post a non-blocking receive; return a future for the received buffer */
std::future<std::vector<double>>
async_recv(int source, int tag, int count, MPI_Comm comm)
{
    auto task = std::packaged_task<std::vector<double>()>(
        [=]() -> std::vector<double> {
            std::vector<double> buf(count);
            MPI_Recv(buf.data(), count, MPI_DOUBLE, source, tag,
                     comm, MPI_STATUS_IGNORE);
            return buf;
        }
    );

    auto fut = task.get_future();
    /* Use std::thread for fire-and-forget; jthread must NOT be detached
       (detaching a jthread orphans the stop_source and negates RAII safety) */
    std::thread([t = std::move(task)]() mutable { t(); }).detach();
    return fut;
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int N = 1024;

    if (rank == 0) {
        /* Post receives for all peers before doing other work */
        std::vector<std::future<std::vector<double>>> futures;
        for (int src = 1; src < size; src++)
            futures.push_back(async_recv(src, 0, N, MPI_COMM_WORLD));

        /* Do independent work while receives are in progress */
        do_local_computation();

        /* Collect results as futures complete */
        for (int src = 1; src < size; src++) {
            auto buf = futures[src-1].get();
            process_result(src, buf);
        }
    } else {
        std::vector<double> data(N);
        fill_data(data, rank);
        MPI_Send(data.data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
```

### Task Queue Pattern

A thread pool processing MPI operations using `packaged_task`:

```cpp
#include <mpi.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <thread>

class MPITaskQueue {
    std::queue<std::packaged_task<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::jthread worker_;

public:
    MPITaskQueue() : worker_([this](std::stop_token st) {
        while (!st.stop_requested()) {
            std::packaged_task<void()> task;
            {
                std::unique_lock lk(mtx_);
                cv_.wait(lk, [&]{ return !tasks_.empty() || st.stop_requested(); });
                if (tasks_.empty()) break;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();   /* execute MPI operation on dedicated thread */
        }
    }) {}

    template<typename F>
    std::future<std::invoke_result_t<F>> submit(F&& f)
    {
        using R = std::invoke_result_t<F>;
        std::packaged_task<R()> pt(std::forward<F>(f));
        auto fut = pt.get_future();
        {
            std::lock_guard lk(mtx_);
            tasks_.emplace([p = std::move(pt)]() mutable { p(); });
        }
        cv_.notify_one();
        return fut;
    }
};

/* Usage */
MPITaskQueue mpi_queue;

auto fut = mpi_queue.submit([=]() {
    double result;
    MPI_Reduce(&local_val, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return result;
});

do_independent_work();
double global_sum = fut.get();
```

---

## 27.6 MPI_THREAD_FUNNELED with jthread

When `MPI_THREAD_FUNNELED` is sufficient (usually the right choice), jthreads
handle computation while the main thread handles all MPI:

```cpp
#include <mpi.h>
#include <thread>
#include <barrier>   /* C++20 */
#include <vector>
#include <span>

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    constexpr int NTHREADS = 8;
    const int local_N = DOMAIN_SIZE / NTHREADS;

    std::vector<double> data(DOMAIN_SIZE + 2);   /* +2 halos */
    std::vector<double> left_halo_buf(HALO), right_halo_buf(HALO);

    /* C++20 barrier: synchronize NTHREADS workers + 1 main thread = NTHREADS+1 total.
       Completion function runs on the last arriving thread (not guaranteed to be
       the main thread — it could be any worker). MPI calls are made by the main
       thread outside the barrier completion function. */
    std::barrier sync_point(NTHREADS + 1, [&]() noexcept {
        /* intentionally empty — MPI exchange done by main thread below */
    });

    /* Flag: main thread signals when halos are ready */
    std::atomic<bool> halos_ready{false};

    std::vector<std::jthread> workers;
    for (int t = 0; t < NTHREADS; t++) {
        workers.emplace_back([&, t](std::stop_token st) {
            while (!st.stop_requested()) {
                /* Phase 1: compute interior (does not need halos) */
                compute_interior(data, t * local_N, local_N);

                /* Wait for main thread to complete halo exchange */
                sync_point.arrive_and_wait();
                while (!halos_ready.load(std::memory_order_acquire))
                    std::this_thread::yield();

                /* Phase 2: compute boundary using received halos */
                if (t == 0) compute_left_boundary(data, left_halo_buf);
                if (t == NTHREADS-1) compute_right_boundary(data, right_halo_buf);

                sync_point.arrive_and_wait();
                if (t == 0) halos_ready.store(false, std::memory_order_release);
            }
        });
    }

    /* Main thread: drives halo exchange between phases */
    for (int step = 0; step < NSTEPS; step++) {
        /* Wait for all workers to finish phase 1 */
        sync_point.arrive_and_wait();

        /* MPI halo exchange — only main thread calls MPI */
        pack_halos(data, left_halo_buf, right_halo_buf);
        exchange_halos(left_halo_buf, right_halo_buf, rank, size);
        halos_ready.store(true, std::memory_order_release);

        sync_point.arrive_and_wait();
    }

    /* jthreads join on destruction */
    MPI_Finalize();
}
```

---

## 27.7 Communicator-per-Thread Pattern

With `MPI_THREAD_MULTIPLE`, give each thread its own communicator to eliminate
tag/ordering conflicts:

```cpp
#include <mpi.h>
#include <thread>
#include <vector>

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    constexpr int NTHREADS = 4;

    /* Create per-thread communicators (collective — do before launching threads) */
    std::vector<MPI_Comm> thread_comms(NTHREADS);
    for (int t = 0; t < NTHREADS; t++)
        MPI_Comm_dup(MPI_COMM_WORLD, &thread_comms[t]);

    {
        std::vector<std::jthread> workers;
        for (int t = 0; t < NTHREADS; t++) {
            workers.emplace_back([&, t]() {
                /* Each thread has an isolated communicator context */
                MPI_Comm comm = thread_comms[t];
                double val = compute_thread_value(t);
                MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE,
                              MPI_SUM, comm);
                use_result(val, t);
            });
        }
    }   /* jthreads join here */

    /* Free per-thread communicators */
    for (auto &comm : thread_comms)
        MPI_Comm_free(&comm);

    MPI_Finalize();
}
```

---

## 27.8 Common Pitfalls

### Pitfall 1: Detached Thread Outliving MPI_Finalize

```cpp
/* BUG: detached thread may call MPI after Finalize */
std::thread t([]{ MPI_Barrier(MPI_COMM_WORLD); });
t.detach();
MPI_Finalize();   /* may complete before detached thread runs Barrier */
```

Fix: use `std::jthread` with a `std::stop_source`. Never detach threads that call
MPI functions. If you must detach, ensure MPI work completes via a `std::future`
before `MPI_Finalize`.

### Pitfall 2: Collective on Subset of Threads

```cpp
/* BUG: only some threads call Barrier — deadlock */
std::jthread t1([&]{ MPI_Barrier(MPI_COMM_WORLD); });
std::jthread t2([&]{ compute_only(); });   /* no MPI */
MPI_Barrier(MPI_COMM_WORLD);   /* main thread */
```

Collectives must be called by all processes, not all threads. With
`MPI_THREAD_MULTIPLE`, only one thread per process calls any given collective
(use `std::barrier` or a mutex to enforce this).

### Pitfall 3: MPI_Request Shared Between Threads

```cpp
MPI_Request req;
std::jthread sender([&]{ MPI_Isend(buf, N, MPI_DOUBLE, 1, 0, comm, &req); });
MPI_Wait(&req, MPI_STATUS_IGNORE);   /* BUG: req may not be set yet */
```

Fix: synchronize between the Isend and the Wait with a `std::promise`/`std::future`
or `std::atomic<MPI_Request>` (though `MPI_Request` is not lock-free — use a
protecting mutex or the `packaged_task` pattern from Section 27.5).

---

## 27.9 Parallel Execution Policies with MPI

C++17 parallel algorithms (universally available in C++20 compilers) accept an
execution policy as their first argument:

| Policy | Parallelism | Notes |
|---|---|---|
| `std::execution::seq` | None | Same as no policy; safe with any MPI thread level |
| `std::execution::par` | Multi-threaded | Uses an implementation thread pool (TBB, OpenMP, or pthreads) |
| `std::execution::par_unseq` | Threads + SIMD | `par` plus vectorisation; **forbids blocking calls inside the body** |
| `std::execution::unseq` | SIMD only (C++20) | Single thread, vectorised; safe with `MPI_THREAD_FUNNELED` |

### Correct Pattern: Compute Parallel, Communicate Single-Threaded

The cleanest approach is `MPI_THREAD_FUNNELED`: parallel algorithms handle
computation, the main thread handles all MPI.

```cpp
#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    std::vector<double> local(local_N);

    for (int step = 0; step < NSTEPS; ++step) {
        /* Parallel stencil — threads internal to std::for_each, no MPI */
        std::for_each(std::execution::par_unseq,
                      local.begin(), local.end(),
                      [](double &x) { x = stencil(x); });

        /* Local reduction (parallel), then MPI (main thread only) */
        double local_err = std::reduce(std::execution::par_unseq,
                                       local.begin(), local.end(), 0.0);
        MPI_Allreduce(MPI_IN_PLACE, &local_err, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);

        if (local_err < TOLERANCE) break;
    }

    MPI_Finalize();
}
```

### MPI Inside a Parallel Algorithm Body

If the body must call MPI, `MPI_THREAD_MULTIPLE` is required and `par_unseq` is
forbidden (blocking calls are undefined behaviour inside `par_unseq`):

```cpp
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

/* par is safe with MPI_THREAD_MULTIPLE */
std::for_each(std::execution::par,
              indices.begin(), indices.end(),
              [&](int i) {
                  MPI_Send(&local[i], 1, MPI_DOUBLE,
                           dest_rank[i], i, MPI_COMM_WORLD);
              });
```

In practice, calling MPI from inside `std::for_each(par, ...)` creates many
concurrent small sends — high message count with poor aggregation. Prefer
collecting `MPI_Isend` requests inside the body and `MPI_Waitall` outside:

```cpp
std::vector<MPI_Request> reqs(N);
std::transform(std::execution::par,
               indices.begin(), indices.end(), reqs.begin(),
               [&](int i) {
                   MPI_Request r;
                   MPI_Isend(&local[i], 1, MPI_DOUBLE,
                             dest_rank[i], i, MPI_COMM_WORLD, &r);
                   return r;
               });
MPI_Waitall(N, reqs.data(), MPI_STATUSES_IGNORE);
```

---

## 27.10 C++20 Coroutines

C++20 provides the low-level **stackless coroutine** machinery: `co_await`,
`co_yield`, `co_return` keywords and `<coroutine>`. Stackless means only the
coroutine's local variables are saved at each suspension point — not the full
call stack. `co_await` can only appear in the coroutine function itself; called
functions cannot suspend unless they are also coroutines.

C++20 ships no standard coroutine *types*. C++23 adds `std::generator<T>` for
synchronous pull-sequences. For async work you need a library-provided `task<T>`
(cppcoro, libcoro, unifex) or write your own.

### Wrapping MPI_Isend as an Awaitable

```cpp
#include <coroutine>
#include <mpi.h>

/* An awaitable wraps a non-blocking MPI operation into the co_await protocol.
   Three methods are mandatory: */
struct MPIRequestAwaiter {
    MPI_Request req;

    /* Called immediately: if already done, skip suspension entirely */
    bool await_ready() noexcept {
        int flag = 0;
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        return flag != 0;
    }

    /* Called when not ready: h is the suspended coroutine's handle.
       A real event loop would register h and call h.resume() once
       MPI_Test returns flag=1. Without an event loop, this never resumes. */
    void await_suspend(std::coroutine_handle<> h) noexcept {
        event_loop::register_mpi_completion(req, h); /* user-provided */
    }

    void await_resume() noexcept {}  /* return value of co_await expression */
};

MPIRequestAwaiter async_send(const void *buf, int count, MPI_Datatype dt,
                              int dest, int tag, MPI_Comm comm)
{
    MPI_Request req;
    MPI_Isend(buf, count, dt, dest, tag, comm, &req);
    return MPIRequestAwaiter{req};
}
```

Usage inside a coroutine:

```cpp
/* Task<void> is a library-provided coroutine task type */
Task<void> pipeline(double *buf, int N, int dest, MPI_Comm comm)
{
    co_await async_send(buf, N, MPI_DOUBLE, dest, 0, comm);
    /* resumes here only after the send completes */
    co_await async_send(buf, N, MPI_DOUBLE, dest, 1, comm);
}
```

The missing piece is the **event loop**: something that calls `MPI_Test` on all
outstanding requests and calls `h.resume()` when each completes. Without it,
`await_suspend` is reached and the coroutine never resumes. For production MPI
code, the `packaged_task` pattern (Section 27.5) is simpler and does not require
an event loop infrastructure.

---

## 27.11 Boost.Fiber — Stackful Coroutines

### Stackful vs Stackless

| | C++20 coroutines | Boost.Fiber |
|---|---|---|
| **Stack** | Stackless — coroutine frame only | Stackful — full OS-thread-equivalent stack saved |
| **Yield point** | Only at explicit `co_await`/`co_yield` in the coroutine itself | Anywhere in the call graph, including from called functions |
| **Propagation** | `co_await` must appear at every level of the call chain | Transparent — called code yields without modification |
| **Overhead** | Low (small heap-allocated frame) | Higher (full stack, typically 4–128 KB per fiber) |
| **Scheduling** | No built-in scheduler — caller-driven | Pluggable cooperative scheduler built-in |

Boost.Fiber fibers are **green threads**: they present the same API as OS threads
(blocking calls look blocking) but are multiplexed cooperatively onto one or more
OS threads by the fiber scheduler.

### Boost.Fiber with MPI

```cpp
#include <boost/fiber/all.hpp>
#include <mpi.h>

/* Non-blocking MPI wrapped for fibers: polls with yield between tests */
void fiber_send(const double *buf, int N, int dest, int tag)
{
    MPI_Request req;
    MPI_Isend(buf, N, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &req);

    int flag = 0;
    while (!flag) {
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        boost::this_fiber::yield();  /* let other fibers run while waiting */
    }
}

int main(int argc, char *argv[])
{
    /* All fibers run on the main OS thread — FUNNELED is sufficient */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    double buf0[N], buf1[N];
    boost::fibers::fiber f0([&]{ compute(buf0, N); fiber_send(buf0, N, 1, 0); });
    boost::fibers::fiber f1([&]{ compute(buf1, N); fiber_send(buf1, N, 2, 0); });
    f0.join();
    f1.join();

    MPI_Finalize();
}
```

**The blocking problem**: any truly blocking MPI call (`MPI_Recv` waiting for a
message, `MPI_Barrier` waiting for all ranks, `MPI_Send` in rendezvous mode) blocks
the OS thread and with it every fiber on that thread. The fiber scheduler has no way
to preempt a blocked OS thread. The pattern above avoids this by using `MPI_Isend` +
`MPI_Test` + `yield` — never calling a blocking MPI function from a fiber.

### std::fiber_context — Not in C++26

Proposal **P0876** (`std::fiber_context`) aimed to standardise stackful coroutines as
a low-level C++ primitive, analogous to what `std::coroutine_handle` is for stackless
coroutines. It progressed through the C++23 and C++26 study groups but was **voted out
of C++26** at the 2025 Sofia plenary meeting. It remains an active post-C++26 proposal.

Until standardisation, Boost.Fiber is the practical choice. On Linux, `ucontext_t`
from `<ucontext.h>` is the raw POSIX primitive that Boost.Fiber wraps.

---

## 27.12 C++26 Execution — Senders and Receivers

P2300 (`std::execution`) was merged into the C++26 working draft. It provides a
composable, lazy, type-safe async framework built on three concepts:

- **Sender** — a lazy description of async work; does nothing until connected to a
  receiver and started. Carries its result type in its type signature.
- **Receiver** — handles three completion channels: `set_value` (success),
  `set_error` (failure), `set_stopped` (cancellation).
- **Scheduler** — a factory that produces senders which schedule work onto a
  specific execution context (thread pool, I/O ring, GPU queue).

### Key Algorithms

```cpp
namespace ex = std::execution;

ex::schedule(sched)           // start work on sched's context
ex::then(sender, fn)          // fn runs when sender completes; returns sender
ex::when_all(s1, s2, ...)     // completes when all input senders complete
ex::let_value(sender, fn)     // fn receives value and returns a new sender (monadic bind)
ex::on(sched, sender)         // transfer sender's execution to sched
ex::sync_wait(sender)         // block calling thread until sender completes
```

### MPI Operations as Senders

```cpp
#include <execution>   /* C++26 */
#include <mpi.h>

namespace ex = std::execution;

/* Wrap a blocking MPI_Recv as a sender.
   The blocking call runs on a pool thread, freeing the calling thread. */
auto mpi_recv_sender(void *buf, int count, MPI_Datatype dt,
                     int src, int tag, MPI_Comm comm,
                     ex::scheduler auto pool_sched)
{
    return ex::schedule(pool_sched)
         | ex::then([=]() mutable {
               MPI_Recv(buf, count, dt, src, tag, comm, MPI_STATUS_IGNORE);
           });
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    std::execution::thread_pool pool(4);
    auto sched = pool.get_scheduler();

    double buf0[N], buf1[N], result[N];

    /* Receive from two peers concurrently, then reduce */
    auto pipeline =
        ex::when_all(
            mpi_recv_sender(buf0, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, sched),
            mpi_recv_sender(buf1, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, sched)
        )
      | ex::then([&]() {
            /* both recvs complete before this lambda runs */
            for (int i = 0; i < N; ++i) result[i] = buf0[i] + buf1[i];
            MPI_Send(result, N, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
        });

    ex::sync_wait(std::move(pipeline));  /* block until pipeline completes */

    MPI_Finalize();
}
```

### Cancellation Integration with stop_token

The Senders model propagates cancellation through the pipeline automatically. It
integrates with `std::stop_token` (Section 27.4) at the `sync_wait` boundary:

```cpp
std::stop_source ss;

auto work = ex::schedule(sched)
          | ex::then([]{ return compute_local(); })
          | ex::then([](double v) {
                MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE,
                              MPI_SUM, MPI_COMM_WORLD);
                return v;
            });

/* sync_wait checks the stop_token; if stop is requested before the
   pipeline starts, it delivers set_stopped instead of set_value */
auto result = ex::sync_wait(ex::on(sched, std::move(work)));
if (result) {
    double global = std::get<0>(*result);
}
```

### Implementations

| Library | Notes |
|---|---|
| **stdexec** (NVIDIA/Sandia) | Reference implementation; became the basis for the C++26 draft |
| **libunifex** (Meta) | Production use at Meta; predates P2300, partially compatible |
| **Asio** (Boost/standalone) | `asio::execution` supports the Senders model; widely deployed; Boost.Asio 1.78+ / standalone Asio 1.24+ select io_uring as the I/O backend at compile time (`ASIO_HAS_IO_URING`), making it the practical scheduler for pipelines that combine MPI communication with local file I/O |

MPI implementations have not yet adopted Senders natively. Wrapping blocking MPI
calls in scheduler-dispatched senders (as shown above) is the practical approach
until MPI-native async senders exist. The cancellation model aligns naturally with
`std::stop_token`, making Senders and the Section 27.4 `jthread` patterns compose
without impedance mismatch.

---

## 27.13 stdexec — The P2300 Reference Implementation

[**stdexec**](https://github.com/NVIDIA/stdexec) (NVIDIA + Sandia National
Laboratories) is the reference implementation of P2300 that became the basis for
the C++26 `std::execution` draft. It is header-only, available on GitHub, and
ships with two important extensions not in the C++26 standard:

- **`exec::static_thread_pool`** — a work-stealing thread pool with a Senders-native
  scheduler; the standard does not mandate a concrete thread pool type
- **`nvexec::stream_scheduler`** — schedules sender chains onto a CUDA stream;
  allows GPU kernels and MPI operations to be composed in the same pipeline

### Installation

```bash
# CMake FetchContent (header-only — no build step)
include(FetchContent)
FetchContent_Declare(
    stdexec
    GIT_REPOSITORY https://github.com/NVIDIA/stdexec.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(stdexec)

target_link_libraries(myapp PRIVATE stdexec)
```

```bash
# vcpkg
vcpkg install stdexec
```

Compile with C++23 (`-std=c++23`) for full feature coverage; C++20 with
`-DSTDEXEC_NO_SENDER_CONCEPTS` works for most examples.

### Core Scheduler Types

```cpp
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

namespace ex = stdexec;

exec::static_thread_pool pool(4);
auto sched = pool.get_scheduler();

/* schedule(sched) produces a sender that, when started, posts work to the pool */
auto sender = ex::schedule(sched)
            | ex::then([] { return compute_local(); });

auto [result] = ex::sync_wait(std::move(sender)).value();
```

### Blocking MPI Calls on a Thread-Pool Scheduler

The fundamental pattern: offload blocking MPI calls to a pool thread so the
calling thread is free to drive the pipeline.

```cpp
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>
#include <mpi.h>

namespace ex = stdexec;

/* Wrap blocking MPI_Recv as a sender — runs on a pool thread */
auto mpi_recv_sender(exec::static_thread_pool::scheduler sched,
                     double *buf, int count,
                     int src, int tag, MPI_Comm comm)
{
    return ex::schedule(sched)
         | ex::then([=]() mutable {
               MPI_Recv(buf, count, MPI_DOUBLE, src, tag,
                        comm, MPI_STATUS_IGNORE);
           });
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    exec::static_thread_pool pool(4);
    auto sched = pool.get_scheduler();

    double buf_left[N], buf_right[N];

    /* Receive from left and right neighbours concurrently */
    auto pipeline =
        ex::when_all(
            mpi_recv_sender(sched, buf_left,  N, rank-1, TAG_LEFT,  MPI_COMM_WORLD),
            mpi_recv_sender(sched, buf_right, N, rank+1, TAG_RIGHT, MPI_COMM_WORLD)
        )
      | ex::then([&]() {
            /* Both receives complete before this lambda runs */
            update_halos(buf_left, buf_right);
        });

    ex::sync_wait(std::move(pipeline));

    MPI_Finalize();
}
```

### nvexec: GPU Work in the Same Pipeline

The `nvexec::stream_scheduler` schedules work on a CUDA stream. This allows GPU
kernels and MPI receives to be expressed as a single composable pipeline:

```cpp
#include <stdexec/execution.hpp>
#include <nvexec/stream_context.hpp>
#include <exec/static_thread_pool.hpp>
#include <mpi.h>

namespace ex = stdexec;

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    /* CPU thread pool for blocking MPI calls */
    exec::static_thread_pool cpu_pool(2);
    auto cpu_sched = cpu_pool.get_scheduler();

    /* GPU stream context — one CUDA stream per context */
    nvexec::stream_context gpu_ctx;
    auto gpu_sched = gpu_ctx.get_scheduler();

    double *d_buf;  /* CUDA device pointer */
    cudaMalloc(&d_buf, N * sizeof(double));

    /* Pipeline:
       1. CPU thread: MPI_Recv into a staging buffer
       2. Transfer to GPU (on the stream scheduler)
       3. GPU kernel: process data
       4. CPU thread: MPI_Send result */
    double h_buf[N];

    auto pipeline =
        /* Step 1: blocking recv on pool thread */
        ex::schedule(cpu_sched)
      | ex::then([&]() {
            MPI_Recv(h_buf, N, MPI_DOUBLE, src, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        })
        /* Step 2–3: transfer to GPU and run kernel */
      | ex::continues_on(gpu_sched)
      | ex::then([&]() {
            cudaMemcpy(d_buf, h_buf, N * sizeof(double),
                       cudaMemcpyHostToDevice);
            launch_stencil_kernel(d_buf, N);  /* launches on the stream */
        })
        /* Step 4: send result back from CPU thread */
      | ex::continues_on(cpu_sched)
      | ex::then([&]() {
            cudaMemcpy(h_buf, d_buf, N * sizeof(double),
                       cudaMemcpyDeviceToHost);
            MPI_Send(h_buf, N, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        });

    ex::sync_wait(std::move(pipeline));

    cudaFree(d_buf);
    MPI_Finalize();
}
```

```bash
nvcc -std=c++23 -O2 -o solver solver.cu \
    -I/path/to/stdexec/include \
    $(mpicc --showme:compile) $(mpicc --showme:link)
```

`ex::continues_on(sched)` (the P2300 spelling; `ex::transfer` in older snapshots)
transfers the pipeline's execution context from the current scheduler to the
target one. The CUDA stream advances automatically as each GPU-side `then` lambda
returns — no explicit `cudaStreamSynchronize` is needed between GPU steps within
the same pipeline.

For CUDA context setup and CUDA-aware MPI details, see Chapter 30.

### Namespace Quick Reference

| Symbol | Header | Purpose |
|---|---|---|
| `stdexec::then` | `<stdexec/execution.hpp>` | Chain work after a sender |
| `stdexec::when_all` | `<stdexec/execution.hpp>` | Fan-in: wait for all senders |
| `stdexec::sync_wait` | `<stdexec/execution.hpp>` | Block until sender completes |
| `stdexec::continues_on` | `<stdexec/execution.hpp>` | Transfer to a different scheduler |
| `exec::static_thread_pool` | `<exec/static_thread_pool.hpp>` | Work-stealing CPU pool |
| `nvexec::stream_context` | `<nvexec/stream_context.hpp>` | CUDA stream scheduler |
| `nvexec::stream_scheduler` | (from `stream_context`) | `get_scheduler()` result |

---

## 27.14 std::simd — Portable SIMD in C++26

`std::simd` (P1928, merged into C++26) is a portable abstraction over CPU SIMD
registers. It replaces hand-written AVX/SSE intrinsics or compiler-specific
`__attribute__((vector_size(...)))` extensions with standard C++ that the
compiler lowers to native vector instructions.

The section 27.9 `par_unseq` policy internally exploits SIMD — but through the
compiler's auto-vectoriser, not the programmer's explicit control. `std::simd`
gives explicit control when auto-vectorisation fails or when fine-grained mask
operations are needed.

### Core Types

```cpp
#include <simd>   /* C++26; <experimental/simd> in C++23 implementations */
namespace stdx = std::experimental;   /* GCC 11+, Clang 17+ ship the experimental version */

/* simd<T, Abi> — a vector of T elements, width determined by Abi tag */
using floatv  = stdx::simd<float,  stdx::simd_abi::native<float>>;   /* e.g. 8 on AVX2 */
using doublev = stdx::simd<double, stdx::simd_abi::native<double>>;  /* e.g. 4 on AVX2 */

/* fixed_size<N>: explicit width, portable across targets */
using float8 = stdx::simd<float, stdx::simd_abi::fixed_size<8>>;

std::cout << floatv::size() << "\n";   /* 8 on AVX2, 16 on AVX-512 */
```

### Stencil Kernel with std::simd

A 1D Laplacian stencil — the innermost kernel between MPI halo exchanges:

```cpp
#include <simd>
#include <span>
namespace stdx = std::experimental;

using dv = stdx::simd<double, stdx::simd_abi::native<double>>;
constexpr int W = dv::size();   /* SIMD width: 4 (AVX2) or 8 (AVX-512) */

/* Apply u_new[i] = 0.5*(u[i-1] + u[i+1]) over the interior
   buf[-1] and buf[N] are ghost cells already exchanged via MPI */
void laplacian_simd(const double *__restrict__ u,
                    double       *__restrict__ u_new,
                    int N)
{
    int i = 1;
    for (; i + W <= N - 1; i += W) {
        dv left (&u[i - 1], stdx::element_aligned);
        dv centre(&u[i    ], stdx::element_aligned);
        dv right (&u[i + 1], stdx::element_aligned);

        dv result = 0.5 * (left + right);
        result.copy_to(&u_new[i], stdx::element_aligned);
    }
    /* Scalar tail for remaining elements */
    for (; i < N - 1; i++)
        u_new[i] = 0.5 * (u[i-1] + u[i+1]);
}
```

### Masked Operations

`simd_mask` enables per-lane conditional writes without branching:

```cpp
dv vals(&data[i], stdx::element_aligned);
auto mask = (vals > 0.0);          /* simd_mask<double>: true where positive */
dv clamped = stdx::where(mask, vals, dv(0.0));   /* zero out negatives */
clamped.copy_to(&data[i], stdx::element_aligned);
```

### MPI Compatibility

`std::simd` storage is **contiguous** and trivially copyable. A `simd<double, fixed_size<N>>` is
binary-identical to `double[N]` in memory and can be sent directly via MPI:

```cpp
float8 local_vec(&local_data[0], stdx::element_aligned);

/* Send 8 floats — binary layout is identical to float[8] */
MPI_Send(&local_vec, float8::size(), MPI_FLOAT, dest, 0, MPI_COMM_WORLD);

float8 recv_vec;
MPI_Recv(&recv_vec, float8::size(), MPI_FLOAT, src, 0,
         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

Use `fixed_size<N>` rather than `native<T>` for inter-rank sends if processes may
run on hardware with different SIMD widths (heterogeneous clusters). `native<T>` is
safe only when all ranks have the same vector ISA.

### Build Flags

```bash
# GCC / Clang: enable AVX2 (256-bit) or AVX-512 (512-bit)
mpic++ -std=c++23 -O3 -march=native -o solver solver.cpp

# Check which vector ISA the compiler selected:
mpic++ -std=c++23 -O3 -march=native -fopt-info-vec solver.cpp
```

The `<experimental/simd>` header is available in GCC 11+ and Clang 17+ (as
`<experimental/simd>`). The `<simd>` header without `experimental` requires
GCC 15+ or a C++26-mode Clang. For production code today, use the experimental
header with a compatibility alias:

```cpp
#if __cpp_lib_experimental_parallel_simd >= 201803L
#  include <experimental/simd>
   namespace stdx = std::experimental;
#else
#  error "std::simd not available"
#endif
```

---

## Summary

| Feature | MPI Use |
|---|---|
| `std::jthread` | RAII thread ownership; auto-joins before `MPI_Finalize` |
| `std::stop_token` / `stop_callback` | Cooperative shutdown; unblock waiting MPI calls |
| `std::packaged_task` + `future` | Async MPI operations with typed result |
| `std::barrier` | Coordinate funneled threads around MPI calls |
| Per-thread `MPI_Comm_dup` | Isolate thread MPI traffic; eliminate tag conflicts |
| `std::execution::par` | Parallel compute phases; keep MPI outside the algorithm body |
| `std::execution::par_unseq` | Parallel + SIMD compute; **never** call MPI inside body |
| C++20 coroutines (`co_await`) | Awaitable MPI ops; requires a user-built event loop |
| Boost.Fiber | Stackful green threads; use non-blocking MPI + `yield` to avoid blocking the OS thread |
| C++26 `std::execution` (Senders) | Composable async pipelines; wrap MPI ops as senders; cancellation via `stop_token` |
| stdexec (`exec::static_thread_pool`) | Reference P2300 impl; offload blocking MPI to pool thread; `when_all` for concurrent receives |
| stdexec (`nvexec::stream_scheduler`) | Compose GPU kernels + MPI in one pipeline; `continues_on` to switch CPU↔GPU contexts |
| `std::simd` / `fixed_size<N>` | Vectorised stencil kernels; binary-compatible with MPI datatypes; use `fixed_size` for cross-ISA portability |

**Thread level guide**:
- Prefer `MPI_THREAD_FUNNELED` + `std::jthread` workers for compute/communicate overlap
- Use `MPI_THREAD_MULTIPLE` + per-thread `MPI_Comm_dup` for independent thread comms
- `std::execution::par` inside MPI programs requires `MPI_THREAD_MULTIPLE` if the body calls MPI
- Boost.Fiber on one OS thread is safe with `MPI_THREAD_FUNNELED`; never block an OS thread from a fiber
- Never detach threads that call MPI
- All MPI collectives must be called by exactly one thread per process at a time

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
