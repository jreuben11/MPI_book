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

The `std::stop_callback` variant lets the shutdown signal interrupt a blocking wait:

```cpp
void progress_thread(std::stop_token stop, MPI_Comm comm)
{
    std::stop_callback on_stop{stop, [&]() {
        /* Send ourselves a sentinel to unblock MPI_Recv */
        int rank;
        MPI_Comm_rank(comm, &rank);
        int sentinel = -1;
        MPI_Send(&sentinel, 1, MPI_INT, rank, CONTROL_TAG, comm);
    }};

    int msg;
    MPI_Status status;
    while (true) {
        MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, CONTROL_TAG, comm, &status);
        if (msg == -1) break;   /* sentinel: shut down */
        handle_control_message(msg);
    }
}
```

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

## Summary

| C++20 Feature | MPI Use |
|---|---|
| `std::jthread` | RAII thread ownership; auto-joins before `MPI_Finalize` |
| `std::stop_token` | Cooperative shutdown of MPI progress threads |
| `std::stop_callback` | Unblock a waiting MPI call on cancellation |
| `std::packaged_task` | Async MPI operations with `std::future` result |
| `std::barrier` | Coordinate funneled threads around MPI calls |
| Per-thread communicators | Isolate thread MPI traffic; no tag conflicts |

**Thread level guide**:
- Prefer `MPI_THREAD_FUNNELED` + `std::jthread` workers for compute/communicate overlap
- Use `MPI_THREAD_MULTIPLE` + per-thread `MPI_Comm_dup` for independent thread comms
- Never detach threads that call MPI
- All MPI collectives must come from exactly one thread at a time per process
