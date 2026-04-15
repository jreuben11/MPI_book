# Chapter 37: MPI with Intel Threading Building Blocks (TBB)

## 37.1 Why TBB with MPI?

Intel Threading Building Blocks (oneTBB, open-source since 2017) is the most
widely used C++ task-parallelism library in HPC outside of OpenMP. Despite the
"Intel" brand name, oneTBB is fully portable — it uses standard POSIX threads
internally and runs on x86-64 (Intel and AMD), AArch64 (Apple Silicon, AWS
Graviton, NVIDIA Grace), PowerPC/POWER, and RISC-V. The only Intel-specific
feature is `tbb::info::core_types()` for P-core/E-core distinction on Intel
12th Gen+ hybrid CPUs; on all other architectures it returns a single core type
and the rest of the API is identical.

oneTBB provides several capabilities that C++26 does not standardise:

- **Concurrent containers** — thread-safe hash maps, vectors, and queues with
  no external locking required
- **Flow graph** — dataflow/DAG programming model with back-pressure
- **Work-stealing task arenas** — NUMA-aware, P/E-core-aware, nested-parallel scheduler
- **Partitioner control** — cache-affinity, adaptive, and static partitioning for
  iterative algorithms
- **Enumerable thread-local storage** — per-thread accumulators that can be
  iterated and combined
- **Scalable allocator** — thread-local pool allocator that eliminates false
  sharing

In an MPI program, TBB takes over intra-node parallelism. MPI handles inter-node
communication; TBB parallelises computation within each MPI rank.

---

## 37.2 Setup

### Installation

```bash
# Ubuntu / Debian (oneTBB)
sudo apt install libtbb-dev

# Fedora / RHEL
sudo dnf install tbb-devel

# vcpkg
vcpkg install tbb

# Intel oneAPI HPC Toolkit (includes optimised TBB)
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html
```

### CMake Integration

```cmake
cmake_minimum_required(VERSION 3.20)
project(MPITBBProject CXX)

find_package(MPI  REQUIRED)
find_package(TBB  REQUIRED)   # finds oneTBB via TBBConfig.cmake

add_executable(myprogram main.cpp)
target_link_libraries(myprogram
    MPI::MPI_CXX
    TBB::tbb)
target_compile_features(myprogram PRIVATE cxx_std_20)
```

### Minimal Includes

```cpp
#include <mpi.h>
#include <tbb/tbb.h>          /* umbrella header — includes everything */

/* Or include selectively: */
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/flow_graph.h>
#include <tbb/task_arena.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/cache_aligned_allocator.h>
```

---

## 37.3 Thread Safety and MPI Thread Levels

TBB creates its own internal thread pool. From MPI's perspective these are
ordinary OS threads, subject to the same rules as any threaded program.

| TBB usage pattern | Required MPI thread level |
|---|---|
| TBB threads do **no** MPI (compute only) | `MPI_THREAD_FUNNELED` |
| TBB threads call MPI, one at a time (mutex-protected) | `MPI_THREAD_SERIALIZED` |
| TBB threads call MPI concurrently | `MPI_THREAD_MULTIPLE` |

```cpp
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
/* TBB threads do computation; main thread does all MPI */
```

TBB's thread pool is sized at startup by the global arena. To avoid
over-subscribing with MPI ranks:

```cpp
/* Limit TBB to leave one core free for MPI progress */
int hw_concurrency = std::thread::hardware_concurrency();
tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                        hw_concurrency - 1);
```

---

## 37.4 Concurrent Containers

### concurrent_hash_map

A scalable hash map that supports concurrent insertions, lookups, erasures, and
iterations without external locking. Access is via accessor RAII guards that hold
a bucket lock for the duration of the operation.

```cpp
#include <tbb/concurrent_hash_map.h>

using RankMap = tbb::concurrent_hash_map<int, double>;
RankMap rank_results;

/* Multiple TBB threads insert simultaneously — no mutex needed */
tbb::parallel_for(0, local_N, [&](int i) {
    RankMap::accessor acc;          /* write lock on the bucket */
    rank_results.insert(acc, i);
    acc->second += compute(i);
});  /* accessor destructor releases the bucket lock */

/* Read-only access — concurrent with other readers */
RankMap::const_accessor cacc;
if (rank_results.find(cacc, key))
    printf("key %d → %.3f\n", key, cacc->second);

/* After TBB work, collect results for MPI reduction */
std::vector<double> vals;
vals.reserve(rank_results.size());
for (auto &kv : rank_results)      /* safe single-threaded iteration */
    vals.push_back(kv.second);

MPI_Reduce(vals.data(), global.data(), vals.size(),
           MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
```

### concurrent_vector

A dynamically growing vector that is safe for concurrent `push_back` without
invalidating existing element references or iterators. Unlike `std::vector`,
memory is not contiguous (it uses a segmented layout) — access by index is O(1)
but not cache-ideal for sequential scans.

```cpp
#include <tbb/concurrent_vector.h>

tbb::concurrent_vector<double> results;
results.reserve(local_N);

tbb::parallel_for(0, local_N, [&](int i) {
    if (passes_filter(i))
        results.push_back(compute(i));   /* concurrent, safe */
});

/* Linearise for MPI (concurrent_vector is not contiguous) */
std::vector<double> flat(results.begin(), results.end());
int local_count = static_cast<int>(flat.size());

/* Gather variable-length results from all ranks */
std::vector<int> counts(size);
MPI_Allgather(&local_count, 1, MPI_INT,
              counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
```

### concurrent_queue and concurrent_bounded_queue

```cpp
#include <tbb/concurrent_queue.h>

tbb::concurrent_queue<int> work_queue;

/* Producer: fill queue from MPI receives */
for (int i = 0; i < n_items; ++i)
    work_queue.push(i);

/* Consumers: TBB threads drain the queue */
tbb::parallel_for(0, NTHREADS, [&](int) {
    int item;
    while (work_queue.try_pop(item))
        process(item);
});
```

`tbb::concurrent_bounded_queue` adds a capacity limit: `push` blocks (or can
`try_push`) when full, enabling back-pressure between a producer thread and a
pool of consumers.

```cpp
tbb::concurrent_bounded_queue<double*> pipeline_queue;
pipeline_queue.set_capacity(32);    /* at most 32 buffers in flight */

/* Producer (main thread, from MPI) */
for (int step = 0; step < NSTEPS; ++step) {
    double *buf = new double[N];
    MPI_Recv(buf, N, MPI_DOUBLE, src, step, comm, MPI_STATUS_IGNORE);
    pipeline_queue.push(buf);       /* blocks if queue is full */
}

/* Consumer (TBB thread) */
double *buf;
while (pipeline_queue.pop(buf)) {  /* blocks until item available */
    process(buf);
    delete[] buf;
}
```

---

## 37.5 Task Arenas and NUMA-Aware Scheduling

### tbb::task_arena

A task arena is an isolated scheduler context with its own thread count, NUMA
binding, and work-stealing domain. Multiple arenas do not steal work from each
other.

```cpp
#include <tbb/task_arena.h>
#include <tbb/info.h>

/* Query hardware topology */
std::vector<tbb::numa_node_id> numa_nodes = tbb::info::numa_nodes();
std::vector<tbb::core_type_id> core_types = tbb::info::core_types();

printf("NUMA nodes: %zu, core types: %zu\n",
       numa_nodes.size(), core_types.size());

/* Create an arena pinned to NUMA node 0 */
tbb::task_arena arena0(
    tbb::task_arena::constraints{}
        .set_numa_id(numa_nodes[0])
        .set_max_concurrency(16));

/* Run TBB work on NUMA node 0's cores */
arena0.execute([&]{
    tbb::parallel_for(0, local_N, [&](int i){ compute(data0, i); });
});

/* Separate arena for NUMA node 1 — no cross-NUMA stealing */
if (numa_nodes.size() > 1) {
    tbb::task_arena arena1(
        tbb::task_arena::constraints{}
            .set_numa_id(numa_nodes[1])
            .set_max_concurrency(16));
    arena1.execute([&]{
        tbb::parallel_for(0, local_N, [&](int i){ compute(data1, i); });
    });
}
```

### Hybrid CPU Scheduling (P-core / E-core — Intel 12th Gen+ only)

On Intel 12th Gen+ (Alder Lake and later) with P-cores and E-cores, TBB exposes
`core_type_id` to steer work. On AMD, ARM, and other platforms `core_types()`
returns a single entry and this subsection does not apply:

```cpp
/* Get the core type IDs */
auto core_types = tbb::info::core_types();
/* Typically: core_types[0] = E-cores, core_types[1] = P-cores (higher id) */

/* Pin latency-sensitive MPI progress work to P-cores */
tbb::task_arena latency_arena(
    tbb::task_arena::constraints{}
        .set_core_type(core_types.back())   /* highest id = P-cores */
        .set_max_concurrency(8));

/* Background work on E-cores */
tbb::task_arena background_arena(
    tbb::task_arena::constraints{}
        .set_core_type(core_types.front())  /* lowest id = E-cores */
        .set_max_concurrency(16));

latency_arena.execute([&]{
    /* Progress thread or time-critical compute */
});
background_arena.execute([&]{
    /* Prefetch, pre-processing, background I/O */
});
```

### Limiting Global Thread Count with MPI

When running multiple MPI ranks per node (e.g., one rank per NUMA node), each
rank should restrict TBB to its NUMA domain to avoid cross-NUMA false sharing:

```cpp
/* Determine which NUMA node this MPI rank is on */
MPI_Comm shared_comm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                    rank, MPI_INFO_NULL, &shared_comm);
int local_rank;
MPI_Comm_rank(shared_comm, &local_rank);

auto nodes = tbb::info::numa_nodes();
tbb::task_arena my_arena(
    tbb::task_arena::constraints{}
        .set_numa_id(nodes[local_rank % nodes.size()])
        .set_max_concurrency(
            tbb::info::default_concurrency(
                tbb::task_arena::constraints{}.set_numa_id(
                    nodes[local_rank % nodes.size()]))));
```

---

## 37.6 Parallel Algorithms with Partitioner Control

### parallel_for with blocked_range

```cpp
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>

/* 1D range */
tbb::parallel_for(tbb::blocked_range<int>(0, N),
    [&](const tbb::blocked_range<int> &r) {
        for (int i = r.begin(); i < r.end(); ++i)
            output[i] = stencil(input, i);
    });

/* 2D range — natural for matrix/grid operations */
tbb::parallel_for(
    tbb::blocked_range2d<int>(0, ROWS, 0, COLS),
    [&](const tbb::blocked_range2d<int> &r) {
        for (int i = r.rows().begin(); i < r.rows().end(); ++i)
            for (int j = r.cols().begin(); j < r.cols().end(); ++j)
                grid[i][j] = stencil2d(grid, i, j);
    });
```

### Partitioners

```cpp
/* auto_partitioner (default): adaptive grain; self-tuning */
tbb::parallel_for(tbb::blocked_range<int>(0, N),
    kernel, tbb::auto_partitioner{});

/* affinity_partitioner: remembers which thread handled which range.
   Pass the SAME object across iterations — it carries affinity state.
   Critical for iterative stencil codes: same thread = same cache lines. */
tbb::affinity_partitioner ap;
for (int step = 0; step < NSTEPS; ++step) {
    tbb::parallel_for(tbb::blocked_range<int>(0, N), kernel, ap);

    /* MPI halo exchange between steps (main thread only) */
    exchange_halos(data, rank, size);
}

/* static_partitioner: fixed upfront division, no work stealing.
   Zero runtime overhead; use when all ranges have equal cost. */
tbb::parallel_for(tbb::blocked_range<int>(0, N),
    kernel, tbb::static_partitioner{});

/* simple_partitioner: always splits to grain_size; maximum exposure.
   Use to force fine-grained splits for profiling. */
tbb::parallel_for(tbb::blocked_range<int>(0, N, /*grain=*/64),
    kernel, tbb::simple_partitioner{});
```

### parallel_reduce

Tree-based reduction with user-defined split/join:

```cpp
#include <tbb/parallel_reduce.h>

double local_sum = tbb::parallel_reduce(
    tbb::blocked_range<int>(0, N),
    0.0,                                    /* identity */
    [&](const tbb::blocked_range<int> &r, double partial) {
        for (int i = r.begin(); i < r.end(); ++i)
            partial += data[i];
        return partial;
    },
    std::plus<double>{}                     /* join function */
);

/* Reduce across MPI ranks */
double global_sum;
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE,
              MPI_SUM, MPI_COMM_WORLD);
```

### parallel_scan (prefix scan)

```cpp
#include <tbb/parallel_scan.h>

std::vector<double> prefix(N);

tbb::parallel_scan(
    tbb::blocked_range<int>(0, N),
    0.0,                                    /* identity */
    [&](const tbb::blocked_range<int> &r, double sum, bool is_final_scan) {
        for (int i = r.begin(); i < r.end(); ++i) {
            sum += data[i];
            if (is_final_scan) prefix[i] = sum;
        }
        return sum;
    },
    std::plus<double>{}
);
/* prefix[i] = sum of data[0..i] (inclusive) — computed in parallel */
```

---

## 37.7 Enumerable Thread-Specific Storage

`tbb::enumerable_thread_specific<T>` provides per-thread instances that can be
combined after the parallel region. Unlike `thread_local`, it supports iteration
over all threads' values.

```cpp
#include <tbb/enumerable_thread_specific.h>

/* Each TBB thread gets its own histogram — no atomics, no locks */
tbb::enumerable_thread_specific<std::vector<int>> local_hist(
    [&]{ return std::vector<int>(NBINS, 0); }   /* initialiser lambda */
);

tbb::parallel_for(tbb::blocked_range<int>(0, N),
    [&](const tbb::blocked_range<int> &r) {
        auto &h = local_hist.local();           /* this thread's histogram */
        for (int i = r.begin(); i < r.end(); ++i)
            h[bin_of(data[i])]++;
    });

/* Combine all threads' histograms into one */
std::vector<int> hist(NBINS, 0);
for (const auto &h : local_hist)               /* iterate over all threads */
    for (int b = 0; b < NBINS; ++b)
        hist[b] += h[b];

/* MPI reduction across ranks */
MPI_Allreduce(MPI_IN_PLACE, hist.data(), NBINS,
              MPI_INT, MPI_SUM, MPI_COMM_WORLD);
```

`tbb::combinable<T>` is a lighter alternative when you only need the final
`combine` result and not intermediate per-thread access:

```cpp
tbb::combinable<double> partial_max([]{ return -std::numeric_limits<double>::max(); });

tbb::parallel_for(0, N, [&](int i) {
    partial_max.local() = std::max(partial_max.local(), data[i]);
});

double local_max = partial_max.combine([](double a, double b){
    return std::max(a, b);
});

double global_max;
MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE,
              MPI_MAX, MPI_COMM_WORLD);
```

---

## 37.8 Flow Graph — Dataflow Programming

`tbb::flow_graph` represents computation as a DAG of nodes. Data flows as
messages between nodes; the TBB scheduler runs nodes when their inputs are
available. This maps naturally to producer/consumer and pipeline patterns that
span MPI receives and local computation.

### Node Types

| Node | Behaviour |
|---|---|
| `function_node<In, Out>` | Applies a function; `unlimited` or `serial` concurrency |
| `buffer_node<T>` | Stores messages; forwards one at a time |
| `broadcast_node<T>` | Forwards each message to all successors |
| `join_node<tuple<...>>` | Waits for one message on each port; forwards a tuple |
| `split_node<tuple<...>>` | Splits a tuple to separate output ports |
| `limiter_node<T>` | Allows at most N messages in flight (back-pressure) |
| `sequencer_node<T>` | Reorders out-of-order messages by sequence number |
| `source_node<Out>` | Generates messages from a callable (deprecated → `input_node`) |
| `input_node<Out>` | Replaces `source_node`; generates until callable returns false |

### Producer/Consumer Pipeline with MPI Input

```cpp
#include <tbb/flow_graph.h>
#include <mpi.h>

constexpr int MAX_INFLIGHT = 8;   /* back-pressure token limit */

tbb::flow::graph g;

/* Stage 1: receive from MPI (serial — MPI_Recv must serialize) */
tbb::flow::input_node<double *> recv_node(g,
    [&](tbb::flow_control &fc) -> double * {
        if (no_more_data()) { fc.stop(); return nullptr; }
        double *buf = new double[N];
        MPI_Recv(buf, N, MPI_DOUBLE, src, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return buf;
    });

/* Stage 2: transform in parallel (unlimited concurrency) */
tbb::flow::function_node<double *, double *> transform_node(
    g, tbb::flow::unlimited,
    [&](double *buf) -> double * {
        for (int i = 0; i < N; ++i) buf[i] = process(buf[i]);
        return buf;
    });

/* Stage 3: write results (serial — in-order output) */
tbb::flow::function_node<double *, tbb::flow::continue_msg> write_node(
    g, tbb::flow::serial,
    [&](double *buf) {
        write_output(buf);
        delete[] buf;
        return tbb::flow::continue_msg{};
    });

/* Wire the graph */
tbb::flow::make_edge(recv_node,      transform_node);
tbb::flow::make_edge(transform_node, write_node);

recv_node.activate();
g.wait_for_all();
```

### Join Node — Correlating Two MPI Streams

```cpp
/* Receive from two sources; process pairs */
using DataPair = std::tuple<double *, double *>;

tbb::flow::join_node<DataPair, tbb::flow::queueing> joiner(g);

tbb::flow::function_node<DataPair> pair_processor(
    g, tbb::flow::unlimited,
    [&](DataPair pair) {
        auto [a, b] = pair;
        for (int i = 0; i < N; ++i) a[i] += b[i];
        emit_result(a);
        delete[] a; delete[] b;
    });

tbb::flow::make_edge(stream_a_node, tbb::flow::input_port<0>(joiner));
tbb::flow::make_edge(stream_b_node, tbb::flow::input_port<1>(joiner));
tbb::flow::make_edge(joiner, pair_processor);
```

---

## 37.9 parallel_pipeline

`tbb::parallel_pipeline` runs a linear pipeline of serial and parallel stages.
A token limit controls how many items are in-flight simultaneously, providing
back-pressure without explicit queues.

```cpp
#include <tbb/parallel_pipeline.h>

struct Chunk { double *data; int size; int tag; };

tbb::parallel_pipeline(
    /* max tokens in flight */
    MAX_INFLIGHT,

    /* Stage 1: MPI receive (serial_in_order — one receive at a time) */
    tbb::make_filter<void, Chunk *>(
        tbb::filter_mode::serial_in_order,
        [&](tbb::flow_control &fc) -> Chunk * {
            if (done()) { fc.stop(); return nullptr; }
            auto *c = new Chunk{new double[CHUNK_N], CHUNK_N, step++};
            MPI_Recv(c->data, c->size, MPI_DOUBLE,
                     src, c->tag, comm, MPI_STATUS_IGNORE);
            return c;
        })

    /* Stage 2: transform (parallel — many chunks concurrently) */
  & tbb::make_filter<Chunk *, Chunk *>(
        tbb::filter_mode::parallel,
        [&](Chunk *c) -> Chunk * {
            for (int i = 0; i < c->size; ++i) c->data[i] = f(c->data[i]);
            return c;
        })

    /* Stage 3: MPI send result (serial_in_order — preserve order) */
  & tbb::make_filter<Chunk *, void>(
        tbb::filter_mode::serial_in_order,
        [&](Chunk *c) {
            MPI_Send(c->data, c->size, MPI_DOUBLE, dst, c->tag, comm);
            delete[] c->data;
            delete c;
        })
);
```

| Filter mode | Meaning |
|---|---|
| `serial_in_order` | One item at a time; items processed in input order |
| `serial_out_of_order` | One item at a time; order not preserved |
| `parallel` | Many items concurrently; order not preserved |

---

## 37.10 Scalable Memory Allocation

TBB's `tbbmalloc` is a thread-local pool allocator that eliminates the global
heap lock and reduces false sharing between threads.

### cache_aligned_allocator

Guarantees alignment to `hardware_destructive_interference_size` (typically 64
bytes), preventing false sharing between adjacent elements accessed by different
threads:

```cpp
#include <tbb/cache_aligned_allocator.h>

/* Each element on its own cache line — no false sharing across threads */
std::vector<double, tbb::cache_aligned_allocator<double>> per_thread_buf(NTHREADS);

tbb::parallel_for(0, NTHREADS, [&](int t) {
    per_thread_buf[t] = compute_thread_result(t);   /* no false sharing */
});

double total = std::accumulate(per_thread_buf.begin(), per_thread_buf.end(), 0.0);
MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
```

### scalable_allocator

Drop-in replacement for `std::allocator` that uses TBB's scalable pool:

```cpp
#include <tbb/scalable_allocator.h>

/* High-allocation-rate container using TBB's pool allocator */
std::vector<double, tbb::scalable_allocator<double>> buf(N);

/* Or use scalable_malloc/free directly */
double *raw = static_cast<double *>(scalable_malloc(N * sizeof(double)));
/* ... */
scalable_free(raw);
```

---

## 37.11 Worked Example: MPI+TBB 2D Stencil

This example combines the key patterns: NUMA-pinned arenas, affinity partitioner
for cache reuse, enumerable thread-specific storage for per-step error, and MPI
halo exchange on the main thread.

```cpp
#include <mpi.h>
#include <tbb/tbb.h>
#include <vector>
#include <cmath>

int main(int argc, char *argv[])
{
    /* MPI_THREAD_FUNNELED: TBB threads do computation only */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ── Topology ── */
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);

    int north, south, east, west;
    MPI_Cart_shift(cart, 0, 1, &north, &south);
    MPI_Cart_shift(cart, 1, 1, &west,  &east);

    /* ── TBB: pin to local NUMA node ── */
    auto numa_nodes = tbb::info::numa_nodes();
    tbb::task_arena arena(
        tbb::task_arena::constraints{}
            .set_numa_id(numa_nodes[rank % numa_nodes.size()])
            .set_max_concurrency(
                tbb::info::default_concurrency(
                    tbb::task_arena::constraints{}.set_numa_id(
                        numa_nodes[rank % numa_nodes.size()]))));

    const int LR = LOCAL_ROWS, LC = LOCAL_COLS;

    /* Allocate grid with halos; cache-aligned for TBB threads */
    using CaAlloc = tbb::cache_aligned_allocator<double>;
    std::vector<double, CaAlloc> grid((LR+2) * (LC+2), 0.0);
    std::vector<double, CaAlloc> new_grid((LR+2) * (LC+2), 0.0);

    auto idx = [&](int r, int c){ return r * (LC+2) + c; };

    /* Affinity partitioner: same thread handles same rows each step */
    tbb::affinity_partitioner ap;

    /* Per-thread error accumulator */
    tbb::enumerable_thread_specific<double> thread_err(0.0);

    const double TOLERANCE = 1e-6;
    double global_err = std::numeric_limits<double>::max();

    for (int step = 0; step < MAX_STEPS && global_err > TOLERANCE; ++step) {

        /* ── Phase 1: parallel interior stencil ── */
        arena.execute([&]{
            for (auto &e : thread_err) e = 0.0;   /* reset per-thread error */

            tbb::parallel_for(
                tbb::blocked_range2d<int>(1, LR+1, 1, LC+1),
                [&](const tbb::blocked_range2d<int> &r) {
                    double &local_err = thread_err.local();
                    for (int i = r.rows().begin(); i < r.rows().end(); ++i)
                    for (int j = r.cols().begin(); j < r.cols().end(); ++j) {
                        new_grid[idx(i,j)] = 0.25 * (
                            grid[idx(i-1,j)] + grid[idx(i+1,j)] +
                            grid[idx(i,j-1)] + grid[idx(i,j+1)]);
                        double diff = std::abs(new_grid[idx(i,j)] - grid[idx(i,j)]);
                        local_err = std::max(local_err, diff);
                    }
                },
                ap);   /* affinity partitioner: preserves thread-range affinity */
        });

        /* Combine per-thread errors */
        double local_err = thread_err.combine([](double a, double b){
            return std::max(a, b);
        });

        /* Swap grids */
        std::swap(grid, new_grid);

        /* ── Phase 2: MPI halo exchange (main thread only) ── */
        MPI_Request reqs[8];
        int nreqs = 0;

        /* Send/recv top and bottom rows */
        if (north != MPI_PROC_NULL) {
            MPI_Irecv(&grid[idx(0,1)],    LC, MPI_DOUBLE, north, 0, cart, &reqs[nreqs++]);
            MPI_Isend(&grid[idx(1,1)],    LC, MPI_DOUBLE, north, 1, cart, &reqs[nreqs++]);
        }
        if (south != MPI_PROC_NULL) {
            MPI_Irecv(&grid[idx(LR+1,1)], LC, MPI_DOUBLE, south, 1, cart, &reqs[nreqs++]);
            MPI_Isend(&grid[idx(LR,1)],   LC, MPI_DOUBLE, south, 0, cart, &reqs[nreqs++]);
        }

        /* Column halos via vector datatype */
        MPI_Datatype col_t;
        MPI_Type_vector(LR, 1, LC+2, MPI_DOUBLE, &col_t);
        MPI_Type_commit(&col_t);

        if (west != MPI_PROC_NULL) {
            MPI_Irecv(&grid[idx(1,0)],    1, col_t, west, 2, cart, &reqs[nreqs++]);
            MPI_Isend(&grid[idx(1,1)],    1, col_t, west, 3, cart, &reqs[nreqs++]);
        }
        if (east != MPI_PROC_NULL) {
            MPI_Irecv(&grid[idx(1,LC+1)], 1, col_t, east, 3, cart, &reqs[nreqs++]);
            MPI_Isend(&grid[idx(1,LC)],   1, col_t, east, 2, cart, &reqs[nreqs++]);
        }

        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
        MPI_Type_free(&col_t);

        /* ── Phase 3: global convergence check ── */
        MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
}
```

Key design decisions in this example:

- **`MPI_THREAD_FUNNELED`**: TBB threads touch only `grid`/`new_grid`; the main
  thread has exclusive ownership of all MPI calls. No `MPI_THREAD_MULTIPLE` overhead.
- **NUMA-pinned arena**: TBB threads allocated near the local DRAM bank; cross-socket
  bandwidth is not wasted on intra-rank computation.
- **`affinity_partitioner`**: each step assigns the same row ranges to the same threads,
  keeping those cache lines warm across iterations.
- **`enumerable_thread_specific`**: per-thread error accumulation with no atomics; a
  single `combine` pass merges before the MPI reduction.
- **Overlap**: halo exchange and convergence check happen while TBB threads are idle
  (no wasted parallelism — the overlap is sequential within the rank, not concurrent).

---

## Summary

| TBB Feature | C++26 Equivalent | Use with MPI |
|---|---|---|
| `concurrent_hash_map` | None | Thread-safe result collection before MPI reduction |
| `concurrent_vector` | None | Lock-free accumulation; linearise before `MPI_Gather` |
| `concurrent_queue` | None | Producer (MPI recv) / consumer (TBB workers) decoupling |
| `task_arena` | Basic `thread_pool` only | NUMA binding; P/E core steering; per-rank isolation |
| `affinity_partitioner` | None | Cache reuse across iterative stencil steps |
| `enumerable_thread_specific` | None (only `thread_local`) | Per-thread accumulators; combine before MPI collective |
| `flow_graph` | Lower-level Senders only | Dataflow DAGs spanning MPI recv + compute + MPI send |
| `parallel_pipeline` | None | Staged MPI recv → transform → MPI send with back-pressure |
| `cache_aligned_allocator` | None | False-sharing prevention for per-thread arrays |
| `scalable_allocator` | None | High-throughput allocation in parallel regions |

**Rules**:
- Use `MPI_THREAD_FUNNELED` when TBB threads do not call MPI — it has the lowest overhead
- Pin each MPI rank's TBB arena to its NUMA node via `task_arena::constraints`
- Limit global TBB concurrency with `tbb::global_control` to avoid over-subscribing when multiple ranks share a node
- Use `affinity_partitioner` (same object across steps) for iterative codes — it is the single highest-impact TBB tuning for stencil computations
- Prefer `enumerable_thread_specific` over `std::atomic` for per-thread accumulation — no contention, no memory ordering overhead

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
