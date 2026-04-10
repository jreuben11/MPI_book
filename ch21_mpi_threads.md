# Chapter 21: MPI + Threads

## 21.1 Why Hybrid MPI+Threads?

Pure MPI programs use one MPI process per core. On modern multi-core nodes (64–128
cores per node), this creates practical problems:

- **Memory overhead**: each MPI process has its own copy of data structures, page
  tables, and MPI buffers. On a 128-core node, 128 separate MPI processes consume
  significantly more memory than 1 process with 128 threads.
- **Message count**: intra-node MPI communication, while implemented with shared
  memory, still has higher overhead than direct thread-level sharing via OpenMP or
  pthreads.
- **Scalability**: some algorithms have collective calls that become bottlenecks
  at very high process counts.

The hybrid model — **one MPI process per node (or per NUMA domain), with threads
within each process** — reduces inter-node message count and allows efficient
intra-node sharing.

---

## 21.2 Thread Support Levels

MPI defines four levels of thread safety, in increasing order of support:

| Level | Constant | Meaning |
|---|---|---|
| 0 | `MPI_THREAD_SINGLE` | Only one thread exists; no threading |
| 1 | `MPI_THREAD_FUNNELED` | Multiple threads, but only the main thread calls MPI |
| 2 | `MPI_THREAD_SERIALIZED` | Multiple threads, but MPI calls are serialized (one at a time) |
| 3 | `MPI_THREAD_MULTIPLE` | Multiple threads, each calling MPI concurrently |

Request the required level at initialization:

```c
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

if (provided < MPI_THREAD_MULTIPLE) {
    fprintf(stderr, "Need MPI_THREAD_MULTIPLE, got level %d\n", provided);
    MPI_Abort(MPI_COMM_WORLD, 1);
}
```

Always check `provided` — implementations are allowed to provide a lower level than
requested. If `provided < required`, your threading model must be adjusted.

Query the current thread level:

```c
int level;
MPI_Query_thread(&level);
```

Check if this is the main (funneled) thread:

```c
int is_main;
MPI_Is_thread_main(&is_main);
/* is_main == 1 if the calling thread is the one that called MPI_Init_thread */
```

---

## 21.3 MPI_THREAD_FUNNELED — Most Common

With `MPI_THREAD_FUNNELED`, only the **master thread** (the one that called
`MPI_Init_thread`) may call MPI functions. Worker threads do computation only.

This is the easiest model to implement correctly and has the lowest overhead:

```c
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

#pragma omp parallel
{
    /* Worker threads do computation */
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    compute_local_slice(data, tid, nthreads);

    /* Only master thread does MPI */
    #pragma omp master
    {
        /* Pack boundary data (done by master after barrier) */
        /* MPI communication — only the master thread */
        MPI_Allreduce(MPI_IN_PLACE, &global_val, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
    }
    #pragma omp barrier  /* other threads wait for master's MPI */
}
```

Alternatively, using `#pragma omp single` instead of `#pragma omp master` gives
an implicit barrier at the end of the single region.

---

## 21.4 MPI_THREAD_SERIALIZED

Multiple threads may call MPI, but not simultaneously. You must serialize access
with a mutex:

```c
#include <pthread.h>

pthread_mutex_t mpi_mutex = PTHREAD_MUTEX_INITIALIZER;

void thread_safe_allreduce(double *val)
{
    pthread_mutex_lock(&mpi_mutex);
    MPI_Allreduce(MPI_IN_PLACE, val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    pthread_mutex_unlock(&mpi_mutex);
}
```

`MPI_THREAD_SERIALIZED` is useful when multiple threads need to initiate
communications independently but performance is acceptable with serialization.

---

## 21.5 MPI_THREAD_MULTIPLE

The most permissive level: any thread may call any MPI function at any time.
The MPI implementation guarantees thread safety internally.

```c
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

/* Each thread sends its local result independently */
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    double local_result = compute_my_part(tid);
    int target = 0;
    int tag = tid;   /* different tags to avoid conflicts */

    MPI_Send(&local_result, 1, MPI_DOUBLE, target, tag, MPI_COMM_WORLD);
}
```

### Performance Cost of MPI_THREAD_MULTIPLE

Thread safety requires locking inside MPI. Most MPI operations under
`MPI_THREAD_MULTIPLE` are slower than under `MPI_THREAD_SINGLE`:

- **Progress locks**: the MPI progress engine must be locked when threads call
  simultaneously.
- **Message ordering**: with multiple threads sending to the same destination,
  the order of messages from different threads is non-deterministic.
- **Hot spots**: the communicator's match lists may become a contention point.

Measured overhead: 10–50% latency increase in microbenchmarks; less in real
applications where threads are not all calling MPI simultaneously.

### Avoiding Conflicts in MPI_THREAD_MULTIPLE

Use separate tags or separate communicators per thread to avoid message ordering
ambiguity:

```c
/* Per-thread communicator — duplex costs one collective but prevents future conflicts */
MPI_Comm thread_comm[MAX_THREADS];
for (int t = 0; t < nthreads; t++)
    MPI_Comm_dup(MPI_COMM_WORLD, &thread_comm[t]);

#pragma omp parallel
{
    int tid = omp_get_thread_num();
    /* Each thread uses its own communicator — no tag collisions */
    MPI_Allreduce(MPI_IN_PLACE, &my_val, 1, MPI_DOUBLE, MPI_SUM, thread_comm[tid]);
}
```

---

## 21.6 Hybrid MPI+OpenMP Patterns

### Pattern 1: Funneled — Overlap Halo Exchange with Computation

```c
/* Standard hybrid pattern: MPI between nodes, OpenMP within */

/* Step 1: pack halos (parallel) */
#pragma omp parallel for
for (int i = 0; i < halo_size; i++) halo_buf[i] = data[boundary_idx[i]];

/* Step 2: exchange halos (MPI, master thread only) */
MPI_Sendrecv(halo_buf, halo_size, MPI_DOUBLE, left_rank, 0,
             recv_halo, halo_size, MPI_DOUBLE, right_rank, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

/* Step 3: compute interior (parallel), unpack halos (serial after) */
#pragma omp parallel for
for (int i = halo_size; i < local_n - halo_size; i++) stencil(data, i);

/* Step 4: merge received halos into data (parallel) */
#pragma omp parallel for
for (int i = 0; i < halo_size; i++) data[boundary_idx2[i]] = recv_halo[i];
```

### Pattern 2: MPI Ranks as Node Masters

```c
/* One MPI rank per node; OpenMP parallelizes within the node */
/* Rank communicates on behalf of all threads */

int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

double *node_data = malloc(node_n * sizeof(double));

/* Parallel computation across all node cores */
#pragma omp parallel for
for (int i = 0; i < node_n; i++) node_data[i] = compute(i);

/* Reduction across nodes — MPI communicates between node masters */
MPI_Allreduce(MPI_IN_PLACE, node_data, node_n, MPI_DOUBLE, MPI_SUM,
              MPI_COMM_WORLD);

/* Results used by all threads on the node */
#pragma omp parallel for
for (int i = 0; i < node_n; i++) use_result(node_data[i]);
```

---

## 21.7 Common Race Conditions

### Race 1: Buffer Reuse Before MPI_Wait

```c
/* BUG: multiple threads share the same buffer */
double shared_buf[N];

#pragma omp parallel sections
{
    #pragma omp section
    {
        MPI_Isend(shared_buf, N, MPI_DOUBLE, dest, 0, comm, &req);
        /* send in progress */
    }
    #pragma omp section
    {
        for (int i = 0; i < N; i++) shared_buf[i] = new_val; /* BUG */
    }
}
MPI_Wait(&req, MPI_STATUS_IGNORE);
```

Fix: use separate buffers per thread, or ensure the send completes before the write.

### Race 2: Collective Ordering

```c
/* BUG: threads call Barrier in different orders */
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    if (tid == 0) MPI_Barrier(MPI_COMM_WORLD);
    if (tid == 1) MPI_Allreduce(...);  /* may execute before Barrier on some threads */
}
```

Fix: only one thread calls MPI in funneled/serialized mode. In multiple mode, ensure
all threads follow the same call sequence through application-level coordination.

### Race 3: Tag Conflicts Between Threads

```c
/* BUG: two threads send to the same destination with the same tag */
#pragma omp parallel sections
{
    #pragma omp section
    { MPI_Send(buf1, N, MPI_DOUBLE, 0, 42, comm); }
    #pragma omp section
    { MPI_Send(buf2, N, MPI_DOUBLE, 0, 42, comm); }
}
/* Receiver cannot distinguish which message is which */
```

Fix: use unique tags per thread or per operation.

---

## Summary

| Thread Level | Who Calls MPI | Overhead | Use Case |
|---|---|---|---|
| `MPI_THREAD_SINGLE` | One thread only | None | No threading |
| `MPI_THREAD_FUNNELED` | Main thread only | Minimal | OpenMP compute + MPI boundary exchange |
| `MPI_THREAD_SERIALIZED` | Any thread, one at a time | Mutex lock | Multiple communicating threads, low frequency |
| `MPI_THREAD_MULTIPLE` | Any thread, concurrently | Internal locking | Multi-threaded communication servers |

**Rules**:
- Always use `MPI_Init_thread`, never `MPI_Init` in threaded programs
- Always check `provided` level
- Prefer `MPI_THREAD_FUNNELED` — simpler and faster than `MPI_THREAD_MULTIPLE`
- Use per-thread communicators or unique tags to prevent message ordering ambiguity
