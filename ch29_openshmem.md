# Chapter 29: OpenSHMEM

## 29.1 What OpenSHMEM Is

OpenSHMEM is a partitioned global address space (PGAS) library API for parallel
programming. Like MPI RMA (Chapters 16–18), it provides one-sided communication —
but it is architecturally simpler: every object placed in the **symmetric heap** is
automatically accessible by all PEs (processing elements, OpenSHMEM's term for ranks).

Key differences from MPI:

| | MPI RMA | OpenSHMEM |
|---|---|---|
| Memory registration | Explicit (`MPI_Win_create`) | Automatic (symmetric heap) |
| Addressing | Displacement from window base | Pointer to symmetric variable |
| Synchronization | Fence / lock / PSCW | `shmem_fence`, `shmem_quiet`, `shmem_barrier_all` |
| Standard body | MPI Forum | OpenSHMEM consortium |
| Primary implementations | Open MPI, MPICH | Sandia OpenSHMEM, Open MPI, Cray SHMEM, OSSS |
| C++ support | C API from C++ | C API from C++ |

OpenSHMEM is defined by the OpenSHMEM specification (current: 1.5). It is available
as a standalone library and is built into Open MPI (`oshmem` library).

---

## 29.2 Initialization and Shutdown

```c
#include <shmem.h>

int main(void)
{
    shmem_init();

    int me   = shmem_my_pe();    /* my PE number (like MPI rank) */
    int npes = shmem_n_pes();    /* total PE count */

    printf("PE %d of %d\n", me, npes);

    shmem_finalize();
    return 0;
}
```

Compile and run:

```bash
oshcc -O2 -o hello hello.c      # OpenSHMEM C wrapper
oshrun -n 4 ./hello             # launcher (or: srun, mpiexec, depending on installation)
```

### Thread-Safe Initialization

```c
int provided;
shmem_init_thread(SHMEM_THREAD_MULTIPLE, &provided);
if (provided < SHMEM_THREAD_MULTIPLE) {
    fprintf(stderr, "Need SHMEM_THREAD_MULTIPLE\n");
    shmem_global_exit(1);
}
```

Thread levels mirror MPI: `SHMEM_THREAD_SINGLE`, `SHMEM_THREAD_FUNNELED`,
`SHMEM_THREAD_SERIALIZED`, `SHMEM_THREAD_MULTIPLE`.

---

## 29.3 The Symmetric Heap

The symmetric heap is shared memory accessible by all PEs. Any variable allocated
on the symmetric heap on PE `i` can be read or written by any other PE using that
address.

```c
/* Allocate on symmetric heap */
double *shared_array = shmem_malloc(N * sizeof(double));

/* Use like regular memory locally */
for (int i = 0; i < N; i++) shared_array[i] = (double)(shmem_my_pe() * N + i);

shmem_barrier_all();   /* ensure all PEs have initialized */

/* PE 0 reads PE 1's shared_array[0] */
if (shmem_my_pe() == 0) {
    double remote_val;
    shmem_double_get(&remote_val, shared_array, 1, 1);
    printf("PE 1's first value: %f\n", remote_val);
}

shmem_free(shared_array);
```

`shmem_malloc` is collective — all PEs must call it simultaneously, and each PE
allocates the same number of bytes. The allocations have the same **symmetric**
relationship: the virtual address of `shared_array` on PE `i` is the same as the
virtual address on PE `j` in terms of symmetric offset (though not necessarily
the same absolute virtual address on all platforms).

### Static Symmetric Variables

Global and static variables are automatically symmetric in OpenSHMEM:

```c
/* This variable is accessible from any PE */
static long counter;
static double result_buf[1024];

int main(void)
{
    shmem_init();
    counter = shmem_my_pe();   /* initialize local copy */
    shmem_barrier_all();
    /* PE 0 can read any PE's counter directly */
    shmem_finalize();
}
```

---

## 29.4 Put and Get Operations

### shmem_put — Write to Remote PE

```c
/* General form: TYPE can be double, float, int, long, etc. */
void shmem_TYPE_put(TYPE *dest, const TYPE *src, size_t nelems, int pe);

/* Typed examples */
double local[N], remote_dst[N];
shmem_double_put(remote_dst, local, N, target_pe);
/* dest = pointer to target's symmetric variable */
/* src  = pointer to local (any) variable        */
```

Put is non-blocking at the initiator — data may not have arrived at `target_pe`
when the call returns. Call `shmem_quiet()` to ensure all puts have completed.

```c
shmem_double_put(target_buf, src_buf, N, target_pe);
shmem_quiet();   /* all puts to all PEs are now complete */
```

### shmem_get — Read from Remote PE

```c
void shmem_TYPE_get(TYPE *dest, const TYPE *src, size_t nelems, int pe);

double remote_val[N];
shmem_double_get(remote_val, remote_sym_ptr, N, source_pe);
/* get is blocking — remote_val is valid when the call returns */
```

Unlike `shmem_put`, `shmem_get` is **blocking**: the data is present in `dest`
when the function returns.

### Non-Blocking Put and Get

```c
shmem_ctx_t ctx;
shmem_ctx_create(SHMEM_CTX_NOSTORE, &ctx);

/* Non-blocking put */
shmem_ctx_double_put_nbi(ctx, remote_dest, src, N, target_pe);

/* Do other work */
compute_local();

/* Wait for non-blocking operations to complete */
shmem_ctx_quiet(ctx);
shmem_ctx_destroy(ctx);
```

**Context** (`shmem_ctx_t`) is OpenSHMEM's equivalent of MPI's request-based
operations — operations posted on a context are completed by `shmem_ctx_quiet`.

---

## 29.5 Synchronization

### shmem_barrier_all

```c
shmem_barrier_all();
/* All PEs have reached this point; all prior puts are visible everywhere */
```

`shmem_barrier_all` provides both point synchronization and memory ordering.
It implies a `shmem_quiet` — all pending puts are complete before the barrier returns.

### shmem_fence

```c
shmem_fence();
/* Ensures ordering: all puts before fence are visible before puts after fence */
/* Does NOT block until puts complete — only orders them */
```

Use `shmem_fence` to enforce ordering without full synchronization.

### shmem_quiet

```c
shmem_quiet();
/* All pending puts on the default context are complete (at remote PEs) */
/* No process-level synchronization — other PEs may not have read the data yet */
```

`shmem_quiet` is completion, not synchronization. After `shmem_quiet`, the puts
have been delivered to remote memory, but the remote PE has not necessarily seen them.
To ensure the remote PE sees the data, coordinate with a barrier or atomics.

---

## 29.6 Atomic Operations

OpenSHMEM provides atomic operations on symmetric variables:

```c
/* Atomic fetch-and-add */
long old = shmem_long_atomic_fetch_add(&remote_counter, 1L, target_pe);

/* Atomic fetch-and-set (swap) */
long prev = shmem_long_atomic_swap(&remote_var, new_value, target_pe);

/* Atomic compare-and-swap */
long expected = 0;
long success = shmem_long_atomic_compare_swap(&remote_lock, expected, 1L, target_pe);
/* Returns old value; if == expected, swap succeeded */

/* Atomic add (no return) */
shmem_long_atomic_add(&remote_counter, increment, target_pe);

/* Atomic inc */
shmem_long_atomic_inc(&remote_counter, target_pe);
```

These map to hardware atomic operations on RDMA-capable interconnects.

### Distributed Counter Pattern

```c
/* Atomic work queue: PEs fetch work items from PE 0's counter */
static long work_index;   /* symmetric; initialized on PE 0 */

if (shmem_my_pe() == 0) work_index = 0;
shmem_barrier_all();

while (1) {
    long my_item = shmem_long_atomic_fetch_add(&work_index, 1L, 0);
    if (my_item >= TOTAL_WORK_ITEMS) break;
    process_item(my_item);
}
shmem_barrier_all();
```

---

## 29.7 Collective Operations

OpenSHMEM 1.5 provides collectives mirroring MPI's:

```c
/* Active set: PEs 0..npes-1 with stride 1 */
shmem_team_t team = SHMEM_TEAM_WORLD;

/* Broadcast: PE 0 sends to all */
double buf[N];
shmem_double_broadcast(team, buf, buf, N, 0);

/* Reduce */
double local_val = compute_local();
double global_result;
shmem_double_sum_reduce(team, &global_result, &local_val, 1);

/* Allreduce (to_all variants) */
long src[N], dst[N];
shmem_long_sum_reduce(team, dst, src, N);
```

OpenSHMEM 1.5 introduced **teams** (`shmem_team_t`) analogous to MPI communicators:

```c
/* Split world into even/odd PE subsets using shmem_team_split_strided.
   Parameters: (parent, start_pe, stride, size, config, config_mask, new_team)
   Unlike MPI_Comm_split, there is no "color" — specify explicit start+stride. */
/* shmem_team_split_strided is collective — ALL PEs must call with identical args.
   Call once per sub-team to create complementary splits. */
int me = shmem_my_pe(), npes = shmem_n_pes();
shmem_team_t even_team, odd_team;

/* All PEs call: even PE team (start=0, stride=2, size=npes/2) */
shmem_team_split_strided(SHMEM_TEAM_WORLD, 0, 2, npes / 2, NULL, 0, &even_team);
/* All PEs call: odd PE team (start=1, stride=2, size=npes/2) */
shmem_team_split_strided(SHMEM_TEAM_WORLD, 1, 2, npes / 2, NULL, 0, &odd_team);

/* Each PE uses its own team and destroys the other */
shmem_team_t my_team = (me % 2 == 0) ? even_team : odd_team;
shmem_team_t other_team = (me % 2 == 0) ? odd_team : even_team;

if (my_team != SHMEM_TEAM_INVALID) {
    double val = 1.0;
    shmem_double_sum_reduce(my_team, &val, &val, 1);
    shmem_team_destroy(my_team);
}
if (other_team != SHMEM_TEAM_INVALID)
    shmem_team_destroy(other_team);
```

---

## 29.8 Wait Operations

OpenSHMEM provides point-to-point synchronization using memory polling:

```c
/* Wait until a symmetric variable reaches a condition */
static volatile long signal_flag;

/* PE 0 sets flag to signal PE 1 */
if (shmem_my_pe() == 0) {
    do_work();
    shmem_long_p(&signal_flag, 1L, 1);   /* remote store of value 1 to PE 1 */
    shmem_quiet();
}

/* PE 1 waits for flag */
if (shmem_my_pe() == 1) {
    shmem_long_wait_until(&signal_flag, SHMEM_CMP_EQ, 1L);
    /* flag is now 1 — PE 0 has completed its work */
    process_pe0_result();
}
```

`shmem_long_wait_until` efficiently polls the local memory location (which the
network updates via RDMA) without repeatedly sending probe messages.

---

## 29.9 MPI + OpenSHMEM Interoperability

On supported platforms (Open MPI built with `--with-oshmem`), MPI and OpenSHMEM
can coexist in the same program:

```c
#include <mpi.h>
#include <shmem.h>

int main(int argc, char *argv[])
{
    /* Initialize both */
    MPI_Init(&argc, &argv);
    shmem_init();

    int mpi_rank, shmem_pe;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    shmem_pe = shmem_my_pe();
    /* mpi_rank == shmem_pe on compatible implementations */

    /* Use MPI for collectives (richer feature set) */
    double val;
    MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Use OpenSHMEM for fine-grained one-sided access (simpler API than MPI RMA) */
    static double shmem_buf[N];
    shmem_double_get(local_buf, shmem_buf, N, target_pe);

    shmem_finalize();
    MPI_Finalize();
}
```

Interoperability rules (implementation-dependent):
- Do not call MPI collectives from a context where an OpenSHMEM epoch is active.
- OpenSHMEM symmetric variables are separate from MPI windows.
- PMIx bootstrap is shared; do not double-initialize.

---

## 29.10 When to Use OpenSHMEM vs. MPI RMA

| Scenario | Recommendation |
|---|---|
| Already using MPI; need one-sided | Use MPI RMA (`MPI_Win_*`) — no new dependency |
| New code; simple one-sided patterns | OpenSHMEM — simpler API, less boilerplate |
| Need rich collectives (Alltoallv, etc.) | MPI — more complete collective library |
| Distributed atomic counters / locks | Either; OpenSHMEM atomics are slightly simpler |
| GPU-attached memory | MPI (GPU-aware); OpenSHMEM GPU support varies |
| Existing Cray SHMEM / NVSHMEM code | Port to OpenSHMEM 1.5 for portability |

---

## Summary

| Function | Purpose |
|---|---|
| `shmem_init` / `shmem_finalize` | Initialize / shut down |
| `shmem_my_pe` / `shmem_n_pes` | Identity (rank / size) |
| `shmem_malloc` / `shmem_free` | Symmetric heap allocation |
| `shmem_TYPE_put` | Non-blocking write to remote PE |
| `shmem_TYPE_get` | Blocking read from remote PE |
| `shmem_quiet` | Complete all pending puts |
| `shmem_fence` | Order puts without waiting |
| `shmem_barrier_all` | Full barrier + quiet |
| `shmem_long_atomic_fetch_add` | Atomic fetch-and-add |
| `shmem_long_wait_until` | Spin-wait on symmetric variable |
| `shmem_double_sum_reduce` | Collective reduction via team |
| `SHMEM_TEAM_WORLD` | Default team (all PEs) |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
