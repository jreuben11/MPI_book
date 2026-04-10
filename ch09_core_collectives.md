# Chapter 9: Core Collectives

## 9.1 What Collectives Are

A **collective** operation involves all processes in a communicator. Every process
must call the same collective function (in the same order), though they may take
different roles (root vs. non-root, or all-equal). Collectives cannot complete until
all processes in the communicator have called them.

Collectives are not implemented as a series of point-to-point messages in your code —
MPI implements them internally using algorithms tuned to the network topology and
message size. Prefer collectives over hand-rolled reductions or broadcasts.

---

## 9.2 MPI_Barrier

```c
int MPI_Barrier(MPI_Comm comm);
```

All processes block until every process in `comm` has called `MPI_Barrier`. After
return, all processes have passed the barrier.

```c
/* Ensure all processes have written their output before rank 0 prints summary */
printf("Rank %d: local_result = %f\n", rank, local_result);
MPI_Barrier(MPI_COMM_WORLD);
if (rank == 0) printf("--- All ranks reported ---\n");
```

**Warning**: `MPI_Barrier` synchronizes — it does not transmit data. It is often
misused as a performance fix for race conditions. In most cases, the correct
collective for your operation (`MPI_Reduce`, `MPI_Allreduce`) already provides the
necessary synchronization as a side effect.

`MPI_Barrier` has measurable cost on large process counts. Use it sparingly.

---

## 9.3 MPI_Bcast — Broadcast

```c
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm);
```

Root sends `buffer` to all other processes. All non-root processes receive into the
same `buffer` argument. The `root` rank must be consistent across all processes.

```c
int config[10];

if (rank == 0) {
    /* Only rank 0 initializes the data */
    read_config(config);
}

/* All ranks call Bcast — root=0 sends, others receive */
MPI_Bcast(config, 10, MPI_INT, 0, MPI_COMM_WORLD);

/* All ranks now have identical config[] */
```

The buffer on non-root processes before `MPI_Bcast` returns is overwritten. Do not
initialize it to a meaningful value expecting it to be preserved.

---

## 9.4 MPI_Scatter and MPI_Gather

### MPI_Scatter — one-to-all distribution

```c
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm);
```

Root distributes **equal-sized chunks** to each process. Process `i` receives
elements `[i*sendcount .. (i+1)*sendcount - 1]` from root's `sendbuf`.

```c
int global[400]; /* on root: 400 elements for 4 processes × 100 each */
int local[100];

if (rank == 0) fill_data(global, 400);

MPI_Scatter(global, 100, MPI_INT,   /* root sends 100 ints per process */
            local,  100, MPI_INT,   /* each process receives 100 ints  */
            0, MPI_COMM_WORLD);

/* Each rank now has its 100-element slice in local[] */
```

Non-root processes can pass any value for `sendbuf` and `sendcount` — they are ignored.
Passing `NULL` for sendbuf on non-root is conventional and safe.

### MPI_Gather — all-to-one collection

```c
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm);
```

Each process sends its data to root. Root assembles them in rank order.

```c
int local_result[100];
int global_results[400]; /* only used on root */

compute(local_result, 100);

MPI_Gather(local_result, 100, MPI_INT,
           global_results, 100, MPI_INT,   /* recvcount = count per sender */
           0, MPI_COMM_WORLD);

if (rank == 0) process_results(global_results, 400);
```

---

## 9.5 MPI_Allgather

```c
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm);
```

Like `MPI_Gather`, but all processes receive the result — not just root. Every
process ends up with the full assembled array.

```c
int local_size = local_data_count(rank);
int all_sizes[size]; /* size = number of processes */

MPI_Allgather(&local_size, 1, MPI_INT,
              all_sizes,   1, MPI_INT,
              MPI_COMM_WORLD);

/* all_sizes[i] now contains the local count for process i, on every rank */
/* Useful for computing displacements for MPI_Allgatherv */
```

---

## 9.6 MPI_Reduce and MPI_Allreduce

### MPI_Reduce — reduction to root

```c
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
```

Combines each process's `sendbuf` using operation `op`, accumulating into root's
`recvbuf`. Element-wise: `recvbuf[i] = op(sendbuf[i] from rank 0, rank 1, ..., rank N-1)`.

```c
double local_sum = compute_local_sum();
double global_sum;

MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE,
           MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0) printf("Global sum: %f\n", global_sum);
```

### MPI_Allreduce — reduction to all

```c
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
```

Same as `MPI_Reduce` but all processes receive the result. No `root` parameter.

```c
double local_error = compute_local_error();
double global_max_error;

MPI_Allreduce(&local_error, &global_max_error, 1, MPI_DOUBLE,
              MPI_MAX, MPI_COMM_WORLD);

if (global_max_error < TOLERANCE) break; /* all ranks agree on convergence */
```

`MPI_Allreduce` is one of the most frequently used MPI functions. It is the standard
pattern for convergence checks in iterative solvers.

### Predefined Reduction Operations

| Operation | Meaning |
|---|---|
| `MPI_SUM` | Sum |
| `MPI_PROD` | Product |
| `MPI_MAX` | Maximum |
| `MPI_MIN` | Minimum |
| `MPI_MAXLOC` | Maximum and its location (rank) |
| `MPI_MINLOC` | Minimum and its location (rank) |
| `MPI_LAND` | Logical AND |
| `MPI_LOR` | Logical OR |
| `MPI_LXOR` | Logical XOR |
| `MPI_BAND` | Bitwise AND |
| `MPI_BOR` | Bitwise OR |
| `MPI_BXOR` | Bitwise XOR |

### MPI_MAXLOC and MPI_MINLOC

These require a special "value-index" datatype:

```c
/* Find the global maximum and which rank holds it */
struct { double val; int rank; } local_max, global_max;

local_max.val  = compute_local_max();
local_max.rank = rank;

MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT,
              MPI_MAXLOC, MPI_COMM_WORLD);

printf("Global max = %f on rank %d\n", global_max.val, global_max.rank);
```

Predefined types for `MAXLOC`/`MINLOC`: `MPI_FLOAT_INT`, `MPI_DOUBLE_INT`,
`MPI_LONG_INT`, `MPI_2INT`, `MPI_SHORT_INT`, `MPI_LONG_DOUBLE_INT`.

---

## 9.7 MPI_Alltoall

```c
int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm);
```

Every process sends a distinct chunk to every other process. The result is a
"transpose" of the data: process `j` receives from process `i` into
`recvbuf[i * recvcount]`.

```c
/* Matrix transpose across processes */
/* Each rank holds a row of the matrix; after Alltoall, each holds a column */
int sendbuf[size * N];   /* N elements to send to each of `size` processes */
int recvbuf[size * N];

fill_row(sendbuf, rank, N);

MPI_Alltoall(sendbuf, N, MPI_INT,
             recvbuf, N, MPI_INT,
             MPI_COMM_WORLD);
```

`MPI_Alltoall` is expensive: it performs O(P²) message transfers. Use it only when
the all-to-all communication is genuinely required by the algorithm.

---

## 9.8 In-Place Collectives

Many collectives support `MPI_IN_PLACE` to eliminate an extra buffer:

```c
/* Allreduce in place */
double val = compute_local_value();
MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, comm);
/* val now holds the global sum */

/* Reduce in place — only root passes MPI_IN_PLACE as sendbuf */
double result;
if (rank == 0) result = compute_local_value();
else           result = compute_local_value(); /* ignored on non-root */

MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &result,
           &result, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
/* On root: result holds global sum; on others: result is unmodified */
```

---

## 9.9 Collective Correctness Rules

1. **All processes must call the collective.** If any process skips it, the program
   will hang or produce wrong results. No exceptions.

2. **The root argument must be the same on all processes.**

3. **The count and datatype must be consistent** across all processes (send and recv
   counts may differ for scatter/gather, but each sender's count must match root's
   expectation for that sender).

4. **Collectives on `MPI_COMM_WORLD` cannot be interleaved with point-to-point
   messages using the same communicator** in a way that could deadlock. Use separate
   communicators for independent streams of communication.

5. **Collective calls on the same communicator are ordered**: they complete in the
   same logical order across all processes. Two different collectives on the same
   communicator cannot overlap.

---

## 9.10 Worked Example: Parallel Sum with Verification

```c
/* parallel_sum.c — distribute array, compute partial sums, reduce */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000 * size;   /* total elements, divisible by size */
    const int local_n = N / size;

    double *global = NULL;
    if (rank == 0) {
        global = malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) global[i] = (double)(i + 1);
    }

    /* Distribute */
    double *local = malloc(local_n * sizeof(double));
    MPI_Scatter(global, local_n, MPI_DOUBLE,
                local,  local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Local sum */
    double local_sum = 0.0;
    for (int i = 0; i < local_n; i++) local_sum += local[i];

    /* Global sum */
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        double expected = (double)N * (N + 1) / 2.0;
        printf("Computed: %.0f  Expected: %.0f  %s\n",
               global_sum, expected,
               global_sum == expected ? "PASS" : "FAIL");
        free(global);
    }

    free(local);
    MPI_Finalize();
    return 0;
}
```

---

## Summary

| Function | Pattern |
|---|---|
| `MPI_Barrier` | Synchronize all processes; no data movement |
| `MPI_Bcast` | Root → all: broadcast a buffer |
| `MPI_Scatter` | Root → each: distribute equal chunks |
| `MPI_Gather` | Each → root: collect in rank order |
| `MPI_Allgather` | Each → all: everyone sees everyone's data |
| `MPI_Reduce` | Each → root: element-wise reduction |
| `MPI_Allreduce` | Each → all: reduction result on every rank |
| `MPI_Alltoall` | Each ↔ each: personalized exchange |
| `MPI_IN_PLACE` | Eliminate extra buffer on root or all ranks |
