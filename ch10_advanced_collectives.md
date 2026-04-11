# Chapter 10: Advanced Collectives

## 10.1 Vector Variants

The core collectives in Chapter 9 require equal-sized chunks for each process.
The **vector variants** (suffix `v`) allow each process to send or receive a
different number of elements, specified through arrays of counts and displacements.

---

## 10.2 MPI_Scatterv

```c
int MPI_Scatterv(
    const void *sendbuf,
    const int sendcounts[],   /* how many elements to send to each rank */
    const int displs[],       /* offset in sendbuf for each rank's data */
    MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype,
    int root, MPI_Comm comm);
```

```c
/* Distribute a non-uniformly partitioned array */
int data_size = 1000;
int *global_data = NULL;

int sendcounts[size], displs[size];

if (rank == 0) {
    global_data = malloc(data_size * sizeof(int));
    /* ... fill global_data ... */

    /* Compute per-rank counts (unequal partition) */
    int base = data_size / size;
    int rem  = data_size % size;
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base + (i < rem ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }
}

/* Each rank receives its own count */
int my_count;
MPI_Scatter(sendcounts, 1, MPI_INT, &my_count, 1, MPI_INT, 0, comm);

int *local = malloc(my_count * sizeof(int));

MPI_Scatterv(global_data, sendcounts, displs, MPI_INT,
             local, my_count, MPI_INT,
             0, MPI_COMM_WORLD);
```

---

## 10.3 MPI_Gatherv

```c
int MPI_Gatherv(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf,
    const int recvcounts[],   /* how many elements to receive from each rank */
    const int displs[],       /* where to place each rank's data in recvbuf */
    MPI_Datatype recvtype,
    int root, MPI_Comm comm);
```

Root must pre-allocate `recvbuf` large enough for all incoming data and provide
`recvcounts[]` and `displs[]`. These arrays must be consistent with what each
rank actually sends.

```c
/* Gather variable-length results from each rank */
int local_count = compute_local_result_count();
int *local_results = malloc(local_count * sizeof(int));
/* ... compute ... */

/* Collect counts on root */
int *all_counts = (rank == 0) ? malloc(size * sizeof(int)) : NULL;
MPI_Gather(&local_count, 1, MPI_INT,
           all_counts, 1, MPI_INT,
           0, MPI_COMM_WORLD);

int *recvbuf = NULL;
int *displs_arr = NULL;
int total = 0;

if (rank == 0) {
    displs_arr = malloc(size * sizeof(int));
    displs_arr[0] = 0;
    for (int i = 1; i < size; i++)
        displs_arr[i] = displs_arr[i-1] + all_counts[i-1];
    total = displs_arr[size-1] + all_counts[size-1];
    recvbuf = malloc(total * sizeof(int));
}

MPI_Gatherv(local_results, local_count, MPI_INT,
            recvbuf, all_counts, displs_arr, MPI_INT,
            0, MPI_COMM_WORLD);
```

---

## 10.4 MPI_Allgatherv

```c
int MPI_Allgatherv(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf,
    const int recvcounts[],
    const int displs[],
    MPI_Datatype recvtype,
    MPI_Comm comm);
```

All processes receive the assembled result. All processes must provide the
`recvcounts` and `displs` arrays (not just root). Commonly used after `MPI_Allgather`
of counts to pre-compute the displacement array.

```c
/* Pattern: gather counts, compute displs, then Allgatherv the data */
int local_n = compute_local_count(rank);

int all_n[size];
MPI_Allgather(&local_n, 1, MPI_INT, all_n, 1, MPI_INT, comm);

int displs[size];
displs[0] = 0;
for (int i = 1; i < size; i++) displs[i] = displs[i-1] + all_n[i-1];
int total_n = displs[size-1] + all_n[size-1];

double *local_data = malloc(local_n * sizeof(double));
double *all_data   = malloc(total_n * sizeof(double));
/* ... fill local_data ... */

MPI_Allgatherv(local_data, local_n, MPI_DOUBLE,
               all_data,   all_n,   displs, MPI_DOUBLE, comm);
/* all_data now holds the concatenated data from all ranks, on every rank */
```

---

## 10.5 MPI_Alltoallv and MPI_Alltoallw

### MPI_Alltoallv

Variable-count version of `MPI_Alltoall`. Each process sends a different number of
elements to each other process.

```c
int MPI_Alltoallv(
    const void *sendbuf, const int sendcounts[], const int sdispls[],
    MPI_Datatype sendtype,
    void *recvbuf,       const int recvcounts[], const int rdispls[],
    MPI_Datatype recvtype,
    MPI_Comm comm);
```

The send counts, send displacements, receive counts, and receive displacements are
all arrays of length `size`. This is the most general fixed-type all-to-all.

### MPI_Alltoallw

The most general collective: different type, count, and displacement per destination:

```c
int MPI_Alltoallw(
    const void *sendbuf,
    const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[],
    void *recvbuf,
    const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[],
    MPI_Comm comm);
```

`MPI_Alltoallw` is the "escape hatch" for heterogeneous all-to-all patterns.

**Important:** Unlike `MPI_Alltoallv` where displacements (`sdispls`, `rdispls`) are
in units of the corresponding datatype's extent, `MPI_Alltoallw` displacements are
always in **bytes**. This is a frequent source of bugs when porting from `MPI_Alltoallv`.

Its complexity means it is rarely needed directly — derived datatypes (Chapter 13) often
provide a cleaner way to express the same pattern.

---

## 10.6 MPI_Scan and MPI_Exscan

### MPI_Scan — inclusive prefix reduction

```c
int MPI_Scan(const void *sendbuf, void *recvbuf, int count,
             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
```

Process `k` receives the reduction of values from ranks 0 through `k` inclusive:

```
rank 0:  recv = v[0]
rank 1:  recv = op(v[0], v[1])
rank 2:  recv = op(v[0], v[1], v[2])
...
rank N-1: recv = op(v[0], ..., v[N-1])  = same as MPI_Reduce result
```

```c
/* Compute global offsets for variable-length data */
int local_count = compute_local_count();
int prefix_sum;

MPI_Scan(&local_count, &prefix_sum, 1, MPI_INT, MPI_SUM, comm);
/* prefix_sum = sum of local_count for ranks 0..rank (inclusive) */
/* offset for writing this rank's data = prefix_sum - local_count */
int my_offset = prefix_sum - local_count;
```

### MPI_Exscan — exclusive prefix reduction

```c
int MPI_Exscan(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
```

Process `k` receives the reduction of values from ranks 0 through `k-1` (exclusive):

```
rank 0:  recv = undefined (MPI_Exscan does not define rank 0's result)
rank 1:  recv = v[0]
rank 2:  recv = op(v[0], v[1])
...
```

`MPI_Exscan` is more convenient than `MPI_Scan` when you need the offset before
your own contribution:

```c
int local_n = compute_local_count();
int offset = 0;

if (size > 1) {
    MPI_Exscan(&local_n, &offset, 1, MPI_INT, MPI_SUM, comm);
    /* rank 0: recvbuf value is undefined by the MPI standard — rely on the
       pre-initialized offset = 0, not on any implementation-specific behavior */
}
/* offset = starting position for this rank's data in the global array */
```

---

## 10.7 MPI_Reduce_scatter and MPI_Reduce_scatter_block

### MPI_Reduce_scatter_block

Combines a reduction with a scatter of equal-sized chunks. Equivalent to
`MPI_Reduce` followed by `MPI_Scatter`, but potentially more efficient:

```c
int MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf,
                              int recvcount, MPI_Datatype datatype,
                              MPI_Op op, MPI_Comm comm);
```

Each process contributes `recvcount * size` elements; receives a `recvcount`-element
slice of the reduction result.

### MPI_Reduce_scatter

Variable-count version:

```c
int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf,
                        const int recvcounts[], MPI_Datatype datatype,
                        MPI_Op op, MPI_Comm comm);
```

Used in distributed sparse linear algebra where the reduction result is naturally
partitioned unevenly.

---

## 10.8 Custom Reduction Operations

When predefined operations (`MPI_SUM`, `MPI_MAX`, etc.) are insufficient, define
your own:

```c
/* Reduce struct: simultaneously find min value and sum of weights */
typedef struct { double min_val; double weight_sum; } Stat;

void stat_reduce(void *invec_raw, void *inoutvec_raw, int *len,
                 MPI_Datatype *dtype)
{
    Stat *in    = (Stat *)invec_raw;
    Stat *inout = (Stat *)inoutvec_raw;
    for (int i = 0; i < *len; i++) {
        if (in[i].min_val < inout[i].min_val)
            inout[i].min_val = in[i].min_val;
        inout[i].weight_sum += in[i].weight_sum;
    }
}
```

Registering and using the custom operation:

```c
MPI_Op stat_op;
MPI_Op_create(stat_reduce, 1 /* commutative */, &stat_op);

/* Create a matching derived datatype for the struct (Chapter 13) */
MPI_Datatype stat_type;
MPI_Type_contiguous(2, MPI_DOUBLE, &stat_type);
MPI_Type_commit(&stat_type);

Stat local_stat = { compute_local_min(), compute_local_weight() };
Stat global_stat;
MPI_Allreduce(&local_stat, &global_stat, 1, stat_type, stat_op, comm);

MPI_Op_free(&stat_op);
MPI_Type_free(&stat_type);
```

Rules for custom operations:
- The function signature must match exactly.
- `commutative = 1` tells MPI the operation is commutative (may reorder operands).
  Set to 0 if order matters (e.g., matrix multiplication).
- The function must be associative for correctness across different tree orderings.
- The function is called multiple times on subsets of data; it must be pure (no
  side effects that depend on call order).

---

## Summary

| Function | Pattern |
|---|---|
| `MPI_Scatterv` | Root → each: variable-size distribution |
| `MPI_Gatherv` | Each → root: variable-size collection |
| `MPI_Allgatherv` | Each → all: variable-size, all receive |
| `MPI_Alltoallv` | Each ↔ each: variable-count personalized exchange |
| `MPI_Alltoallw` | Each ↔ each: variable type+count exchange |
| `MPI_Scan` | Inclusive prefix reduction |
| `MPI_Exscan` | Exclusive prefix reduction |
| `MPI_Reduce_scatter_block` | Reduce + scatter (equal chunks) |
| `MPI_Reduce_scatter` | Reduce + scatter (variable chunks) |
| `MPI_Op_create` | Register a user-defined reduction function |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
