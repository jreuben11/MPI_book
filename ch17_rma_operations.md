# Chapter 17: RMA Operations

## 17.1 Basic Put, Get, Accumulate

RMA operations address remote memory by a **displacement** from the window's base
address. Displacements are measured in units of `disp_unit` (set at window creation).

All RMA operations must occur within a **synchronization epoch** (Chapter 18).

### MPI_Put — Write to Remote Memory

```c
int MPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp,
            int target_count, MPI_Datatype target_datatype,
            MPI_Win win);
```

| Argument | Meaning |
|---|---|
| `origin_addr` | Pointer to local data to send |
| `origin_count`, `origin_datatype` | Type of local data |
| `target_rank` | Rank of remote process |
| `target_disp` | Displacement in remote window (in `disp_unit` units) |
| `target_count`, `target_datatype` | Type of remote memory |
| `win` | Window handle |

```c
/* Write value 42.0 to element 5 of rank 3's window */
double val = 42.0;
MPI_Put(&val, 1, MPI_DOUBLE,
        3,            /* target rank */
        5,            /* target displacement (element 5) */
        1, MPI_DOUBLE,
        win);
```

`MPI_Put` returns immediately — the data may not have arrived yet. Completion
requires a synchronization call (Chapter 18).

### MPI_Get — Read from Remote Memory

```c
int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp,
            int target_count, MPI_Datatype target_datatype,
            MPI_Win win);
```

```c
/* Read 10 doubles starting at displacement 20 from rank 2's window */
double buf[10];
MPI_Get(buf, 10, MPI_DOUBLE,
        2,    /* target rank */
        20,   /* displacement */
        10, MPI_DOUBLE,
        win);
/* buf contents are NOT valid until after synchronization */
```

### MPI_Accumulate — Atomic Remote Update

```c
int MPI_Accumulate(const void *origin_addr, int origin_count,
                   MPI_Datatype origin_datatype,
                   int target_rank, MPI_Aint target_disp,
                   int target_count, MPI_Datatype target_datatype,
                   MPI_Op op, MPI_Win win);
```

Atomically combines local data with remote data using `op`. Equivalent to:
`remote[target_disp] = op(remote[target_disp], local_value)`.

```c
/* Add local_count to rank 0's counter at displacement 0 */
int local_count = compute_local_count();
MPI_Accumulate(&local_count, 1, MPI_INT,
               0,   /* target rank 0 */
               0,   /* displacement 0 */
               1, MPI_INT, MPI_SUM, win);
```

`MPI_Accumulate` is atomic with respect to other `MPI_Accumulate` calls on the same
location (they do not interleave). It is the correct way to implement distributed
counters or histograms without locks.

---

## 17.2 Extended Operations

### MPI_Get_accumulate — Fetch-then-Update

```c
int MPI_Get_accumulate(const void *origin_addr, int origin_count,
                       MPI_Datatype origin_datatype,
                       void *result_addr, int result_count,
                       MPI_Datatype result_datatype,
                       int target_rank, MPI_Aint target_disp,
                       int target_count, MPI_Datatype target_datatype,
                       MPI_Op op, MPI_Win win);
```

Atomically reads the old value from remote memory into `result_addr`, then applies
`op`. The sequence (read-then-update) is atomic — no other RMA operation can
interpose.

```c
/* Fetch-and-add: atomically read counter, then add 1 */
int old_val, increment = 1;
MPI_Get_accumulate(&increment, 1, MPI_INT,   /* origin: value to add */
                   &old_val,   1, MPI_INT,   /* result: old remote value */
                   0,           /* target rank */
                   0,           /* displacement */
                   1, MPI_INT, MPI_SUM, win);
/* After synchronization: old_val contains the value BEFORE the increment */
```

`MPI_Get_accumulate` with `op = MPI_NO_OP` is a non-destructive atomic read
(MPI_FETCH_AND_OP shortcut — see below).

### MPI_Fetch_and_op — Optimized Scalar Atomic

```c
int MPI_Fetch_and_op(const void *origin_addr, void *result_addr,
                     MPI_Datatype datatype,
                     int target_rank, MPI_Aint target_disp,
                     MPI_Op op, MPI_Win win);
```

A simplified version of `MPI_Get_accumulate` restricted to scalar operations (one
element, same type for origin and result). Implementations can map this directly to
hardware atomic operations.

```c
/* Atomic fetch-and-add on a single integer */
int old, inc = 1;
MPI_Fetch_and_op(&inc, &old, MPI_INT, target_rank, 0, MPI_SUM, win);
```

Common operations: `MPI_SUM` (fetch-and-add), `MPI_REPLACE` (swap), `MPI_NO_OP`
(atomic read).

### MPI_Compare_and_swap

```c
int MPI_Compare_and_swap(const void *origin_addr, const void *compare_addr,
                          void *result_addr, MPI_Datatype datatype,
                          int target_rank, MPI_Aint target_disp,
                          MPI_Win win);
```

Atomic compare-and-swap (CAS). If the remote value equals `*compare_addr`, replaces
it with `*origin_addr`. Always writes the old remote value into `*result_addr`.

```c
/* Lock-free state machine: transition from STATE_IDLE to STATE_BUSY */
int expected = STATE_IDLE;
int new_val  = STATE_BUSY;
int old_val;

MPI_Compare_and_swap(&new_val, &expected, &old_val, MPI_INT,
                     owner_rank, state_disp, win);
/* After sync: if old_val == STATE_IDLE, the CAS succeeded */
```

`MPI_Compare_and_swap` enables implementing distributed lock-free algorithms. The
semantics are identical to C11 `atomic_compare_exchange_strong`.

---

## 17.3 Request-Based Operations

All basic RMA operations have request-based counterparts that return immediately and
allow explicit completion tracking:

| Blocking | Request-Based |
|---|---|
| `MPI_Put` | `MPI_Rput` |
| `MPI_Get` | `MPI_Rget` |
| `MPI_Accumulate` | `MPI_Raccumulate` |
| `MPI_Get_accumulate` | `MPI_Rget_accumulate` |

```c
MPI_Request put_req;
MPI_Rput(&val, 1, MPI_DOUBLE, target, disp, 1, MPI_DOUBLE, win, &put_req);

do_other_work();

MPI_Wait(&put_req, MPI_STATUS_IGNORE);
/* Now the put has been initiated (but not necessarily completed at target) */
/* Full completion still requires epoch synchronization */
```

Note: `MPI_Wait` on a request-based RMA call completes the local operation (buffer
is safe to reuse), but does not guarantee remote completion. Remote completion still
requires the synchronization calls in Chapter 18.

---

## 17.4 Derived Datatypes with RMA

Like point-to-point, RMA operations support derived datatypes for non-contiguous data:

```c
/* Strided get: read every 4th element from remote window */
MPI_Datatype strided;
MPI_Type_vector(N, 1, 4, MPI_DOUBLE, &strided);
MPI_Type_commit(&strided);

double result_buf[N];
MPI_Get(result_buf, N, MPI_DOUBLE,   /* local: contiguous */
        target_rank, 0,              /* remote: starts at disp 0 */
        1, strided,                  /* remote: strided type */
        win);

MPI_Type_free(&strided);
```

The origin datatype and target datatype can differ, as long as the total bytes
transferred match.

---

## 17.5 Performance Notes

### Origin vs. Target Bandwidth

RMA operations consume resources on both the origin (initiates) and the target (provides
memory). On passive-target synchronization, the target truly does no CPU work.
On active-target, both sides participate in synchronization.

### Granularity

Many small puts/gets are much more expensive than one large put/get (per-operation
overhead on RDMA hardware). Batch small operations into one large transfer where
possible:

```c
/* Inefficient: N separate puts */
for (int i = 0; i < N; i++)
    MPI_Put(&vals[i], 1, MPI_DOUBLE, target, i, 1, MPI_DOUBLE, win);

/* Efficient: one put of N elements */
MPI_Put(vals, N, MPI_DOUBLE, target, 0, N, MPI_DOUBLE, win);
```

### Contiguous vs. Derived

Hardware RDMA typically requires contiguous buffers. Derived datatypes may cause
extra memcpy operations at the origin or target. Profile if non-contiguous RMA
is on the critical path.

---

## Summary

| Function | Purpose |
|---|---|
| `MPI_Put` | Write local data to remote window |
| `MPI_Get` | Read remote window data to local buffer |
| `MPI_Accumulate` | Atomic remote update with `op` |
| `MPI_Get_accumulate` | Atomic fetch-then-update |
| `MPI_Fetch_and_op` | Scalar atomic operation (hardware-mappable) |
| `MPI_Compare_and_swap` | Atomic CAS for distributed lock-free algorithms |
| `MPI_Rput/Rget/Raccumulate` | Request-based variants; buffer-safe after `MPI_Wait` |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
