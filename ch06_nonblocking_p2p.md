# Chapter 6: Non-Blocking Communication

## 6.1 Why Non-Blocking?

Blocking `MPI_Send` and `MPI_Recv` force the calling process to wait. When a process
could be doing useful computation while waiting for communication to complete, blocking
calls waste CPU cycles.

Non-blocking operations split communication into two phases:
1. **Initiation**: post the send or receive. Returns immediately with a `MPI_Request`.
2. **Completion**: wait (or test) until the operation finishes.

Between initiation and completion, the process can do other work — typically
computation that does not touch the communication buffers.

```
Timeline (blocking):         Timeline (non-blocking):
 Send|==waiting==|Compute     Isend|Compute........|Wait
                                    ^ overlap ^
```

The overlap is only useful if the computation is substantial and the network can
progress independently. Whether true overlap occurs depends on the MPI implementation
and network hardware (requires async progress — see Chapter 25).

---

## 6.2 MPI_Isend and MPI_Irecv

```c
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request);

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request);
```

Both return immediately with a `MPI_Request` handle. The `request` tracks the
in-progress operation.

```c
double sendbuf[N], recvbuf[N];
MPI_Request send_req, recv_req;

/* Post both operations */
MPI_Isend(sendbuf, N, MPI_DOUBLE, dest,   tag, MPI_COMM_WORLD, &send_req);
MPI_Irecv(recvbuf, N, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &recv_req);

/* Overlap: do work that does not touch sendbuf or recvbuf */
compute_interior(interior_data, N);

/* Wait for both to complete */
MPI_Wait(&send_req, MPI_STATUS_IGNORE);
MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

/* Now sendbuf and recvbuf are safe to use */
```

**Critical rule**: do not read or write `buf` between `MPI_Isend`/`MPI_Irecv` and
the corresponding `MPI_Wait`/`MPI_Test`. The buffer is "owned" by MPI until the
operation completes.

---

## 6.3 Completion Functions

### MPI_Wait

```c
int MPI_Wait(MPI_Request *request, MPI_Status *status);
```

Blocks until the request is complete. Sets `*request = MPI_REQUEST_NULL` on return.
Pass `MPI_STATUS_IGNORE` if you do not need the status.

### MPI_Test

```c
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
```

Non-blocking check. Returns immediately. `*flag = 1` if complete, `0` if still in
progress. If complete, sets `*request = MPI_REQUEST_NULL`.

```c
int flag;
while (1) {
    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
    if (flag) break;
    do_some_small_work();
}
```

`MPI_Test` may be used to drive progress in some implementations — calling it
periodically can advance communication even in the polling thread.

### MPI_Waitall

```c
int MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[]);
```

Waits for all `count` requests. On return, all are complete and all requests are
set to `MPI_REQUEST_NULL`. Pass `MPI_STATUSES_IGNORE` if statuses are not needed.

```c
MPI_Request reqs[4];
MPI_Irecv(buf0, N, MPI_DOUBLE, 0, 0, comm, &reqs[0]);
MPI_Irecv(buf1, N, MPI_DOUBLE, 1, 0, comm, &reqs[1]);
MPI_Isend(sbuf0, N, MPI_DOUBLE, 2, 0, comm, &reqs[2]);
MPI_Isend(sbuf1, N, MPI_DOUBLE, 3, 0, comm, &reqs[3]);

do_interior_computation();

MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
```

### MPI_Waitany

```c
int MPI_Waitany(int count, MPI_Request requests[], int *index,
                MPI_Status *status);
```

Waits until at least one of the `count` requests completes. `*index` gives the
index of the completed request. Use when you want to process results as they arrive
rather than waiting for all:

```c
while (pending_count > 0) {
    int idx;
    MPI_Status status;
    MPI_Waitany(pending_count, reqs, &idx, &status);
    process_result(results[idx]);
    /* compact the arrays or track completion */
    pending_count--;
}
```

### MPI_Waitsome

```c
int MPI_Waitsome(int incount, MPI_Request requests[],
                 int *outcount, int indices[], MPI_Status statuses[]);
```

Waits until at least one request completes, then returns all currently complete
requests. `*outcount` gives how many completed; `indices[]` gives their positions.

```c
int outcount;
int indices[N];
MPI_Status statuses[N];
MPI_Waitsome(N, reqs, &outcount, indices, statuses);
for (int i = 0; i < outcount; i++) {
    process_result(results[indices[i]]);
}
```

`MPI_Waitsome` is more efficient than calling `MPI_Waitany` in a loop when you want
to process all currently-available results without waiting for new ones.

### MPI_Testall, MPI_Testany, MPI_Testsome

Non-blocking equivalents of the above — they return immediately rather than blocking.
Useful for polling in a progress loop.

---

## 6.4 Request Management

### MPI_REQUEST_NULL

A null request (`MPI_REQUEST_NULL`) is a valid value that represents "no pending
operation." `MPI_Wait` on `MPI_REQUEST_NULL` returns immediately. `MPI_Waitall`
with some elements set to `MPI_REQUEST_NULL` skips those positions.

This simplifies code where some processes have no work in a particular direction:

```c
MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

if (rank > 0)           /* receive from left neighbor */
    MPI_Irecv(lbuf, N, MPI_DOUBLE, rank-1, 0, comm, &reqs[0]);
if (rank < size-1)      /* receive from right neighbor */
    MPI_Irecv(rbuf, N, MPI_DOUBLE, rank+1, 0, comm, &reqs[1]);

MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
/* reqs[0] or reqs[1] may be MPI_REQUEST_NULL — Waitall handles it */
```

### MPI_Request_free

```c
int MPI_Request_free(MPI_Request *request);
```

Marks a request for deallocation when it completes. The request handle is set to
`MPI_REQUEST_NULL` immediately. **Use with caution**: you lose the ability to wait on
the request, so you cannot know when the buffer is safe to reuse.

```c
/* Correct use: "fire and forget" when buffer lifetime is managed elsewhere */
MPI_Isend(static_buf, N, MPI_DOUBLE, dest, tag, comm, &req);
MPI_Request_free(&req);
/* Do NOT reuse static_buf until you are certain the send completed */
```

For most code, `MPI_Request_free` is unnecessary. Use `MPI_Wait` instead.

---

## 6.5 Halo Exchange Pattern

### Stencil Computations

A **stencil** updates every point in a grid from a fixed neighborhood of surrounding
points. The update rule is applied at every grid point, repeatedly over many time
steps or solver iterations. Common stencils:

| Stencil | Dimensions | Points used | Typical application |
|---|---|---|---|
| 3-point | 1D | center ± 1 | 1D diffusion, tridiagonal solvers |
| 5-point (star) | 2D | N/S/E/W + center | 2D Laplace, heat equation |
| 9-point (box) | 2D | all 8 neighbors + center | Isotropic diffusion |
| 7-point | 3D | 6 faces + center | 3D heat, Poisson equation |
| 27-point | 3D | all face/edge/corner + center | Higher-order 3D solvers |

A **4th-order** stencil uses a radius of 2 (±2 neighbors per axis) instead of 1,
improving accuracy at the cost of a wider neighborhood dependency.

A minimal 1D heat equation stencil:

```c
/* One time step of explicit 1D heat diffusion */
for (int i = 1; i < N-1; i++)
    u_new[i] = u[i] + dt * (u[i-1] - 2*u[i] + u[i+1]) / (dx*dx);
```

This is representative of all stencil codes: each updated cell depends only on a
fixed set of neighbors, the neighborhood radius is small and uniform, and the same
rule applies at every interior point. Iterative solvers (Jacobi, Gauss-Seidel,
conjugate gradient) use this structure at their core.

### Domain Decomposition

To parallelize a stencil, the grid is partitioned among MPI ranks. Each rank owns a
contiguous sub-domain and is solely responsible for updating the points within it.

**1D decomposition** (slabs): each rank gets a contiguous block of rows or columns.
Simple to implement; each rank has at most 2 neighbors.

```
Global array (16 elements, 4 ranks):
  Rank 0        Rank 1        Rank 2        Rank 3
 [0  1  2  3] [4  5  6  7] [8  9 10 11] [12 13 14 15]
```

**2D decomposition** (tiles): for a 2D grid, each rank owns a rectangular tile.
Each rank has up to 4 neighbors (8 with diagonals for 9-point stencils). The key
advantage is a better **surface-to-volume ratio** — the amount of data exchanged
per step grows as O(N/√P) instead of O(N):

```
8×8 global grid, 4 ranks (2×2 tile layout):

  ┌─────────┬─────────┐
  │  Rank 0 │  Rank 1 │
  │  4×4    │  4×4    │
  ├─────────┼─────────┤
  │  Rank 2 │  Rank 3 │
  │  4×4    │  4×4    │
  └─────────┴─────────┘
```

**3D decomposition** (sub-volumes): for 3D simulations (CFD, molecular dynamics,
climate models), each rank gets a 3D brick with up to 6 face neighbors.

As a rule: use the decomposition that most closely matches the dimensionality of
your problem. More neighbors means more messages but fewer bytes per message.

### Ghost Cells (Halos)

When a rank applies a stencil near the edge of its sub-domain, it needs values from
cells owned by a neighboring rank. Those values are pre-fetched and stored in a
layer of extra storage called **ghost cells** (also called **halo cells** or
**overlap cells**) surrounding the rank's local array.

```
Rank 1's local array with halo width = 1:

  ┌──────┬───────────────────────────────┬──────┐
  │ghost │   owned interior cells        │ghost │
  │ ← from Rank 0              from Rank 2 →   │
  └──────┴───────────────────────────────┴──────┘
   index 0   index 1 ... index local_N   index local_N+1
```

The **halo width** equals the stencil radius:
- 3-point or 5-point stencil (radius 1): 1 ghost layer per side
- 4th-order stencil (radius 2): 2 ghost layers per side

After each time step, each rank's interior changes, so its neighbors' ghost copies
become stale. The **halo exchange** refreshes them:

1. Send this rank's boundary cells to each neighbor.
2. Receive neighbors' boundary cells into local ghost slots.
3. Apply stencil to **interior** cells (no ghost data needed) — this can overlap with steps 1–2.
4. Wait for the exchange to complete.
5. Apply stencil to **boundary** cells (which depend on the now-fresh ghost data).

This split between interior and boundary work is the key optimization: posting
non-blocking sends/receives, then computing the interior, hides most or all of
the communication latency.

### Implementation — 1D Halo Exchange

```c
/* 1D domain decomposition — each rank owns a slab of a 1D array */
/* halos: rank exchanges boundary cells with left and right neighbors */

double *local = malloc((local_N + 2) * sizeof(double)); /* +2 for halos */
/* local[0] = left halo, local[local_N+1] = right halo */
/* local[1..local_N] = interior data */

MPI_Request reqs[4];
int nreqs = 0;

/* Post receives first — always post receives before sends */
if (rank > 0)
    MPI_Irecv(&local[0],         1, MPI_DOUBLE, rank-1, 0, comm, &reqs[nreqs++]);
if (rank < size-1)
    MPI_Irecv(&local[local_N+1], 1, MPI_DOUBLE, rank+1, 1, comm, &reqs[nreqs++]);

/* Then post sends */
if (rank > 0)
    MPI_Isend(&local[1],       1, MPI_DOUBLE, rank-1, 1, comm, &reqs[nreqs++]);
if (rank < size-1)
    MPI_Isend(&local[local_N], 1, MPI_DOUBLE, rank+1, 0, comm, &reqs[nreqs++]);

/* Overlap: compute interior stencil (indices 2..local_N-1) */
compute_interior_stencil(local, local_N);

/* Wait for halos */
MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

/* Apply boundary stencil (indices 1 and local_N) */
compute_boundary_stencil(local, local_N);
```

Key points in this pattern:
- **Post receives before sends**: avoids a race where the sender's buffer is consumed
  before the receiver's buffer is ready. Most implementations handle this correctly
  either way, but posting `MPI_Irecv` before `MPI_Isend` is the standard convention.
- **Overlap interior computation**: interior cells (indices 2 through local_N−1) do not
  depend on ghost data, so the stencil over them can run concurrently with the exchange.
  Only the boundary cells (indices 1 and local_N) need to wait for the halos to arrive.
- **Use `MPI_REQUEST_NULL`** for boundary ranks (rank 0 has no left neighbor; rank
  size−1 has no right neighbor). `MPI_Waitall` silently skips null requests.
- **Halo width**: this example uses width 1 (a single ghost cell each side), matching
  a 3-point stencil. A 4th-order stencil would require width 2 — send two boundary
  cells, allocate two ghost slots, and shift the index arithmetic accordingly.

---

## 6.6 Common Mistakes

### Mistake 1: Touching the Buffer Too Early

```c
double buf[N];
MPI_Isend(buf, N, MPI_DOUBLE, dest, 0, comm, &req);
for (int i = 0; i < N; i++) buf[i] = 0.0;  /* BUG: buf still owned by MPI */
MPI_Wait(&req, MPI_STATUS_IGNORE);
```

Fix: wait before modifying.

### Mistake 2: Losing the Request Handle

```c
for (int i = 0; i < 10; i++) {
    MPI_Request req;
    MPI_Isend(bufs[i], N, MPI_DOUBLE, i, 0, comm, &req);
    /* req goes out of scope — LEAKED */
}
```

Fix: store all requests in an array and `MPI_Waitall` at the end.

### Mistake 3: Reusing a Request Before Completion

```c
MPI_Request req;
MPI_Isend(buf, N, MPI_DOUBLE, dest, 0, comm, &req);
MPI_Isend(buf, N, MPI_DOUBLE, dest, 1, comm, &req);  /* BUG: overwrites req */
```

Fix: use separate request handles, or wait before reposting.

### Mistake 4: Assuming Test Drives Progress

Some implementations require polling for communication to progress:

```c
/* Might stall on some implementations without polling */
MPI_Isend(buf, N, MPI_DOUBLE, dest, 0, comm, &req);
compute_for_a_long_time();
MPI_Wait(&req, MPI_STATUS_IGNORE);  /* fine on most, but not guaranteed to overlap */
```

If overlap is critical, use an async progress thread (`MPICH_ASYNC_PROGRESS=1`) or
call `MPI_Test` periodically to drive progress manually.

---

## 6.7 Request Arrays and Dynamic Patterns

When the number of communications is dynamic, manage requests with a resizable array:

```c
/* C99 flexible array with manual management */
int nreqs = 0, cap = 16;
MPI_Request *reqs = malloc(cap * sizeof(MPI_Request));

void add_request(MPI_Request **reqs, int *nreqs, int *cap, MPI_Request r) {
    if (*nreqs == *cap) {
        *cap *= 2;
        *reqs = realloc(*reqs, *cap * sizeof(MPI_Request));
    }
    (*reqs)[(*nreqs)++] = r;
}

/* ... post sends and receives dynamically ... */

MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
free(reqs);
```

In C++, `std::vector<MPI_Request>` works well. Pass `requests.data()` to MPI functions.

---

## Summary

| Function | Behavior |
|---|---|
| `MPI_Isend` | Initiate send; returns `MPI_Request` immediately |
| `MPI_Irecv` | Initiate receive; returns `MPI_Request` immediately |
| `MPI_Wait` | Block until one request completes |
| `MPI_Test` | Non-blocking check; returns `flag` |
| `MPI_Waitall` | Block until all N requests complete |
| `MPI_Waitany` | Block until any one request completes |
| `MPI_Waitsome` | Block until at least one completes; returns all done |
| `MPI_REQUEST_NULL` | Placeholder for "no operation"; safe to Wait on |
| `MPI_Request_free` | Detach request; buffer unsafe until complete |

**Rules of thumb**:
- Post Irecv before Isend
- Never touch buffers between Isend/Irecv and the Wait
- Store all request handles; never lose them
- Use Waitall for simple cases; Waitany/Waitsome for server-style patterns

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
