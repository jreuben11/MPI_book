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

The canonical use of non-blocking communication is a 1D or multi-dimensional halo
(ghost cell) exchange, used in stencil computations:

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
  either way, but posting Irecv before Isend is the standard convention.
- **Overlap interior computation**: the interior cells do not depend on halo data, so
  they can be computed while halo exchange is in progress.
- **Use `MPI_REQUEST_NULL`** for boundary ranks that have no left or right neighbor.

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
