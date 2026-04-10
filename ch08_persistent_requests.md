# Chapter 8: Persistent Requests

## 8.1 Motivation

In many HPC applications, the same communication pattern repeats identically in every
iteration of a time-step loop: the same processes exchange data at the same message
sizes with the same tags. With ordinary non-blocking calls, MPI must re-negotiate each
communication setup on every iteration:

```c
/* Without persistent requests — setup cost paid every iteration */
for (int step = 0; step < NSTEPS; step++) {
    update_halos(data);                        /* fill halo buffers */
    /* Post receives first — always before sends to avoid potential deadlock */
    MPI_Irecv(left_recv,  N, MPI_DOUBLE, left,  0, comm, &reqs[0]);
    MPI_Irecv(right_recv, N, MPI_DOUBLE, right, 0, comm, &reqs[1]);
    MPI_Isend(left_halo,  N, MPI_DOUBLE, left,  0, comm, &reqs[2]);
    MPI_Isend(right_halo, N, MPI_DOUBLE, right, 0, comm, &reqs[3]);
    compute_interior(data);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    compute_boundary(data);
}
```

Persistent requests pre-negotiate the communication parameters once during
initialization, then reuse the same "slot" each iteration. On tight loops at small
message sizes, this can reduce latency by eliminating repeated internal bookkeeping.

---

## 8.2 Creating Persistent Requests

```c
int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype,
                  int dest, int tag, MPI_Comm comm, MPI_Request *request);

int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype,
                  int source, int tag, MPI_Comm comm, MPI_Request *request);
```

These functions create a **persistent request** bound to the given buffer, count,
type, endpoint, and tag. The request is initialized but **not yet active** — no
communication has been started.

Variants for all four send modes also exist:

| Function | Mode |
|---|---|
| `MPI_Send_init` | Standard |
| `MPI_Bsend_init` | Buffered |
| `MPI_Ssend_init` | Synchronous |
| `MPI_Rsend_init` | Ready |
| `MPI_Recv_init` | Receive |

---

## 8.3 Starting and Completing Persistent Requests

```c
/* Start a single persistent request */
int MPI_Start(MPI_Request *request);

/* Start multiple persistent requests simultaneously */
int MPI_Startall(int count, MPI_Request requests[]);
```

After `MPI_Start`, the request becomes **active**: communication is in progress.
Use `MPI_Wait` or `MPI_Test` to complete it.

After completion:
- The request becomes **inactive** again — not `MPI_REQUEST_NULL`.
- The request can be restarted with `MPI_Start` in the next iteration.
- To free the request: call `MPI_Request_free`.

---

## 8.4 Complete Example: Halo Exchange with Persistent Requests

```c
double *left_send, *right_send, *left_recv, *right_recv;
/* ... allocate buffers ... */

int left  = (rank - 1 + size) % size;
int right = (rank + 1) % size;

MPI_Request reqs[4];
MPI_Send_init(right_send, HALO, MPI_DOUBLE, right, 0, comm, &reqs[0]);
MPI_Send_init(left_send,  HALO, MPI_DOUBLE, left,  1, comm, &reqs[1]);
MPI_Recv_init(right_recv, HALO, MPI_DOUBLE, right, 1, comm, &reqs[2]);
MPI_Recv_init(left_recv,  HALO, MPI_DOUBLE, left,  0, comm, &reqs[3]);

/* Time-step loop */
for (int step = 0; step < NSTEPS; step++) {
    pack_halos(data, left_send, right_send);

    MPI_Startall(4, reqs);

    compute_interior(data);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    /* After Waitall: requests are inactive, ready to Start again */

    unpack_halos(data, left_recv, right_recv);
    compute_boundary(data);
}

/* Cleanup — free persistent requests */
for (int i = 0; i < 4; i++)
    MPI_Request_free(&reqs[i]);
```

Key differences from the non-persistent version:
- `Send_init` / `Recv_init` called once outside the loop.
- `Startall` replaces `Isend` + `Irecv` inside the loop.
- `Waitall` works identically — after return, requests are inactive, not null.
- `Request_free` called once after the loop.

---

## 8.5 Persistent Request Lifecycle

```
         Send_init / Recv_init
                |
                v
           [INACTIVE]
                |
           MPI_Start / MPI_Startall
                |
                v
            [ACTIVE]  <── communication in progress
                |
           MPI_Wait / MPI_Test (on completion)
                |
                v
           [INACTIVE]  <── can be restarted
                |
           MPI_Request_free
                |
                v
        [FREED / MPI_REQUEST_NULL]
```

A persistent request in the INACTIVE state is **not** `MPI_REQUEST_NULL`. Calling
`MPI_Wait` on an inactive persistent request is an **error** (the MPI standard
explicitly prohibits it). Only `MPI_REQUEST_NULL` is safe to pass to `MPI_Wait` as
a no-op. To safely discard a persistent request, call `MPI_Request_free`, which sets
it to `MPI_REQUEST_NULL`.
Calling `MPI_Start` on an active request is also an error.

---

## 8.6 Performance Notes

Persistent requests were designed to reduce per-message overhead by caching endpoint
information. Whether they actually improve performance depends on the implementation.

- **MPICH** and its derivatives typically show measurable improvement at small message
  sizes (below 1 KB) where per-call overhead dominates.
- **Open MPI** has historically provided less benefit from persistent requests, though
  this varies by version and transport.

Benchmark your specific implementation before committing to persistent requests as a
performance optimization. The code is more complex to reason about, so the benefit
should justify the added cognitive overhead.

For large messages where network transfer time dominates setup overhead, persistent
vs. non-persistent requests make no measurable difference.

---

## 8.7 Relationship to Partitioned Communication

MPI 4.0 introduced **partitioned communication** (`MPI_Psend_init`, `MPI_Precv_init`)
as an extension of the persistent concept. Partitioned communication allows the sender
to mark individual partitions of a message as ready independently, enabling finer
overlap with GPU or producer/consumer pipelines. This is covered in Chapter 23.

Persistent requests (this chapter) remain relevant for CPU-based patterns.
Partitioned communication targets heterogeneous systems where parts of a buffer are
filled asynchronously.

---

## Summary

| Function | Purpose |
|---|---|
| `MPI_Send_init` | Create persistent send request (inactive) |
| `MPI_Recv_init` | Create persistent receive request (inactive) |
| `MPI_Start` | Activate a single persistent request |
| `MPI_Startall` | Activate multiple persistent requests |
| `MPI_Wait` | Complete an active request; leaves it inactive (not null) |
| `MPI_Request_free` | Free a persistent request handle |

**When to use**: tight loops with a fixed communication pattern and small message
sizes where benchmarking shows measurable reduction in per-iteration overhead.
