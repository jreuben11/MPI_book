# Chapter 7: Send Modes

## 7.1 Overview of Send Variants

MPI provides four send modes. They differ in when `MPI_Send` returns and what
guarantees they make about synchronization with the receiver.

| Mode | Function | Returns when... | Receiver must be ready? |
|---|---|---|---|
| Standard | `MPI_Send` | Buffer safe to reuse | No |
| Buffered | `MPI_Bsend` | Buffer safe to reuse | No |
| Synchronous | `MPI_Ssend` | Receiver has started receiving | No, but blocks until matched |
| Ready | `MPI_Rsend` | Buffer safe to reuse | Yes — undefined behavior otherwise |

Each has a non-blocking counterpart: `MPI_Ibsend`, `MPI_Issend`, `MPI_Irsend`.

---

## 7.2 Standard Mode: MPI_Send

```c
MPI_Send(buf, count, datatype, dest, tag, comm);
```

Standard mode is implementation-defined: MPI may either buffer the data internally
(eager protocol) or wait for a rendezvous with the receiver (rendezvous protocol).
The choice is made based on message size and internal thresholds.

- For small messages: MPI copies to an internal buffer, returns immediately.
- For large messages: MPI blocks until the receiver posts a matching `MPI_Recv`.

**Write code that is correct regardless of which protocol is used.** Do not depend on
the eager behavior for messages above a few hundred bytes.

---

## 7.3 Buffered Mode: MPI_Bsend

Buffered mode forces MPI to use a user-supplied buffer, guaranteeing immediate return
regardless of message size. The caller provides the buffer explicitly.

### Attaching and Detaching a Buffer

```c
int bufsize = 10 * (1024 * 1024 + MPI_BSEND_OVERHEAD); /* 10 × 1 MB messages */
void *bsend_buf = malloc(bufsize);

MPI_Buffer_attach(bsend_buf, bufsize);

/* Now MPI_Bsend copies into bsend_buf and returns immediately */
MPI_Bsend(data, count, MPI_DOUBLE, dest, tag, comm);

/* Detach: blocks until all buffered sends complete, then returns buffer */
MPI_Buffer_detach(&bsend_buf, &bufsize);
free(bsend_buf);
```

`MPI_BSEND_OVERHEAD` is the per-message overhead MPI reserves in your buffer for
internal bookkeeping. Always add it to your size calculation.

Only one buffer may be attached at a time. `MPI_Buffer_detach` blocks until all
pending buffered sends are complete.

### When to Use Buffered Mode

Buffered mode is useful when:
- You need guaranteed non-blocking behavior for large messages.
- You are writing a library that cannot control the receiver's readiness.
- Profiling shows that standard sends stall and the buffering cost is acceptable.

In practice, most codes use non-blocking `MPI_Isend` instead of `MPI_Bsend`. The
non-blocking approach gives more control over progress and avoids the fixed buffer
allocation.

---

## 7.4 Synchronous Mode: MPI_Ssend

```c
MPI_Ssend(buf, count, datatype, dest, tag, comm);
```

Synchronous send blocks until the receiver has **posted a matching receive** and the
data transfer has begun. It does not return until the communication is acknowledged
by the receiver side.

Properties:
- `MPI_Ssend` always blocks until the receiver is ready, regardless of message size.
- Completing `MPI_Ssend` guarantees the receiver has called `MPI_Recv` (or equivalent).
- This makes `MPI_Ssend` useful for debugging: it turns potential deadlocks caused by
  buffering into guaranteed deadlocks, making them reproducible.

```c
/* Debugging tool: replace MPI_Send with MPI_Ssend to expose hidden deadlocks */
/* The deadlock pattern in Chapter 5 deadlocks with MPI_Ssend for all message sizes */
if (rank == 0) {
    MPI_Ssend(buf, N, MPI_INT, 1, 0, comm);  /* always blocks */
    MPI_Recv(buf, N, MPI_INT, 1, 0, comm, MPI_STATUS_IGNORE);
} else {
    MPI_Ssend(buf, N, MPI_INT, 0, 0, comm);  /* always blocks */
    MPI_Recv(buf, N, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
}
/* Deadlock occurs here regardless of message size — good for testing */
```

**Use `MPI_Ssend` for debugging, not in production code.** The added synchronization
reduces performance and introduces unnecessary coupling between sender and receiver.

---

## 7.5 Ready Mode: MPI_Rsend

```c
MPI_Rsend(buf, count, datatype, dest, tag, comm);
```

Ready mode informs MPI that the receiver has already posted a matching `MPI_Recv`.
If this is true, MPI can skip the rendezvous handshake and send immediately, which
can improve performance on some networks.

**If the receive has NOT been posted when `MPI_Rsend` is called, the behavior is
undefined.** The program may crash, corrupt data, or silently produce wrong results.

```c
/* Correct usage: receiver posts Irecv first, then signals sender */
if (rank == 1) {
    MPI_Irecv(buf, N, MPI_DOUBLE, 0, 0, comm, &req);
    MPI_Send(NULL, 0, MPI_BYTE, 0, READY_TAG, comm); /* signal: ready */
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

if (rank == 0) {
    MPI_Recv(NULL, 0, MPI_BYTE, 1, READY_TAG, comm, MPI_STATUS_IGNORE);
    MPI_Rsend(buf, N, MPI_DOUBLE, 1, 0, comm); /* safe: recv is posted */
}
```

**Avoid `MPI_Rsend` in new code.** The protocol required to safely establish the
"ready" condition almost always eliminates the performance gain. Modern MPI
implementations provide similar optimizations automatically when using standard mode.

---

## 7.6 Non-Blocking Variants

All four send modes have non-blocking equivalents:

| Blocking | Non-Blocking |
|---|---|
| `MPI_Send` | `MPI_Isend` |
| `MPI_Bsend` | `MPI_Ibsend` |
| `MPI_Ssend` | `MPI_Issend` |
| `MPI_Rsend` | `MPI_Irsend` |

These follow the same semantics as their blocking counterparts, but return immediately
with an `MPI_Request`. Complete with `MPI_Wait` or `MPI_Test`.

```c
MPI_Request req;
MPI_Issend(buf, N, MPI_DOUBLE, dest, tag, comm, &req);
/* ... overlap computation ... */
MPI_Wait(&req, MPI_STATUS_IGNORE);
/* Wait completes only after receiver has started receiving */
```

---

## 7.7 Choosing the Right Mode

```
Use MPI_Send (standard) for all production code.
Use MPI_Isend for overlap patterns.
Use MPI_Ssend temporarily when debugging suspected deadlocks.
Avoid MPI_Bsend and MPI_Rsend in new code.
```

The performance difference between standard, buffered, and synchronous modes is
usually dominated by network latency and message size effects — not by the mode
choice itself. Write for correctness first; tune only with profiler data.

---

## 7.8 Summary Table

| Property | Send | Bsend | Ssend | Rsend |
|---|---|---|---|---|
| Blocks waiting for receiver? | Maybe (large) | No | Yes | No |
| Requires user buffer? | No | Yes | No | No |
| Receiver must pre-post? | No | No | No | **Yes** |
| Safe from deadlock? | Depends on size | Yes | No | Depends on protocol |
| Recommended for production? | Yes | Rarely | No (debug only) | No |
