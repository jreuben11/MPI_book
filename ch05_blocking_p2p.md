# Chapter 5: Blocking Send & Receive

## 5.1 The Point-to-Point Model

Point-to-point communication moves data from one process (the **sender**) to exactly
one other process (the **receiver**). The two processes must coordinate: the sender
names the destination rank; the receiver names the source rank (or `MPI_ANY_SOURCE`).

A message is matched using the **envelope**:
**(communicator, source rank, tag)**

The destination rank determines routing but is not part of the matching envelope —
it is implicit (the calling receiver process). When multiple messages with the same
envelope are in flight, MPI guarantees they arrive in FIFO order. This is the
**non-overtaking guarantee**: messages from the same source on the same communicator
arrive in the order they were sent, regardless of tag.

---

## 5.2 MPI_Send and MPI_Recv

```c
int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm);

int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status);
```

| Argument | Meaning |
|---|---|
| `buf` | Pointer to send/recv buffer |
| `count` | Number of elements of the given datatype |
| `datatype` | Element type descriptor (see Chapter 12) |
| `dest` / `source` | Rank in `comm`; `MPI_ANY_SOURCE` for recv |
| `tag` | Message label; `MPI_ANY_TAG` for recv |
| `comm` | Communicator |
| `status` | (recv only) Filled with source, tag, error; `MPI_STATUS_IGNORE` to skip |

### A Simple Exchange

```c
int data[100];

if (rank == 0) {
    /* Initialize and send */
    for (int i = 0; i < 100; i++) data[i] = i;
    MPI_Send(data, 100, MPI_INT, 1, 0, MPI_COMM_WORLD);

} else if (rank == 1) {
    /* Receive and use */
    MPI_Recv(data, 100, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Received data[99] = %d\n", data[99]);
}
```

### What "Blocking" Means

`MPI_Send` is blocking in the sense that it does not return until the send buffer is
safe to reuse. It does **not** necessarily mean the message has been received by the
destination — it means the MPI runtime has either:
1. Copied the data into an internal buffer (for small messages — "eager" protocol), or
2. Completed a handshake with the receiver and transferred the data directly
   (for large messages — "rendezvous" protocol).

`MPI_Recv` blocks until a matching message has been received and copied into `buf`.

---

## 5.3 Receive Buffer Sizing

The receive buffer must be large enough to hold the incoming message. If too small,
MPI returns `MPI_ERR_TRUNCATE`. You can query the actual number of elements received:

```c
MPI_Status status;
MPI_Recv(buf, MAX_COUNT, MPI_INT, src, tag, comm, &status);

int actual_count;
MPI_Get_count(&status, MPI_INT, &actual_count);
printf("Received %d ints\n", actual_count);
```

For messages whose size is unknown at the receive site, use `MPI_Probe` (Section 5.5).

---

## 5.4 MPI_Sendrecv and MPI_Sendrecv_replace

A very common pattern is for every process to simultaneously send to one neighbor
and receive from another — for example, a ring shift or halo exchange. Coding this
with separate `MPI_Send` and `MPI_Recv` calls leads to deadlock (Section 5.6).

`MPI_Sendrecv` solves this correctly in one call:

```c
int MPI_Sendrecv(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    int dest,   int sendtag,
    void       *recvbuf, int recvcount, MPI_Datatype recvtype,
    int source, int recvtag,
    MPI_Comm comm, MPI_Status *status);
```

Ring shift example — each rank sends to `(rank+1) % size` and receives from
`(rank-1+size) % size`:

```c
int sendbuf[N], recvbuf[N];
/* ... fill sendbuf ... */

int right = (rank + 1) % size;
int left  = (rank - 1 + size) % size;

MPI_Sendrecv(sendbuf, N, MPI_INT, right, 0,
             recvbuf, N, MPI_INT, left,  0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

`MPI_Sendrecv` works even when `dest == source == rank` (self-communication).

`MPI_Sendrecv_replace` uses the same buffer for both send and receive:

```c
int MPI_Sendrecv_replace(
    void *buf, int count, MPI_Datatype datatype,
    int dest, int sendtag, int source, int recvtag,
    MPI_Comm comm, MPI_Status *status);
```

---

## 5.5 MPI_Probe and MPI_Iprobe

When the receiver does not know the message size in advance, probe first to allocate
the right buffer:

```c
/* MPI_Probe: blocks until a matching message is available */
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);

/* MPI_Iprobe: non-blocking probe — returns immediately */
int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag,
               MPI_Status *status);
```

Usage pattern:

```c
MPI_Status status;
MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

/* Now we know the size and source */
int count;
MPI_Get_count(&status, MPI_DOUBLE, &count);
int src = status.MPI_SOURCE;
int tag = status.MPI_TAG;

double *buf = malloc(count * sizeof(double));
MPI_Recv(buf, count, MPI_DOUBLE, src, tag, MPI_COMM_WORLD,
         MPI_STATUS_IGNORE);
/* process buf ... */
free(buf);
```

`MPI_Probe` and the subsequent `MPI_Recv` must be paired carefully in a
multi-threaded context — another thread could receive the message between the probe
and the recv. Use `MPI_Mprobe` / `MPI_Mrecv` (matched probe/receive) to atomically
claim the message:

```c
MPI_Status  status;
MPI_Message msg;
MPI_Mprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &msg, &status);

int count;
MPI_Get_count(&status, MPI_DOUBLE, &count);
double *buf = malloc(count * sizeof(double));

MPI_Mrecv(buf, count, MPI_DOUBLE, &msg, MPI_STATUS_IGNORE);
```

`MPI_Mprobe` atomically "claims" the message — no other thread or probe can steal it.

---

## 5.6 Deadlock: Causes and Avoidance

Deadlock in MPI occurs when every process is waiting for a condition that can only
be satisfied by another waiting process. The classic case:

```c
/* DEADLOCK — both processes call Send first */
if (rank == 0) {
    MPI_Send(buf, N, MPI_INT, 1, 0, MPI_COMM_WORLD);  /* waits for rank 1 recv */
    MPI_Recv(buf, N, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
} else {
    MPI_Send(buf, N, MPI_INT, 0, 0, MPI_COMM_WORLD);  /* waits for rank 0 recv */
    MPI_Recv(buf, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```

Both rank 0 and rank 1 call `MPI_Send` and block, each waiting for the other to
enter `MPI_Recv`. Neither ever does.

### Why This Is Implementation-Dependent

Small messages are often "eagerly" buffered by MPI — the send completes immediately
because MPI copies the data to an internal buffer without waiting for the receiver.
This means the deadlock above may **not** occur with small buffers on some
implementations, but **will** occur once the message exceeds the eager threshold
(often 8–64 KB). Never rely on eager buffering for correctness.

### Avoidance Strategies

**Strategy 1: Use MPI_Sendrecv**

```c
/* Correct — no deadlock */
MPI_Sendrecv(sendbuf, N, MPI_INT, dest,   0,
             recvbuf, N, MPI_INT, source, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

**Strategy 2: Interleave Send/Recv by rank parity**

```c
if (rank % 2 == 0) {
    MPI_Send(sendbuf, N, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
    MPI_Recv(recvbuf, N, MPI_INT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
} else {
    MPI_Recv(recvbuf, N, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(sendbuf, N, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
}
```

**Strategy 3: Use non-blocking operations (preferred for complex patterns)**

```c
MPI_Request reqs[2];
MPI_Isend(sendbuf, N, MPI_INT, dest,   0, MPI_COMM_WORLD, &reqs[0]);
MPI_Irecv(recvbuf, N, MPI_INT, source, 0, MPI_COMM_WORLD, &reqs[1]);
MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
```

Non-blocking is the most general solution and the one you should default to in any
pattern more complex than a simple pairwise exchange.

---

## 5.7 Send/Receive to Self

Sending a message to yourself (`dest == rank`) is legal:

```c
/* Rank 0 sends to itself */
if (rank == 0) {
    MPI_Sendrecv(sendbuf, N, MPI_INT, 0, 0,
                 recvbuf, N, MPI_INT, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```

You cannot use plain `MPI_Send` followed by `MPI_Recv` to self — that deadlocks for
the same reason as the two-process case. Use `MPI_Sendrecv` or non-blocking calls.

---

## 5.8 Measuring Message Throughput

A complete bandwidth benchmark (builds on the ping-pong from Chapter 2):

```c
/* bandwidth.c */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int ITERS = 200;
    const int SIZES[] = {64, 256, 1024, 4096, 16384, 65536,
                         262144, 1048576, 4194304};
    const int NSIZES = sizeof(SIZES) / sizeof(SIZES[0]);

    char *sbuf = malloc(SIZES[NSIZES-1]);
    char *rbuf = malloc(SIZES[NSIZES-1]);

    if (rank == 0)
        printf("%10s  %14s\n", "Bytes", "BW (MB/s)");

    for (int s = 0; s < NSIZES; s++) {
        int sz = SIZES[s];

        /* Warmup */
        for (int i = 0; i < 10; i++)
            MPI_Sendrecv(sbuf, sz, MPI_BYTE, (rank+1)%size, 0,
                         rbuf, sz, MPI_BYTE, (rank-1+size)%size, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int i = 0; i < ITERS; i++)
            MPI_Sendrecv(sbuf, sz, MPI_BYTE, (rank+1)%size, 0,
                         rbuf, sz, MPI_BYTE, (rank-1+size)%size, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double t1 = MPI_Wtime();

        if (rank == 0) {
            double bw = (sz * (double)ITERS) / (t1 - t0) / 1e6;
            printf("%10d  %14.2f\n", sz, bw);
        }
    }

    free(sbuf); free(rbuf);
    MPI_Finalize();
    return 0;
}
```

---

## 5.9 Practical Rules for Blocking P2P

1. **Never call `MPI_Send` then `MPI_Recv` in the same order on every process** without
   knowing that the other side reverses the order. This is the deadlock recipe.

2. **Use `MPI_Sendrecv`** for symmetric pairwise exchanges. It is portable, correct,
   and often optimized by implementations.

3. **Check receive count** with `MPI_Get_count` when the sender may send a variable
   number of elements.

4. **Never rely on eager buffering** for correctness. Write code that is correct under
   the rendezvous protocol (i.e., `MPI_Send` may block).

5. **Tag discipline**: define tags as named constants. A mismatch between send and
   receive tags causes the receive to hang waiting for a message that never arrives.

---

## Summary

| Function | Purpose |
|---|---|
| `MPI_Send` | Blocking send; buffer safe after return |
| `MPI_Recv` | Blocking receive; fills buffer and status |
| `MPI_Sendrecv` | Send and receive simultaneously; deadlock-safe |
| `MPI_Sendrecv_replace` | In-place Sendrecv using one buffer |
| `MPI_Probe` | Block until matching message is available; read status |
| `MPI_Iprobe` | Non-blocking probe; returns `flag` |
| `MPI_Mprobe` / `MPI_Mrecv` | Atomically claim a message (thread-safe probe) |
| `MPI_Get_count` | Get actual number of elements received |
