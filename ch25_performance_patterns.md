# Chapter 25: Performance Patterns

## 25.1 Understanding the Performance Model

MPI performance is governed by a few fundamental parameters:

- **Latency (α)**: time to send a zero-byte message. Typical values:
  - Loopback (same process): < 1 μs
  - Shared memory (same node): 1–3 μs
  - InfiniBand HDR (inter-node): 1–2 μs
  - 100GbE (inter-node): 5–20 μs

- **Bandwidth (1/β)**: bytes per second. Typical peak values:
  - Shared memory: 50–200 GB/s
  - InfiniBand HDR (200 Gb/s): ~22 GB/s
  - HDR-400 / NDR (400 Gb/s): ~44 GB/s
  - 100GbE: ~10 GB/s

The time to send a message of `n` bytes is approximately:
```
T(n) = α + n/β
```

This simple model predicts:
- Small messages are latency-bound: doubling message count doubles time.
- Large messages are bandwidth-bound: doubling message size doubles time.
- Below α×β bytes (~16 KB for typical InfiniBand), latency dominates.

---

## 25.2 Eager vs. Rendezvous Protocol

MPI implementations switch between two internal protocols based on message size:

### Eager Protocol (small messages)

The sender copies the message to an internal buffer and returns immediately. No
handshake with the receiver is needed. The receiver finds the message waiting in
the buffer when it calls `MPI_Recv`.

- **Threshold**: typically 8 KB – 64 KB (configurable per implementation)
- **Pros**: low latency; sender returns immediately without waiting for receiver
- **Cons**: requires internal buffering; may require extra copy

### Rendezvous Protocol (large messages)

Sender sends a "ready-to-send" control message and waits. Receiver acknowledges.
Then data flows directly to the receiver's buffer (zero-copy on RDMA hardware).

- **Pros**: no extra buffering; zero-copy possible
- **Cons**: latency of one full round-trip (one RTT) before data flows; receiver must be posted

Implication for your code: large messages require the receiver to post `MPI_Recv`
(or `MPI_Irecv`) before or simultaneously with the sender's `MPI_Send`. Use
non-blocking calls to avoid deadlock (Chapter 5).

---

## 25.3 Aggregating Small Messages

The single most impactful optimization is often **reducing message count** by batching
small messages into fewer large ones.

### Bad Pattern: Many Small Sends

```c
/* Sends N individual scalars — N round-trips, N × latency */
for (int i = 0; i < N; i++) {
    MPI_Send(&values[i], 1, MPI_DOUBLE, dest, i, comm);
}
```

### Better: One Large Send

```c
/* One message — 1 round-trip */
MPI_Send(values, N, MPI_DOUBLE, dest, 0, comm);
```

**Rule of thumb**: if you are sending less than ~1 KB per message, look for ways
to batch. If messages naturally come in bursts, accumulate and send once.

### Derived Datatypes for Non-Contiguous Data

Instead of packing non-contiguous data into a temporary buffer (two copies), use
a derived datatype (one or zero copies):

```c
/* Pack-then-send: 2 copies */
double tmpbuf[N];
for (int i = 0; i < N; i++) tmpbuf[i] = matrix[i][col];
MPI_Send(tmpbuf, N, MPI_DOUBLE, dest, 0, comm);

/* Derived datatype: 1 copy (or hardware scatter-gather) */
MPI_Datatype col_type;
MPI_Type_vector(N, 1, NCOLS, MPI_DOUBLE, &col_type);
MPI_Type_commit(&col_type);
MPI_Send(&matrix[0][col], 1, col_type, dest, 0, comm);
MPI_Type_free(&col_type);
```

---

## 25.4 Overlapping Communication and Computation

Use non-blocking operations to overlap communication with computation:

```c
/* Without overlap */
double t0 = MPI_Wtime();
MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, comm);
compute_next_step();
printf("Time: %.3f s\n", MPI_Wtime() - t0);

/* With overlap — pipelining reduction with computation */
MPI_Request req;
MPI_Iallreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, comm, &req);
compute_next_step();    /* concurrent with reduction */
MPI_Wait(&req, MPI_STATUS_IGNORE);
printf("Time: %.3f s\n", MPI_Wtime() - t0);
```

Whether the overlap provides actual benefit depends on:
1. Whether the MPI implementation supports asynchronous progress.
2. Whether the computation is on the critical path.
3. Whether the compute and communication use separate hardware resources.

### Async Progress

On most implementations, MPI communication progresses only when an MPI function is
called. To enable true background progress:

**MPICH**:
```bash
export MPICH_ASYNC_PROGRESS=1  # enables a progress thread
```

**Open MPI**:
```bash
mpiexec --mca mpi_async_progress true ./myprogram
```

Async progress uses a dedicated thread that consumes CPU cycles. The tradeoff is
background progress vs. thread overhead. Measure before enabling.

---

## 25.5 Collective Algorithm Selection

MPI implementations offer multiple algorithms for each collective operation. The
optimal choice depends on message size, process count, and network topology.

### Communicating Hints via MPI_Info

Some implementations allow algorithm selection through info keys:

```c
/* Open MPI: select Bcast algorithm */
MPI_Info info;
MPI_Info_create(&info);
MPI_Info_set(info, "bcast_algorithm", "3");  /* e.g., binomial tree */
MPI_Comm new_comm;
MPI_Comm_dup_with_info(MPI_COMM_WORLD, info, &new_comm);
MPI_Info_free(&info);
```

These keys are implementation-specific. Consult your MPI vendor's documentation.

### General Collective Tuning Guidance

| Collective | Small messages | Large messages |
|---|---|---|
| `MPI_Bcast` | Binomial tree (low latency) | Pipeline or flat tree |
| `MPI_Allreduce` | Recursive halving (log P steps) | Reduce-scatter + Allgather |
| `MPI_Alltoall` | Short message: log P | Long message: pairs |
| `MPI_Barrier` | Dissemination or tournament | — |

For most users, the default algorithm selection in your MPI implementation is
near-optimal. Only tune when profiling identifies a specific collective as a
bottleneck.

---

## 25.6 NUMA-Aware Placement

On multi-socket nodes, NUMA (Non-Uniform Memory Access) effects can reduce bandwidth
by 2–3× if processes access memory on a remote socket.

### Binding Processes to Cores

```bash
# Open MPI: bind each process to a specific CPU
mpiexec -n 4 --bind-to core ./myprogram

# SLURM: bind tasks
srun -n 64 --cpus-per-task=1 --ntasks-per-socket=16 ./myprogram
```

### Detecting NUMA from MPI

```c
/* Get shared memory domain rank for intra-node optimization */
MPI_Comm shared_comm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                    MPI_INFO_NULL, &shared_comm);

int local_rank;
MPI_Comm_rank(shared_comm, &local_rank);
/* Processes with local_rank == 0 are likely on different sockets */
```

### Hierarchical Collectives

For bandwidth-sensitive collectives on multi-socket nodes:

```c
/* 2-level Allreduce: reduce within socket, then across sockets */

/* Step 1: reduce within shared memory domain */
MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, shared_comm);

/* Step 2: each socket's representative reduces across sockets */
if (local_rank == 0) {
    MPI_Comm socket_leaders;
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &socket_leaders);
    MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, socket_leaders);
    MPI_Comm_free(&socket_leaders);
}

/* Step 3: broadcast within socket */
MPI_Bcast(&val, 1, MPI_DOUBLE, 0, shared_comm);
```

Many MPI implementations perform this automatically for shared-memory communicators.
Test whether manual hierarchical reduction outperforms the default.

---

## 25.7 Shared Memory Fast Path

`MPI_Win_allocate_shared` provides direct load/store access to other processes'
memory on the same node — zero-copy, hardware cache-coherent:

```c
MPI_Win win;
double *local_ptr;
MPI_Win_allocate_shared(local_n * sizeof(double), sizeof(double),
                        MPI_INFO_NULL, shared_comm, &local_ptr, &win);

/* Get pointer to rank 0's memory on this node */
MPI_Aint size0; int disp_unit0;
double *rank0_ptr;
MPI_Win_shared_query(win, 0, &size0, &disp_unit0, &rank0_ptr);

/* Direct read — no MPI call needed */
double val = rank0_ptr[42];
```

This is faster than `MPI_Send`/`MPI_Recv` for intra-node communication, equivalent
to OpenMP shared-array access.

**Memory ordering**: on x86, load/store have sequential consistency for individual
accesses. For multi-word updates, use `MPI_Win_sync` to ensure ordering:

```c
/* Ensure stores to shared window are visible before other processes read */
MPI_Win_sync(win);  /* acts as a memory fence */
MPI_Barrier(shared_comm);  /* ensure all processes have synced */
```

---

## 25.8 Avoiding Synchronization Bottlenecks

### Avoid Unnecessary Barriers

```c
/* Common anti-pattern: barrier before every collective */
MPI_Barrier(comm);           /* usually unnecessary */
MPI_Allreduce(...);          /* MPI collectives are NOT guaranteed barriers by the standard */
```

Despite common implementation behavior, the MPI standard does not guarantee that
collectives synchronize all processes. Add `MPI_Barrier` only when you need
guaranteed synchronization without data movement.

### Pipelining Time Steps

For iterative codes, pipeline the convergence check:

```c
double error_prev = HUGE_VAL;
MPI_Request req = MPI_REQUEST_NULL;

for (int step = 0; step < MAX_STEPS; step++) {
    /* Wait for previous step's convergence check */
    if (req != MPI_REQUEST_NULL)
        MPI_Wait(&req, MPI_STATUS_IGNORE);

    if (error_prev < TOLERANCE) break;

    compute_step(data, step);

    /* Start convergence check asynchronously (result arrives next iteration) */
    double local_err = compute_local_error(data);
    MPI_Iallreduce(&local_err, &error_prev, 1, MPI_DOUBLE, MPI_MAX, comm, &req);
}
if (req != MPI_REQUEST_NULL) MPI_Wait(&req, MPI_STATUS_IGNORE);
```

---

## Summary

| Technique | Impact | When to Apply |
|---|---|---|
| Batch small messages | Very high | Message count > 1000; sizes < 1 KB |
| Overlap with Iallreduce | Medium | Reduction on critical path |
| Derived datatypes | Medium | Non-contiguous data |
| Shared memory window | High (intra-node) | Replacing intra-node P2P |
| Hierarchical collectives | Medium | Large node counts, multi-socket |
| Process binding | High | NUMA machines, shared compute |
| Async progress | Variable | Large non-blocking ops |
| Avoid unnecessary Barrier | Low-Medium | Present in most codes |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
