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

### What NUMA Is

Modern servers contain multiple CPU sockets, each with its own local DRAM attached
via a dedicated memory bus. All sockets are connected by a high-speed inter-socket
fabric (Intel UPI, AMD Infinity Fabric, IBM X-Bus). The entire physical memory is
addressable from any socket, but **access time differs depending on where the memory
lives**:

```
Socket 0                         Socket 1
┌─────────────────────┐          ┌─────────────────────┐
│  Core 0–31          │          │  Core 32–63         │
│  L1/L2/L3 cache     │          │  L1/L2/L3 cache     │
│  Memory controller  │◄──UPI───►│  Memory controller  │
└────────┬────────────┘          └────────┬────────────┘
         │                                │
      DRAM bank 0                      DRAM bank 1
   (local to Socket 0)             (local to Socket 1)
```

| Access path | Typical latency | Typical bandwidth |
|---|---|---|
| Local DRAM (same socket) | ~80 ns | ~150 GB/s |
| Remote DRAM (cross-socket via UPI) | ~140 ns | ~60–80 GB/s |

The ratio between local and remote bandwidth is typically **1.5–3×**. For
bandwidth-bound HPC kernels (stencils, BLAS, FFT), accessing remote DRAM is the
difference between achieving peak performance and running at half speed.

Each socket (and sometimes each NUMA domain within a socket on AMD EPYC) is called
a **NUMA node**. The OS assigns physical pages to NUMA nodes. The default policy on
Linux is **first-touch**: a page is allocated on the NUMA node of the thread that
first writes to it, regardless of which thread later reads it.

* Note: NUMA effects only arise on **multi-socket** machines. A single-socket workstation has
one memory controller and uniform access to all DRAM — `numactl --hardware` will show
`available: 1 nodes (0)` and a distance matrix of just `10`. The content above applies
to HPC cluster nodes, which are almost always dual- or quad-socket (e.g., 2× AMD EPYC
or 2× Intel Xeon per node).

**First-touch trap**: if a single thread (e.g., the main thread on socket 0)
initialises a large array with `memset` or a serial loop, all pages land on socket 0.
Worker threads on socket 1 then access that data remotely for the entire run.

```c
/* Bad: all pages land on socket 0 (initializing thread) */
double *buf = malloc(N * sizeof(double));
memset(buf, 0, N * sizeof(double));   /* first touch from socket 0 only */

/* Good: first touch matches the thread that will use each partition */
double *buf = malloc(N * sizeof(double));
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) buf[i] = 0.0;  /* each thread touches its own pages */
```

### Inspecting NUMA Topology

```bash
# Show NUMA node layout
numactl --hardware

# Show which CPUs are on which NUMA node
lscpu | grep -i numa

# Run a process pinned to NUMA node 0, using only node 0's memory
numactl --cpunodebind=0 --membind=0 ./myprogram
```

Example `numactl --hardware` output on a dual-socket node:

```
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 ... 31
node 0 size: 128 GB
node 1 cpus: 32 33 34 ... 63
node 1 size: 128 GB
node distances:
node   0   1
  0:  10  21
  1:  21  10
```

The distance values (10 = local, 21 = remote) directly reflect the latency ratio.

### POSIX NUMA-Aware Allocation

POSIX provides `numa_alloc_onnode` (from `libnuma`) for explicit NUMA-local allocation,
and `mbind` / `set_mempolicy` from `<numaif.h>` for policy-based placement:

```c
#include <numa.h>        /* libnuma: -lnuma */
#include <numaif.h>

/* Approach 1: allocate directly on a specific NUMA node */
int target_node = 0;
size_t bytes = N * sizeof(double);
double *buf = (double *)numa_alloc_onnode(bytes, target_node);
if (!buf) { perror("numa_alloc_onnode"); exit(1); }

/* ... use buf ... */
numa_free(buf, bytes);   /* must use numa_free, not free() */


/* Approach 2: malloc then bind pages to a node set via mbind */
double *buf2 = malloc(bytes);
unsigned long nodemask = 1UL << target_node;  /* bit mask: node 0 only */

/* MPOL_BIND: pages are allocated only from the specified node set */
mbind(buf2, bytes,
      MPOL_BIND,
      &nodemask, /* maxnode */ sizeof(nodemask) * 8,
      MPOL_MF_MOVE | MPOL_MF_STRICT);
```

**Memory policies** passed to `mbind`:

| Policy | Behaviour |
|---|---|
| `MPOL_DEFAULT` | Thread's default policy (usually first-touch) |
| `MPOL_BIND` | Allocate only from the specified node set; fail if full |
| `MPOL_PREFERRED` | Prefer specified node, fall back to others if necessary |
| `MPOL_INTERLEAVE` | Round-robin across nodes; good for shared read-mostly data |

`MPOL_INTERLEAVE` is useful for data accessed equally by threads on all sockets — it
distributes the bandwidth load rather than saturating one node's memory bus:

```c
/* Interleave pages across all NUMA nodes for a shared read-only table */
unsigned long all_nodes = (1UL << numa_num_configured_nodes()) - 1;
mbind(shared_table, table_bytes, MPOL_INTERLEAVE, &all_nodes,
      sizeof(all_nodes) * 8, MPOL_MF_MOVE);
```

### Querying NUMA Node of a Running MPI Process

From within an MPI program, determine which NUMA node each rank's memory should
target by combining `sched_getcpu` (which CPU is running) with `numa_node_of_cpu`:

```c
#include <sched.h>   /* sched_getcpu */
#include <numa.h>

int cpu  = sched_getcpu();
int node = numa_node_of_cpu(cpu);
printf("Rank %d: CPU %d, NUMA node %d\n", rank, cpu, node);

/* Allocate rank-local data on that rank's NUMA node */
double *local_buf = (double *)numa_alloc_onnode(local_N * sizeof(double), node);
```

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

**Memory ordering**: x86 uses Total Store Order (TSO), not full sequential consistency
— store-load reordering is permitted. The MPI standard requires `MPI_Win_sync` before
and after load/store accesses to a shared window on all architectures, not only for
multi-word updates:

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
