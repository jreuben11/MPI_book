# Chapter 1: The MPI Model

## 1.1 What MPI Is (and Is Not)

MPI — the Message Passing Interface — is a standardized API for writing programs that
communicate by explicitly sending and receiving messages across a network of processes.
It is not a language, not a compiler, and not a runtime scheduler. It is a library
specification: vendors and open-source projects implement it, and your code calls it.

The key properties of MPI:

- **Explicit parallelism.** You write one program; MPI gives each instance (process) a
  unique identity, and your code branches on that identity to do different work.
- **No shared memory between processes.** Each process has its own address space. Data
  moves only when you call MPI functions to move it.
- **Portability across scales.** The same source code runs on a laptop with 4 cores, a
  cluster with 10,000 nodes, and a supercomputer with millions of cores.
- **Language neutral.** The standard defines C and Fortran bindings. C++ programs use
  the C bindings directly (the old C++ bindings were deprecated in MPI 2.2 and removed
  in MPI 3.0).

MPI is **not** a replacement for threading. It is not designed for fine-grained
synchronization within a shared memory node. In practice, high-performance codes combine
MPI across nodes with OpenMP or pthreads within a node — this is covered in Chapter 21.

---

## 1.2 The SPMD Execution Model

MPI programs almost always follow the **Single Program, Multiple Data (SPMD)** pattern:

```
mpiexec -n 4 ./myprogram
```

This launches 4 identical copies of `myprogram` as separate OS processes. Each process:

1. Calls `MPI_Init` to join the MPI runtime.
2. Queries its **rank** (integer identity, 0 to N-1) within a **communicator**.
3. Uses the rank to determine what work to do and whom to communicate with.
4. Calls `MPI_Finalize` to leave the MPI runtime cleanly.

```
Process 0  |  Process 1  |  Process 2  |  Process 3
-----------+-------------+-------------+-------------
MPI_Init   |  MPI_Init   |  MPI_Init   |  MPI_Init
rank = 0   |  rank = 1   |  rank = 2   |  rank = 3
[work]     |  [work]     |  [work]     |  [work]
MPI_Send   |  MPI_Recv   |             |
           |  MPI_Send   |  MPI_Recv   |
MPI_Finalize| MPI_Finalize| MPI_Finalize| MPI_Finalize
```

Although SPMD is the norm, MPI also supports **MPMD (Multiple Program, Multiple Data)**,
where different executables are launched together. This is used in coupled simulations
(e.g., atmosphere + ocean models), covered briefly in Chapter 14.

### Communicators and Ranks

A **communicator** is a handle representing a group of processes that can communicate
with each other. The default communicator, `MPI_COMM_WORLD`, includes every process
launched by `mpiexec`. Within a communicator:

- **Rank**: an integer from 0 to size−1, uniquely identifying a process.
- **Size**: the total number of processes in the communicator.

Ranks are communicator-local. Process 0 in one communicator may be process 3 in another.
This is how MPI isolates subsystems from each other.

---

## 1.3 MPI Standard Evolution

Understanding what version introduced what feature prevents you from writing code that
silently falls back to slower paths on older implementations.

| Version | Year | Key Additions |
|---------|------|---------------|
| MPI 1.0 | 1994 | Point-to-point, collectives, datatypes, communicators |
| MPI 1.1 | 1995 | Clarifications |
| MPI 1.2 | 1997 | Clarifications |
| MPI 2.0 | 1997 | One-sided (RMA), parallel I/O, dynamic processes, C++ bindings |
| MPI 2.1 | 2008 | Merged 1.x and 2.x documents |
| MPI 2.2 | 2009 | C++ bindings deprecated, `MPI_Dist_graph_create` |
| MPI 3.0 | 2012 | Non-blocking collectives, `MPI_T` interface, C++ bindings removed |
| MPI 3.1 | 2015 | Errata and clarifications; added `MPI_Aint_add`, `MPI_Aint_diff` |
| MPI 4.0 | 2021 | Sessions, partitioned communication, persistent collectives, large-count `_c` variants |
| MPI 5.0 | 2025 | ABI standardization, RMA clarifications, deprecation cleanup, expanded info keys |

### What MPI 4.0 Changed (Practically)

MPI 4.0 is the most significant release since MPI 2.0. The four additions that affect
everyday code:

1. **Sessions model** (`MPI_Session_init`) — allows libraries to initialize MPI
   independently, without touching `MPI_COMM_WORLD`. Covered in Chapter 22.

2. **Partitioned communication** (`MPI_Psend_init`, `MPI_Precv_init`) — allows sender
   and receiver to mark individual partitions of a message as ready/arrived
   independently. Designed for GPU pipelines and producer/consumer patterns.
   Covered in Chapter 23.

3. **Persistent collectives** (`MPI_Allreduce_init`, etc.) — extends the persistent
   request model from point-to-point to collectives. Covered in Chapter 11.

4. **Large count support** — `MPI_Count` (a 64-bit integer type) existed since MPI 3.0
   for status queries. MPI 4.0 added `_c` suffix variants of all communication functions
   that accept `MPI_Count` instead of `int`, enabling messages larger than 2^31 elements.
   Covered in Chapter 12.

### What MPI 5.0 Changed (Practically)

MPI 5.0 focuses on ABI standardization and cleanup:

- **ABI standardization** — a standard Application Binary Interface (`mpi_abi` shared
  library and standardized `mpi.h`) so applications compiled against one MPI
  implementation can run against any other conforming implementation without
  recompilation.
- **RMA memory model clarifications** — the unified memory model is now the default
  recommendation; separate model remains for legacy hardware.
- **Deprecation cleanup** — several rarely-used functions from MPI 1.x are now formally
  deprecated (though implementations will support them for years).
- **New `MPI_Info` keys** — standardized keys for binding, memory placement, and
  collective algorithm selection.
- **Note:** Fault tolerance (`MPIX_Comm_revoke`, `MPIX_Comm_shrink`, `MPIX_Comm_agree`)
  remains in the `MPIX_` extension namespace and was not standardized in MPI 5.0.

---

## 1.4 When to Use MPI

### MPI Fits Well When:

- **Distributed memory is required**: multiple nodes connected by a network (Ethernet,
  InfiniBand, HPE Slingshot https://www.hpe.com/psnow/doc/a50002546enw ). Shared memory is not available across nodes.
- **Scale exceeds a single machine**: problems that need hundreds or thousands of cores.
- **Independent data decomposition**: the problem naturally splits into chunks that
  each process owns, with boundary exchanges between neighbors.
- **Long-running compute jobs**: the overhead of MPI setup is negligible compared to
  hours of compute time.

### MPI Fits Poorly When:

- **Fine-grained synchronization**: many small, irregular tasks that need frequent
  coordination. Use a work-stealing thread pool or a task framework instead.
- **Highly irregular communication patterns**: if every process needs to talk to every
  other process at unpredictable times, MPI's overhead per message becomes significant.
- **Rapid prototyping of algorithms**: MPI requires explicit data layout decisions
  upfront. Consider Python + mpi4py for prototyping, then port to C/C++.

### The Hybrid Model

Most production HPC codes use MPI across nodes and threads within a node:

```
Node 0                      Node 1
+----+----+----+----+       +----+----+----+----+
| T0 | T1 | T2 | T3 |  MPI | T0 | T1 | T2 | T3 |
|         Rank 0    | <--> |         Rank 1    |
+-------------------+       +-------------------+
```

Each MPI rank spawns OpenMP threads (or uses a CUDA/HIP device). Within a node,
threads share data via shared memory. Across nodes, ranks communicate via MPI.

This hybrid approach reduces the number of MPI ranks (and thus message counts) while
still parallelizing within-node work. Chapter 21 covers the thread safety requirements
MPI imposes on this model.

---

## 1.5 The MPI Specification vs. MPI Implementations

The MPI Forum publishes the **specification** — a document defining what functions must
exist, what arguments they take, and what behavior they must exhibit. The specification
does not define performance, internal algorithms, or network protocols.

Major open-source implementations:

| Implementation | Notes |
|---|---|
| **Open MPI** | Most widely deployed on Linux clusters; supports InfiniBand, UCX, PMIx |
| **MPICH** | Reference implementation from Argonne; basis for many vendor derivatives |
| **MVAPICH** | MPICH derivative optimized for InfiniBand |

Major vendor implementations (typically MPICH or Open MPI derivatives):

| Vendor | Implementation |
|---|---|
| Intel | Intel MPI (based on MPICH) |
| HPE | Cray MPICH |
| AWS | AWS EFA + Open MPI or MPICH |
| NVIDIA | HPC-X (Open MPI + UCX + SHARP) |

**Practical rule**: write to the MPI standard, not to implementation-specific behavior.
Use `MPI_Info` hints to request optimizations, but never require them for correctness.

---

## 1.6 The MPI Forum and Specification Documents

The MPI standard is maintained by the MPI Forum, an open body of vendors, labs, and
academic groups. The current documents:

- **MPI 4.0**: finalized June 2021 — 1,139 pages
- **MPI 5.0**: finalized 2025 — includes errata and new chapters

The full specification is the authoritative reference for edge cases and compliance
questions. This guide covers the 20% of the API you will use for 80% of real programs,
organized for learning rather than reference.

---

## Summary

| Concept | Key Points |
|---|---|
| SPMD | One binary, N processes, each with a rank; branch on rank |
| Communicator | Scopes a set of processes; `MPI_COMM_WORLD` is the default |
| Rank / Size | Identity within a communicator; 0-indexed |
| MPI 4.0 | Sessions, partitioned comm, persistent collectives, large counts |
| MPI 5.0 | Fault tolerance standardized, RMA clarified, deprecation cleanup |
| Implementations | Open MPI and MPICH are the open-source foundations |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
