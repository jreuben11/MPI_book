# MPI 5.0 Programming Guide — C/C++

A comprehensive ~200-page programming guide to MPI 4.0/5.0 in C/C++.
Covers the practical 20% of the API used in 80% of real HPC programs.

**[Read the Preface →](preface.md)**

---

## Quick Start

```bash
# Install MPI
sudo apt install libopenmpi-dev openmpi-bin   # Ubuntu/Debian
sudo dnf install openmpi-devel                # Fedora/RHEL

# Compile
mpicc -O2 -o hello hello.c
mpic++ -O2 -std=c++17 -o hello hello.cpp

# Run
mpiexec -n 4 ./hello
```

---

## Table of Contents

### Part I — Foundations (~30 pages)

| Chapter | Topics |
|---|---|
| [Ch 1: The MPI Model](ch01_mpi_model.md) | SPMD model, process ranks, communicators; standard evolution 1.x→5.0; MPI vs threads vs hybrid |
| [Ch 2: Setup & First Programs](ch02_setup_first_programs.md) | `mpicc`/`mpic++`, linking ABI; `mpirun`/`mpiexec`/PMIx launchers; Hello World; `MPI_Init`/`MPI_Finalize` |
| [Ch 3: Core Abstractions](ch03_core_abstractions.md) | Communicators, groups, contexts, tags; all handle types; buffer ownership; `MPI_Status`; `MPI_Info` |
| [Ch 4: Error Handling](ch04_error_handling.md) | Error classes and codes; `MPI_ERRORS_RETURN`, `MPI_ERRORS_ABORT`; custom handlers; `MPI_Abort`; ULFM overview |

### Part II — Point-to-Point Communication (~35 pages)

| Chapter | Topics |
|---|---|
| [Ch 5: Blocking Send & Receive](ch05_blocking_p2p.md) | `MPI_Send`/`MPI_Recv` semantics and buffering; `MPI_Probe`/`MPI_Iprobe`; `MPI_Sendrecv`; deadlock taxonomy |
| [Ch 6: Non-Blocking Communication](ch06_nonblocking_p2p.md) | `MPI_Isend`/`MPI_Irecv` request lifecycle; `Wait`/`Test` variants; overlap patterns; halo exchange |
| [Ch 7: Send Modes](ch07_send_modes.md) | Standard, Buffered (`MPI_Bsend`), Synchronous (`MPI_Ssend`), Ready (`MPI_Rsend`); portability traps |
| [Ch 8: Persistent Requests](ch08_persistent_requests.md) | `MPI_Send_init`/`MPI_Recv_init`, `MPI_Start`, `MPI_Startall`; tight-loop reuse pattern |

### Part III — Collective Communication (~30 pages)

| Chapter | Topics |
|---|---|
| [Ch 9: Core Collectives](ch09_core_collectives.md) | `MPI_Bcast`, `Scatter`/`Gather`, `Allgather`, `Reduce`, `Allreduce`, `Alltoall`, `Barrier`; `MPI_IN_PLACE`; custom `MPI_Op` |
| [Ch 10: Advanced Collectives](ch10_advanced_collectives.md) | Vector variants `Scatterv`/`Gatherv`/`Allgatherv`/`Alltoallv`/`w`; `Scan`/`Exscan`; `Reduce_scatter`; custom ops |
| [Ch 11: Non-Blocking & Persistent Collectives](ch11_nonblocking_persistent_collectives.md) | `MPI_Ibcast`, `MPI_Iallreduce`, etc.; MPI 4.0 `MPI_Bcast_init`/`MPI_Allreduce_init`; init/start/complete lifecycle |

### Part IV — Datatypes (~20 pages)

| Chapter | Topics |
|---|---|
| [Ch 12: Built-in Datatypes](ch12_builtin_datatypes.md) | C type correspondence table; fixed-width types; `MPI_Count`, large-count `_c` variants (MPI 4.0); `MPI_Aint`, `MPI_Offset` |
| [Ch 13: Derived Datatypes](ch13_derived_datatypes.md) | Contiguous, vector, hvector, indexed, hindexed, struct; `MPI_Type_create_subarray`/`darray`; extent and resizing; worked examples |

### Part V — Communicators & Topologies (~20 pages)

| Chapter | Topics |
|---|---|
| [Ch 14: Communicator Operations](ch14_communicator_operations.md) | `MPI_Comm_dup`, `split`, `split_type(SHARED_MEMORY)`, `create`, `create_group`; inter-communicators; keyvals |
| [Ch 15: Process Topologies](ch15_process_topologies.md) | Cartesian: `MPI_Cart_create`, `Cart_coords`, `Cart_shift`, `Cart_sub`; distributed graph topology; neighbor collectives |

### Part VI — One-Sided Communication / RMA (~20 pages)

| Chapter | Topics |
|---|---|
| [Ch 16: Windows & Memory Models](ch16_rma_windows.md) | `MPI_Win_create`, `Win_allocate`, `Win_create_dynamic`; shared memory windows; unified vs separate memory models |
| [Ch 17: RMA Operations](ch17_rma_operations.md) | `MPI_Put`, `MPI_Get`, `MPI_Accumulate`; `Fetch_and_op`, `Compare_and_swap`; `MPI_Get_accumulate`, `Rput`/`Rget` |
| [Ch 18: RMA Synchronization](ch18_rma_synchronization.md) | Fence (active target); PSCW: `Win_post`/`Win_start`/`Win_complete`/`Win_wait`; passive: `Win_lock`/`flush`/`unlock` |

### Part VII — Parallel I/O (~15 pages)

| Chapter | Topics |
|---|---|
| [Ch 19: MPI-IO Basics](ch19_mpio_basics.md) | `MPI_File_open`/`close`/`set_view`; individual vs collective I/O; `File_read_at_all`/`write_at_all`; info hints |
| [Ch 20: Advanced MPI-IO](ch20_advanced_mpio.md) | Shared file pointer; non-blocking I/O `File_iread_at`/`iwrite_at`; derived datatypes for non-contiguous layouts |

### Part VIII — Advanced Topics (~25 pages)

| Chapter | Topics |
|---|---|
| [Ch 21: MPI + Threads](ch21_mpi_threads.md) | `MPI_Init_thread`; thread levels `SINGLE`→`MULTIPLE`; hybrid MPI+OpenMP patterns: funneled, serialized, multiple |
| [Ch 22: The Sessions Model — MPI 4.0](ch22_sessions_model.md) | `MPI_Session_init`/`finalize`; process sets; `MPI_Group` from sessions; library-safe initialization |
| [Ch 23: Partitioned Communication — MPI 4.0](ch23_partitioned_communication.md) | `MPI_Psend_init`/`Precv_init`; `MPI_Pready`/`Parrived`; GPU-direct and producer/consumer pipelines |
| [Ch 24: Large Counts & MPI 4.0/5.0 Additions](ch24_mpi4_mpi5_additions.md) | `MPI_Count`, `_c` suffix functions; `MPI_Comm_idup`; `MPI_Info_*` updates; MPI 5.0 ULFM and RMA clarifications |

### Part IX — Performance & Tooling (~15 pages)

| Chapter | Topics |
|---|---|
| [Ch 25: Performance Patterns](ch25_performance_patterns.md) | Latency/bandwidth model; eager vs rendezvous; collective algorithm selection; synchronization bottlenecks; NUMA |
| [Ch 26: Profiling & Debugging](ch26_profiling_debugging.md) | PMPI profiling layer; `MPI_T` CVars and PVars; Score-P, TAU, mpiP, ITAC; bug checklist |

### Part X — Integrations (~85 pages)

| Chapter | Topics |
|---|---|
| [Ch 27: MPI with C++20 Threads](ch27_cpp20_threads.md) | `std::jthread` RAII; `std::stop_token` progress threads; `std::packaged_task`+`future` async MPI; `std::barrier` coordination |
| [Ch 28: MPI with SLURM](ch28_slurm.md) | `srun` vs `mpiexec`; batch script anatomy; SLURM env vars; process binding; PMIx bootstrapping; job arrays; Lustre I/O |
| [Ch 29: OpenSHMEM](ch29_openshmem.md) | PE model, symmetric heap; `shmem_put`/`get`; synchronization; atomics; OpenSHMEM 1.5 teams; MPI interop |
| [Ch 30: MPI with CUDA and NCCL](ch30_cuda_nccl.md) | GPU-aware MPI; device pointer sends; CUDA streams + non-blocking overlap; NCCL init and collectives; MPI vs NCCL |
| [Ch 31: GPUDirect Storage](ch31_gpudirect_storage.md) | GDS architecture; cuFile API; `O_DIRECT` alignment; MPI+GDS pattern; GDS+partitioned comm; checkpoint/restart |
| [Ch 32: AMD ROCm and HIP](ch32_rocm_hip.md) | HIP API; `hipify` porting; ROCm-aware MPI; Infinity Fabric peer access; RCCL collectives; MI300X unified memory |
| [Ch 33: MPI in Containers](ch33_containers.md) | Host MPI vs PMIx-only models; Singularity/Apptainer definition files; GPU bind-mounts (`--nv`/`--rocm`); OCI images |
| [Ch 34: UCX and libfabric](ch34_ucx_libfabric.md) | UCX `UCX_TLS` transports; libfabric `FI_PROVIDER`; Slingshot (`cxi`), Omni-Path (`psm2`), InfiniBand; OSU benchmarks |
| [Ch 35: Application-Level Checkpointing](ch35_checkpointing.md) | Multi-level hierarchy; SCR API (`SCR_Need_checkpoint`, `SCR_Route_file`); VeloC async flush; manual MPI-IO checkpointing |
| [Ch 36: mpi4py — MPI for Python](ch36_mpi4py.md) | Pickle vs buffer-protocol methods; NumPy zero-copy; all collectives; non-blocking; RMA; C interop; `MPIPoolExecutor` |

### Appendices (~10 pages)

| File | Content |
|---|---|
| [Appendix A: Function Reference](appendix_a_function_reference.md) | All functions grouped by category with C signatures |
| [Appendix B: Constants & Types](appendix_b_constants_types.md) | All `MPI_*` constants, predefined types, error codes |
| [Appendix C: Migration Notes](appendix_c_migration_notes.md) | MPI 3.x→4.0→5.0 changes, deprecations, porting guide |

---

## MPI Standard Coverage

| Feature | Standard | Chapter |
|---|---|---|
| Core P2P, collectives, datatypes | MPI 1.0 | Ch 3–15 |
| RMA, MPI-IO, dynamic processes | MPI 2.0 | Ch 16–20 |
| Non-blocking collectives, `MPI_T` | MPI 3.0 | Ch 11, 26 |
| Sessions, persistent collectives, partitioned comm, large count | MPI 4.0 | Ch 11, 22–24 |
| ULFM fault tolerance standardized | MPI 5.0 | Ch 4, 24 |
| C++20 `jthread` / `packaged_task` | C++20 | Ch 27 |
| SLURM job scheduling | — | Ch 28 |
| OpenSHMEM PGAS | OpenSHMEM 1.5 | Ch 29 |
| CUDA + NCCL | CUDA 11+ / NCCL 2.x | Ch 30 |
| GPUDirect Storage | CUDA 11.4+ / cuFile | Ch 31 |
| AMD ROCm / HIP + RCCL | ROCm 5.x+ | Ch 32 |
| Containers / Apptainer + PMIx | Apptainer 1.x | Ch 33 |
| UCX / libfabric transport layers | UCX 1.14+ / libfabric 1.18+ | Ch 34 |
| SCR / VeloC checkpointing | SCR 3.x / VeloC 1.x | Ch 35 |
| mpi4py Python bindings | mpi4py 3.x | Ch 36 |

---

## Scope

This guide covers C/C++ only. Fortran bindings, the full `MPI_T` specification,
dynamic process creation (`MPI_Comm_spawn`), and formal compliance requirements
are excluded — see the full MPI 5.0 specification for these.

Implementation notes for **Open MPI 5.x**, **MPICH 4.x**, **Intel MPI 2024+**,
and **Cray MPICH** (HPE Cray EX) are included throughout. Where behavior
differs between implementations, it is noted explicitly.

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
