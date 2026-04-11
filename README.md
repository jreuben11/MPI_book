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

1. [Ch 1: The MPI Model](ch01_mpi_model.md) — SPMD model, process ranks, communicators; standard evolution 1.x→5.0; MPI vs threads vs hybrid
2. [Ch 2: Setup & First Programs](ch02_setup_first_programs.md) — `mpicc`/`mpic++`, linking ABI; `mpirun`/`mpiexec`/PMIx launchers; Hello World; `MPI_Init`/`MPI_Finalize`
3. [Ch 3: Core Abstractions](ch03_core_abstractions.md) — Communicators, groups, contexts, tags; all handle types; buffer ownership; `MPI_Status`; `MPI_Info`
4. [Ch 4: Error Handling](ch04_error_handling.md) — Error classes and codes; `MPI_ERRORS_RETURN`, `MPI_ERRORS_ABORT`; custom handlers; `MPI_Abort`; ULFM overview

### Part II — Point-to-Point Communication (~35 pages)

5. [Ch 5: Blocking Send & Receive](ch05_blocking_p2p.md) — `MPI_Send`/`MPI_Recv` semantics and buffering; `MPI_Probe`/`MPI_Iprobe`; `MPI_Sendrecv`; deadlock taxonomy
6. [Ch 6: Non-Blocking Communication](ch06_nonblocking_p2p.md) — `MPI_Isend`/`MPI_Irecv` request lifecycle; `Wait`/`Test` variants; overlap patterns; halo exchange
7. [Ch 7: Send Modes](ch07_send_modes.md) — Standard, Buffered (`MPI_Bsend`), Synchronous (`MPI_Ssend`), Ready (`MPI_Rsend`); portability traps
8. [Ch 8: Persistent Requests](ch08_persistent_requests.md) — `MPI_Send_init`/`MPI_Recv_init`, `MPI_Start`, `MPI_Startall`; tight-loop reuse pattern

### Part III — Collective Communication (~30 pages)

9. [Ch 9: Core Collectives](ch09_core_collectives.md) — `MPI_Bcast`, `Scatter`/`Gather`, `Allgather`, `Reduce`, `Allreduce`, `Alltoall`, `Barrier`; `MPI_IN_PLACE`; custom `MPI_Op`
10. [Ch 10: Advanced Collectives](ch10_advanced_collectives.md) — Vector variants `Scatterv`/`Gatherv`/`Allgatherv`/`Alltoallv`/`w`; `Scan`/`Exscan`; `Reduce_scatter`; custom ops
11. [Ch 11: Non-Blocking & Persistent Collectives](ch11_nonblocking_persistent_collectives.md) — `MPI_Ibcast`, `MPI_Iallreduce`, etc.; MPI 4.0 `MPI_Bcast_init`/`MPI_Allreduce_init`; init/start/complete lifecycle

### Part IV — Datatypes (~20 pages)

12. [Ch 12: Built-in Datatypes](ch12_builtin_datatypes.md) — C type correspondence table; fixed-width types; `MPI_Count`, large-count `_c` variants (MPI 4.0); `MPI_Aint`, `MPI_Offset`
13. [Ch 13: Derived Datatypes](ch13_derived_datatypes.md) — Contiguous, vector, hvector, indexed, hindexed, struct; `MPI_Type_create_subarray`/`darray`; extent and resizing; worked examples

### Part V — Communicators & Topologies (~20 pages)

14. [Ch 14: Communicator Operations](ch14_communicator_operations.md) — `MPI_Comm_dup`, `split`, `split_type(SHARED_MEMORY)`, `create`, `create_group`; inter-communicators; keyvals
15. [Ch 15: Process Topologies](ch15_process_topologies.md) — Cartesian: `MPI_Cart_create`, `Cart_coords`, `Cart_shift`, `Cart_sub`; distributed graph topology; neighbor collectives

### Part VI — One-Sided Communication / RMA (~20 pages)

16. [Ch 16: Windows & Memory Models](ch16_rma_windows.md) — `MPI_Win_create`, `Win_allocate`, `Win_create_dynamic`; shared memory windows; unified vs separate memory models
17. [Ch 17: RMA Operations](ch17_rma_operations.md) — `MPI_Put`, `MPI_Get`, `MPI_Accumulate`; `Fetch_and_op`, `Compare_and_swap`; `MPI_Get_accumulate`, `Rput`/`Rget`
18. [Ch 18: RMA Synchronization](ch18_rma_synchronization.md) — Fence (active target); PSCW: `Win_post`/`Win_start`/`Win_complete`/`Win_wait`; passive: `Win_lock`/`flush`/`unlock`

### Part VII — Parallel I/O (~15 pages)

19. [Ch 19: MPI-IO Basics](ch19_mpio_basics.md) — `MPI_File_open`/`close`/`set_view`; individual vs collective I/O; `File_read_at_all`/`write_at_all`; info hints
20. [Ch 20: Advanced MPI-IO](ch20_advanced_mpio.md) — Shared file pointer; non-blocking I/O `File_iread_at`/`iwrite_at`; derived datatypes for non-contiguous layouts

### Part VIII — Advanced Topics (~25 pages)

21. [Ch 21: MPI + Threads](ch21_mpi_threads.md) — `MPI_Init_thread`; thread levels `SINGLE`→`MULTIPLE`; hybrid MPI+OpenMP patterns: funneled, serialized, multiple
22. [Ch 22: The Sessions Model — MPI 4.0](ch22_sessions_model.md) — `MPI_Session_init`/`finalize`; process sets; `MPI_Group` from sessions; library-safe initialization
23. [Ch 23: Partitioned Communication — MPI 4.0](ch23_partitioned_communication.md) — `MPI_Psend_init`/`Precv_init`; `MPI_Pready`/`Parrived`; GPU-direct and producer/consumer pipelines
24. [Ch 24: Large Counts & MPI 4.0/5.0 Additions](ch24_mpi4_mpi5_additions.md) — `MPI_Count`, `_c` suffix functions; `MPI_Comm_idup`; `MPI_Info_*` updates; MPI 5.0 ULFM and RMA clarifications

### Part IX — Performance & Tooling (~15 pages)

25. [Ch 25: Performance Patterns](ch25_performance_patterns.md) — Latency/bandwidth model; eager vs rendezvous; collective algorithm selection; synchronization bottlenecks; NUMA
26. [Ch 26: Profiling & Debugging](ch26_profiling_debugging.md) — PMPI profiling layer; `MPI_T` CVars and PVars; Score-P, TAU, mpiP, ITAC; bug checklist

### Part X — Integrations (~85 pages)

27. [Ch 27: MPI with C++20 Threads](ch27_cpp20_threads.md) — `std::jthread` RAII; `std::stop_token` progress threads; `std::packaged_task`+`future` async MPI; `std::barrier` coordination
28. [Ch 28: MPI with SLURM](ch28_slurm.md) — `srun` vs `mpiexec`; batch script anatomy; SLURM env vars; process binding; PMIx bootstrapping; job arrays; Lustre I/O
29. [Ch 29: OpenSHMEM](ch29_openshmem.md) — PE model, symmetric heap; `shmem_put`/`get`; synchronization; atomics; OpenSHMEM 1.5 teams; MPI interop
30. [Ch 30: MPI with CUDA and NCCL](ch30_cuda_nccl.md) — GPU-aware MPI; device pointer sends; CUDA streams + non-blocking overlap; NCCL init and collectives; MPI vs NCCL
31. [Ch 31: GPUDirect Storage](ch31_gpudirect_storage.md) — GDS architecture; cuFile API; `O_DIRECT` alignment; MPI+GDS pattern; GDS+partitioned comm; checkpoint/restart
32. [Ch 32: AMD ROCm and HIP](ch32_rocm_hip.md) — HIP API; `hipify` porting; ROCm-aware MPI; Infinity Fabric peer access; RCCL collectives; MI300X unified memory
33. [Ch 33: MPI in Containers](ch33_containers.md) — Host MPI vs PMIx-only models; Singularity/Apptainer definition files; GPU bind-mounts (`--nv`/`--rocm`); OCI images
34. [Ch 34: UCX and libfabric](ch34_ucx_libfabric.md) — UCX `UCX_TLS` transports; libfabric `FI_PROVIDER`; Slingshot (`cxi`), Omni-Path (`psm2`), InfiniBand; OSU benchmarks
35. [Ch 35: Application-Level Checkpointing](ch35_checkpointing.md) — Multi-level hierarchy; SCR API (`SCR_Need_checkpoint`, `SCR_Route_file`); VeloC async flush; manual MPI-IO checkpointing
36. [Ch 36: mpi4py — MPI for Python](ch36_mpi4py.md) — Pickle vs buffer-protocol methods; NumPy zero-copy; all collectives; non-blocking; RMA; C interop; `MPIPoolExecutor`

### Appendices (~10 pages)

- [Appendix A: Function Reference](appendix_a_function_reference.md) — All functions grouped by category with C signatures
- [Appendix B: Constants & Types](appendix_b_constants_types.md) — All `MPI_*` constants, predefined types, error codes
- [Appendix C: Migration Notes](appendix_c_migration_notes.md) — MPI 3.x→4.0→5.0 changes, deprecations, porting guide

---

## MPI Standard Coverage

- **MPI 1.0** — Core P2P, collectives, datatypes (Ch 3–15)
- **MPI 2.0** — RMA, MPI-IO, dynamic processes (Ch 16–20)
- **MPI 3.0** — Non-blocking collectives, `MPI_T` (Ch 11, 26)
- **MPI 4.0** — Sessions, persistent collectives, partitioned comm, large count (Ch 11, 22–24)
- **MPI 5.0** — ULFM fault tolerance, ABI standardization (Ch 4, 24)
- **C++20** — `jthread` / `packaged_task` (Ch 27)
- **SLURM** — Job scheduling (Ch 28)
- **OpenSHMEM 1.5** — PGAS (Ch 29)
- **CUDA 11+ / NCCL 2.x** — GPU + collectives (Ch 30)
- **CUDA 11.4+ / cuFile** — GPUDirect Storage (Ch 31)
- **ROCm 5.x+** — AMD ROCm / HIP + RCCL (Ch 32)
- **Apptainer 1.x** — Containers + PMIx (Ch 33)
- **UCX 1.14+ / libfabric 1.18+** — Transport layers (Ch 34)
- **SCR 3.x / VeloC 1.x** — Checkpointing (Ch 35)
- **mpi4py 3.x** — Python bindings (Ch 36)

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
