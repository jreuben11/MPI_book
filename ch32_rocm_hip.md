# Chapter 32: MPI with ROCm/HIP and RCCL

## 32.1 The AMD GPU Ecosystem

AMD's GPU compute stack consists of:

- **ROCm** (Radeon Open Compute): the open-source platform stack — drivers, runtime,
  math libraries. Analogous to the CUDA platform.
- **HIP** (Heterogeneous-Computing Interface for Portability): the C++ programming
  API. Intentionally mirrors CUDA — most CUDA code can be mechanically ported.
- **RCCL** (ROCm Collective Communications Library): AMD's equivalent of NCCL,
  optimized for AMD interconnects (Infinity Fabric / xGMI between GPUs, InfiniBand
  across nodes).

Major HPC systems using AMD GPUs include Frontier (Oak Ridge, #1 Top500 2022–2024,
AMD MI250X), LUMI (EuroHPC, MI250X), and El Capitan (LLNL, MI300X).

The MPI integration patterns mirror Chapter 30 (CUDA/NCCL) but with different
environment variables, library names, and some API differences worth knowing.

---

## 32.2 HIP Programming Basics

HIP provides a CUDA-compatible API. For programs already using CUDA, a header
compatibility layer allows compilation with either backend:

```cpp
/* Portable header — works with both CUDA and ROCm */
#ifdef __HIP_PLATFORM_AMD__
  #include <hip/hip_runtime.h>
#else
  #include <cuda_runtime.h>
  /* hipXxx → cudaXxx via inline wrappers */
#endif
```

When writing AMD-native code, include HIP directly:

```cpp
#include <hip/hip_runtime.h>
```

### CUDA → HIP API Mapping (Selected)

| CUDA | HIP | Notes |
|---|---|---|
| `cudaMalloc` | `hipMalloc` | Identical semantics |
| `cudaFree` | `hipFree` | |
| `cudaMemcpy` | `hipMemcpy` | |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | |
| `cudaSetDevice` | `hipSetDevice` | |
| `cudaGetDeviceCount` | `hipGetDeviceCount` | |
| `cudaStreamCreate` | `hipStreamCreate` | |
| `cudaStreamSynchronize` | `hipStreamSynchronize` | |
| `cudaGetErrorString` | `hipGetErrorString` | |
| `__global__` kernel attribute | `__global__` | Same |
| `<<<grid, block, smem, stream>>>` | Same syntax | Same |
| `threadIdx`, `blockIdx`, etc. | Same | Same |

### Automated PORTING: hipify

NVIDIA-to-AMD conversion tool:

```bash
hipify-clang myprogram.cu -o myprogram.hip.cpp
# or for whole directories:
hipify-perl --inplace src/*.cu
```

`hipify` converts `cuda*` → `hip*`, `CUDA_*` → `HIP_*`, and updates headers.
Most CUDA code converts cleanly; exceptions involve CUDA-specific extensions
(cooperative groups, CUDA graphs with some features) that need manual attention.

---

## 32.3 ROCm-Aware MPI

Like GPU-aware MPI for CUDA (Chapter 30), ROCm-aware MPI allows passing HIP device
pointers directly to MPI functions:

```
Without ROCm-aware MPI:
  GPU kernel → hipMemcpy D2H → MPI_Send → MPI_Recv → hipMemcpy H2D → GPU

With ROCm-aware MPI:
  GPU kernel → MPI_Send(device_ptr) → MPI_Recv(device_ptr) → GPU kernel
```

### Checking ROCm-Aware Support

```c
#include <mpi.h>
#include <mpi-ext.h>   /* Open MPI extension header */

void check_rocm_aware_mpi(void)
{
/* Open MPI does not provide MPIX_Query_rocm_support() — check build config instead */
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    /* CUDA-aware check exists; ROCm-aware has no equivalent runtime query */
    fprintf(stderr, "ROCm-awareness: check 'ompi_info | grep -i rocm'\n");
#else
    fprintf(stderr, "ROCm-awareness unknown — verify MPI build with 'ompi_info --all'\n");
#endif
}
```

### Runtime Environment for ROCm-Aware Open MPI

```bash
# UCX with ROCm transport (preferred)
export UCX_TLS=rc,rocm_copy,rocm_ipc,gdr_copy
export UCX_ROCM_COPY_MAX_REG_RATIO=1.0

# Open MPI OPAL ROCM support
export OMPI_MCA_opal_rocm_support=1

# For Cray MPICH on Frontier/LUMI
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
```

### Sending HIP Device Memory

```cpp
#include <mpi.h>
#include <hip/hip_runtime.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Assign one GPU per MPI rank */
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        rank, MPI_INFO_NULL, &node_comm);
    int local_rank;
    MPI_Comm_rank(node_comm, &local_rank);
    hipSetDevice(local_rank);
    MPI_Comm_free(&node_comm);

    const int N = 1 << 20;
    double *d_buf;
    hipMalloc(&d_buf, N * sizeof(double));

    /* Fill on GPU */
    fill_kernel<<<(N+255)/256, 256>>>(d_buf, N, rank);
    hipDeviceSynchronize();

    /* Direct device-pointer communication */
    if (rank == 0)
        MPI_Send(d_buf, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    else if (rank == 1)
        MPI_Recv(d_buf, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    hipFree(d_buf);
    MPI_Finalize();
}
```

Compile:
```bash
hipcc -O2 -o mpi_hip mpi_hip.cpp \
    $(mpicc -showme:compile) $(mpicc -showme:link)
```

---

## 32.4 HIP Streams and Non-Blocking MPI

```cpp
hipStream_t compute_stream, comm_stream;
hipStreamCreate(&compute_stream);
hipStreamCreate(&comm_stream);

double *d_interior, *d_halo_send, *d_halo_recv;
/* ... allocate ... */

/* Interior computation on compute_stream */
interior_kernel<<<grid, block, 0, compute_stream>>>(d_interior, N);

/* Pack halo on comm_stream (concurrent) */
pack_halo_kernel<<<hgrid, hblock, 0, comm_stream>>>(d_halo_send, d_data);
hipStreamSynchronize(comm_stream);   /* halo packed */

/* MPI exchange with device pointers */
MPI_Request reqs[2];
MPI_Isend(d_halo_send, HALO, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &reqs[0]);
MPI_Irecv(d_halo_recv, HALO, MPI_DOUBLE, left,  0, MPI_COMM_WORLD, &reqs[1]);

hipStreamSynchronize(compute_stream);
MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

apply_halo_kernel<<<hgrid, hblock>>>(d_data, d_halo_recv);
```

---

## 32.5 AMD Infinity Fabric and xGMI

AMD's intra-node GPU interconnect is **Infinity Fabric** (MI250X: 4 GCDs per node
with xGMI links) and **NVLink**-equivalent **xGMI**. Unlike CUDA's `cudaMemcpyPeer`,
AMD uses:

```cpp
/* Enable peer access between GPUs on same node */
int can_access;
hipDeviceCanAccessPeer(&can_access, device_a, device_b);
if (can_access) {
    hipSetDevice(device_a);
    hipDeviceEnablePeerAccess(device_b, 0);
}

/* Cross-GCD memory copy without staging through host */
hipMemcpyPeerAsync(d_dst, device_b,
                   d_src, device_a,
                   nbytes, stream);
```

On MI250X, each card has 2 GCDs (Graphics Compute Dies). Each GCD appears as a
separate device. The two GCDs on the same card share high-bandwidth xGMI links
(~200 GB/s) superior to PCIe. Peer access is automatically available between
co-located GCDs.

Query topology:

```cpp
/* Check if two devices share xGMI (Infinity Fabric) */
uint32_t link_type;
hipExtGetLinkTypeAndHopCount(device_a, device_b, &link_type, NULL);
/* HIPDEVICE_P2P_LINK_XGMI = xGMI/Infinity Fabric */
```

---

## 32.6 RCCL: ROCm Collective Communications Library

RCCL provides the same collective API as NCCL with AMD-specific topology optimizations.
The initialization pattern is identical:

```cpp
#include <mpi.h>
/* RCCL uses ncclXxx names directly — include rccl/rccl.h, then use nccl* API */
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

ncclComm_t init_rccl(MPI_Comm mpi_comm)
{
    int rank, size;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    /* Rank 0 generates unique ID; all ranks get it via MPI broadcast */
    ncclUniqueId rccl_id;
    if (rank == 0) ncclGetUniqueId(&rccl_id);
    MPI_Bcast(&rccl_id, sizeof(rccl_id), MPI_BYTE, 0, mpi_comm);

    ncclComm_t comm;
    ncclCommInitRank(&comm, size, rccl_id, rank);
    return comm;
}
```

### RCCL Allreduce

```cpp
void rccl_allreduce(ncclComm_t rccl_comm, hipStream_t stream)
{
    const int N = 1 << 22;
    float *d_buf;
    hipMalloc(&d_buf, N * sizeof(float));

    fill_kernel<<<(N+255)/256, 256, 0, stream>>>(d_buf, N);

    /* In-place Allreduce across all GPUs — RCCL uses ncclXxx names */
    ncclAllReduce(d_buf, d_buf, N, ncclFloat, ncclSum, rccl_comm, stream);

    hipStreamSynchronize(stream);
    /* d_buf now holds the global sum on all ranks */

    hipFree(d_buf);
}
```

### RCCL API: Same Names as NCCL

RCCL deliberately uses the **identical** `ncclXxx` API names as NCCL. After including
`<rccl/rccl.h>`, you use `ncclUniqueId`, `ncclComm_t`, `ncclGetUniqueId`, etc. —
there is no `rccl` prefix in the actual API. This means NCCL-based code typically
compiles against RCCL by simply changing the include path:

```cpp
/* NCCL build: */
#include <nccl.h>

/* RCCL build: */
#include <rccl/rccl.h>
/* After either include, the API is identical: ncclUniqueId, ncclComm_t, etc. */
```

| Concept | API name (both NCCL and RCCL) |
|---|---|
| Communicator | `ncclComm_t` |
| Unique ID | `ncclUniqueId`, `ncclGetUniqueId` |
| Init | `ncclCommInitRank` |
| Allreduce | `ncclAllReduce` |
| Broadcast | `ncclBroadcast` |
| Datatypes | `ncclFloat`, `ncclDouble`, `ncclHalf` |
| Ops | `ncclSum`, `ncclProd`, `ncclMax`, `ncclMin` |
| P2P | `ncclSend`, `ncclRecv` (inside Group) |
| Error | `ncclResult_t`, `ncclSuccess`, `ncclGetErrorString` |

---

## 32.7 Error Handling

```cpp
#define HIP_CHECK(call)                                              \
    do {                                                              \
        hipError_t _e = (call);                                      \
        if (_e != hipSuccess) {                                      \
            fprintf(stderr, "HIP error %s:%d: %s\n",                \
                    __FILE__, __LINE__, hipGetErrorString(_e));      \
            MPI_Abort(MPI_COMM_WORLD, 1);                           \
        }                                                             \
    } while (0)

/* RCCL uses ncclResult_t and ncclSuccess — same as NCCL */
#define RCCL_CHECK(call)                                             \
    do {                                                              \
        ncclResult_t _r = (call);                                    \
        if (_r != ncclSuccess) {                                     \
            fprintf(stderr, "RCCL error %s:%d: %s\n",               \
                    __FILE__, __LINE__, ncclGetErrorString(_r));     \
            MPI_Abort(MPI_COMM_WORLD, 1);                           \
        }                                                             \
    } while (0)
```

---

## 32.8 Writing Portable CUDA/HIP Code

For code that must run on both NVIDIA and AMD:

```cpp
/* Portability shim — one header covers both backends */
#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
  #define gpuMalloc       hipMalloc
  #define gpuFree         hipFree
  #define gpuMemcpy       hipMemcpy
  #define gpuDeviceSync   hipDeviceSynchronize
  #define gpuSetDevice    hipSetDevice
  #define gpuGetDevCount  hipGetDeviceCount
  #define gpuStream_t     hipStream_t
  #define gpuMemcpyH2D    hipMemcpyHostToDevice
  #define gpuMemcpyD2H    hipMemcpyDeviceToHost
  #include <rccl/rccl.h>
  #define gpuCommT        rcclComm_t
  /* ... map all nccl* → rccl* ... */
#else
  #include <cuda_runtime.h>
  #define gpuMalloc       cudaMalloc
  #define gpuFree         cudaFree
  /* ... etc ... */
  #include <nccl.h>
  #define gpuCommT        ncclComm_t
#endif
```

Alternatively, Kokkos (Chapter 36) provides this abstraction at a higher level.

---

## 32.9 Build and SLURM Launch

### CMake

```cmake
find_package(HIP REQUIRED)
find_package(MPI REQUIRED)
find_library(RCCL_LIB rccl HINTS ${ROCM_PATH}/lib)
find_path(RCCL_INC rccl/rccl.h HINTS ${ROCM_PATH}/include)

add_executable(myprogram main.hip.cpp)
set_source_files_properties(main.hip.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(myprogram PRIVATE ${RCCL_INC})
target_link_libraries(myprogram hip::host MPI::MPI_CXX ${RCCL_LIB})
```

### SLURM on Frontier / LUMI

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8     # 8 MPI ranks = 8 GCDs per node (MI250X)
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest      # bind each rank to its nearest GCD

module load PrgEnv-amd rocm rccl

export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_DEBUG=INFO           # works for RCCL too (env var compatible)

srun ./myprogram
```

On Frontier, each MI250X node has 4 cards × 2 GCDs = 8 devices. With
`--ntasks-per-node=8` and `--gpus-per-task=1`, each MPI rank gets one GCD.

---

## 32.10 MI300X Unified Memory

The AMD Instinct MI300X (2024) integrates CPU and GPU memory into a single pool —
**unified HBM** with ~128–192 GB shared between CPU and GPU. This changes the
MPI communication model:

```cpp
/* On MI300X: use hipMallocManaged for truly unified CPU+GPU accessible memory.
   hipMalloc alone does NOT return CPU-accessible memory even on MI300X —
   dereferencing a plain hipMalloc pointer from the CPU will segfault.
   Use hipMallocManaged (or hipHostMalloc with appropriate flags) for CPU access. */

double *buf;
hipMallocManaged(&buf, N * sizeof(double));   /* unified CPU+GPU memory */

/* CPU can now read/write buf directly */
for (int i = 0; i < N; i++) buf[i] = initial_value(i);

/* GPU can compute on buf */
compute_kernel<<<grid, block>>>(buf, N);
hipDeviceSynchronize();

/* MPI send — buffer is CPU-accessible, no ROCm-aware MPI required */
MPI_Send(buf, N, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
hipFree(buf);
```

This simplifies MPI integration significantly but requires knowing your hardware.
Check whether the allocation is CPU-accessible:

```cpp
hipPointerAttribute_t attr;
hipPointerGetAttributes(&attr, buf);
bool cpu_accessible = (attr.memoryType == hipMemoryTypeUnified ||
                       attr.isManaged);
```

---

## Summary

| Topic | Key Points |
|---|---|
| HIP API | Mirrors CUDA; `hipMalloc/hipFree/hipMemcpy`; `hipSetDevice` |
| `hipify` | Automated CUDA→HIP conversion tool |
| ROCm-aware MPI | `MPICH_GPU_SUPPORT_ENABLED=1` (Cray) or `UCX_TLS=rc,rocm_copy,rocm_ipc` (UCX) |
| Local rank assignment | Same `MPI_COMM_TYPE_SHARED` pattern as CUDA |
| Peer access | `hipDeviceEnablePeerAccess`; xGMI links on same node |
| RCCL | Uses `ncclXxx` API names directly; include `<rccl/rccl.h>` |
| RCCL init | `rcclGetUniqueId` on rank 0 → `MPI_Bcast` → `rcclCommInitRank` |
| Portability | `#ifdef USE_ROCM` shim or Kokkos abstraction layer |
| MI300X | Unified CPU+GPU memory; no H2D copies; simplifies MPI integration |
| Frontier/LUMI | `--ntasks-per-node=8` for MI250X (8 GCDs); `--gpu-bind=closest` |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
