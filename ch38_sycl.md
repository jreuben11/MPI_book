# Chapter 38: MPI with SYCL

## 38.1 What Is SYCL?

SYCL (pronounced "sickle") is a Khronos Group open standard for portable
heterogeneous computing in single-source C++. It targets CPUs, GPUs, FPGAs, and
accelerators through a unified programming model, in contrast to CUDA (NVIDIA-only)
and HIP (AMD-primary, with NVIDIA support via translation).

```
Portability stack:
  SYCL source (.cpp)
       │
       ├──► Intel oneAPI DPC++ (icpx)  → Intel GPU / CPU / FPGA
       ├──► AdaptiveCpp (acpp)          → NVIDIA (CUDA), AMD (HIP), Intel, CPU
       └──► triSYCL / ComputeCpp        → deprecated / research
```

**Key concepts** that differ from CUDA:

| SYCL | CUDA analogue |
|---|---|
| `sycl::queue` | stream (`cudaStream_t`) + device selection |
| `sycl::buffer<T>` | device memory with accessor-controlled transfers |
| USM (`malloc_device`) | `cudaMalloc` |
| `sycl::handler::parallel_for` | kernel launch (`<<<>>>`) |
| `sycl::nd_item` | thread index (`threadIdx`, `blockIdx`) |
| `q.wait()` | `cudaStreamSynchronize` |

SYCL 2020 is the current standard. Intel oneAPI DPC++ and AdaptiveCpp both support
SYCL 2020.

---

## 38.2 Installation and Setup

### Intel oneAPI DPC++

```bash
# Install Intel oneAPI Base Toolkit (includes icpx, Intel MPI, MKL)
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/oneapi-for-linux/...
sudo sh <installer> -a --silent --cli --eula accept

source /opt/intel/oneapi/setvars.sh   # sets PATH, LD_LIBRARY_PATH, etc.
```

```bash
# Compile with icpx (Intel SYCL compiler)
icpx -fsycl -O2 -o hello hello.cpp

# MPI + SYCL with Intel MPI
mpiicpx -fsycl -O2 -o solver solver.cpp
# mpiicpx is the Intel MPI wrapper for icpx (analogous to mpic++)
```

### AdaptiveCpp (formerly HipSYCL) — Cross-Vendor

```bash
# From source or pre-built packages — see adaptivecpp.github.io
# Target selection at compile time:
acpp --acpp-targets="cuda:sm_80"  solver.cpp   # NVIDIA A100
acpp --acpp-targets="hip:gfx90a"  solver.cpp   # AMD MI250X
acpp --acpp-targets="omp"         solver.cpp   # CPU via OpenMP
```

### CMake

```cmake
cmake_minimum_required(VERSION 3.25)
project(sycl_mpi)

find_package(MPI REQUIRED)
find_package(IntelSYCL REQUIRED)   # Intel oneAPI

add_executable(solver solver.cpp)
add_sycl_to_target(TARGET solver SOURCES solver.cpp)
target_link_libraries(solver PRIVATE MPI::MPI_CXX)
```

---

## 38.3 SYCL Memory Models

SYCL offers two memory management styles. **Unified Shared Memory (USM)** is the
modern approach and maps most directly onto CUDA-aware MPI patterns.

### Buffers + Accessors (SYCL 1.2.1 style)

```cpp
#include <sycl/sycl.hpp>

sycl::queue q{sycl::gpu_selector_v};

std::vector<double> host_data(N, 1.0);

{   /* buffer lifetime controls host↔device synchronisation */
    sycl::buffer<double> buf(host_data.data(), sycl::range<1>(N));

    q.submit([&](sycl::handler &h) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            acc[i] *= 2.0;
        });
    });
}   /* buffer destructor copies data back to host_data */
/* host_data is now updated — safe to pass to MPI */
```

The buffer/accessor model is implicit transfer: the runtime moves data as needed.
This makes it harder to reason about exactly when transfers occur relative to MPI
calls, so USM is generally preferred for MPI integration.

### Unified Shared Memory (USM)

```cpp
#include <sycl/sycl.hpp>

sycl::queue q{sycl::gpu_selector_v};

/* Device allocation — lives on GPU, not accessible from CPU without copy */
double *d_ptr = sycl::malloc_device<double>(N, q);

/* Shared allocation — accessible from both CPU and GPU (via PCIe/NVLink page migration) */
double *s_ptr = sycl::malloc_shared<double>(N, q);

/* Host allocation — pinned host memory; fast DMA to GPU */
double *h_ptr = sycl::malloc_host<double>(N, q);

/* ... use ... */
sycl::free(d_ptr, q);
sycl::free(s_ptr, q);
sycl::free(h_ptr, q);
```

| USM kind | CPU access | GPU access | MPI (standard) | MPI (SYCL-aware) |
|---|---|---|---|---|
| `malloc_device` | No | Yes | No (copy required) | Yes |
| `malloc_shared` | Yes | Yes | Yes (page migration) | Yes |
| `malloc_host` | Yes | Yes (via DMA) | Yes (pinned) | Yes |

---

## 38.4 Basic SYCL + MPI Pattern

For standard MPI (not SYCL-aware), copy device data to host before sending:

```cpp
#include <sycl/sycl.hpp>
#include <mpi.h>
#include <vector>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sycl::queue q{sycl::gpu_selector_v};

    const int N = 1 << 20;

    /* Device allocation */
    double *d_data = sycl::malloc_device<double>(N, q);

    /* Host staging buffer — pinned for fast transfer */
    double *h_send = sycl::malloc_host<double>(N, q);
    double *h_recv = sycl::malloc_host<double>(N, q);

    /* Compute on GPU */
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        d_data[i] = static_cast<double>(rank) + i[0] * 0.001;
    }).wait();

    /* Copy device → host staging */
    q.memcpy(h_send, d_data, N * sizeof(double)).wait();

    /* MPI exchange with neighbour */
    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    MPI_Sendrecv(h_send, N, MPI_DOUBLE, right, 0,
                 h_recv, N, MPI_DOUBLE, left,  0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Copy received data back to device */
    q.memcpy(d_data, h_recv, N * sizeof(double)).wait();

    /* Continue GPU computation with updated data */
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        d_data[i] += 1.0;
    }).wait();

    sycl::free(d_data, q);
    sycl::free(h_send, q);
    sycl::free(h_recv, q);
    MPI_Finalize();
}
```

---

## 38.5 SYCL-Aware MPI

Intel MPI (2021.3+) supports passing USM device pointers directly to MPI when
compiled with `icpx -fsycl` and run with Level Zero or OpenCL backend. This
eliminates the explicit device→host→device copies.

### Querying SYCL-Aware Support

There is no standardised MPIX query for SYCL awareness (contrast with
`MPIX_Query_cuda_support()` from Chapter 30). Intel MPI exposes it via an
environment variable check:

```bash
# Enable SYCL-aware MPI in Intel MPI
export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_TOPOLIB=l0   # Level Zero backend

mpirun -n 4 ./solver
```

At runtime, Intel MPI detects Level Zero devices and uses peer-to-peer DMA
when both source and destination ranks are on the same node with compatible
hardware. Cross-node transfers fall back to host staging automatically.

### SYCL-Aware Send with Device Pointer

```cpp
#include <sycl/sycl.hpp>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sycl::queue q{sycl::gpu_selector_v};
    const int N = 1 << 20;

    double *d_send = sycl::malloc_device<double>(N, q);
    double *d_recv = sycl::malloc_device<double>(N, q);

    /* Fill on GPU */
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        d_send[i] = static_cast<double>(rank);
    }).wait();

    /* Pass device pointers directly to MPI — requires SYCL-aware MPI */
    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    MPI_Sendrecv(d_send, N, MPI_DOUBLE, right, 0,
                 d_recv, N, MPI_DOUBLE, left,  0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* d_recv is now on device — no host copy needed */

    sycl::free(d_send, q);
    sycl::free(d_recv, q);
    MPI_Finalize();
}
```

---

## 38.6 Overlapping Compute and Communication

Non-blocking MPI + SYCL queue submission allows overlap:

```cpp
#include <sycl/sycl.hpp>
#include <mpi.h>

void halo_exchange_sycl(sycl::queue &q,
                         double *d_interior, double *d_halo_left, double *d_halo_right,
                         double *h_send_l, double *h_send_r,
                         double *h_recv_l, double *h_recv_r,
                         int halo_size, int left, int right)
{
    /* Pack halos on GPU */
    q.memcpy(h_send_l, d_halo_left,  halo_size * sizeof(double));
    q.memcpy(h_send_r, d_halo_right, halo_size * sizeof(double));
    q.wait();   /* ensure packing complete before MPI */

    MPI_Request reqs[4];
    MPI_Irecv(h_recv_l, halo_size, MPI_DOUBLE, left,  1, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(h_recv_r, halo_size, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(h_send_l, halo_size, MPI_DOUBLE, left,  0, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(h_send_r, halo_size, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &reqs[3]);

    /* Compute interior while halos are in flight */
    q.parallel_for(sycl::range<1>(INTERIOR_SIZE), [=](sycl::id<1> i) {
        compute_interior(d_interior, i[0]);
    });   /* do NOT .wait() yet — let it overlap with MPI */

    /* Wait for both MPI and GPU interior to complete */
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    q.wait();

    /* Unpack received halos to device */
    q.memcpy(d_halo_left,  h_recv_l, halo_size * sizeof(double));
    q.memcpy(d_halo_right, h_recv_r, halo_size * sizeof(double));
    q.wait();
}
```

---

## 38.7 Multi-GPU with MPI: One Rank per GPU

```cpp
#include <sycl/sycl.hpp>
#include <mpi.h>
#include <vector>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Enumerate GPUs */
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> gpus;
    for (auto &p : platforms)
        for (auto &d : p.get_devices(sycl::info::device_type::gpu))
            gpus.push_back(d);

    if (gpus.empty()) {
        fprintf(stderr, "Rank %d: no GPU found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Round-robin assignment: rank % ngpus */
    sycl::device my_gpu = gpus[rank % gpus.size()];
    sycl::queue q{my_gpu};

    auto name = my_gpu.get_info<sycl::info::device::name>();
    printf("Rank %d using: %s\n", rank, name.c_str());

    /* ... per-rank computation and MPI exchange ... */

    MPI_Finalize();
}
```

On a node with 4 GPUs and `mpirun -n 4`, each rank claims one GPU. The
round-robin modulo ensures that with more ranks than GPUs, GPUs are shared
(multiple ranks share one GPU's context — valid but may reduce throughput).

For deterministic assignment on SLURM:

```bash
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

# GPU affinity via SLURM — each rank sees only its assigned GPU
export SYCL_DEVICE_FILTER=level_zero   # Intel; use "cuda" for NVIDIA
srun --gpu-bind=closest ./solver
```

---

## 38.8 SYCL Group Algorithms and Reductions

SYCL 2020 provides group-level primitives (warp/subgroup and work-group reductions)
that replace hand-written shared memory reduction kernels:

```cpp
#include <sycl/sycl.hpp>

sycl::queue q{sycl::gpu_selector_v};
const int N = 1 << 20;

double *d_data = sycl::malloc_device<double>(N, q);
double *d_result = sycl::malloc_shared<double>(1, q);
*d_result = 0.0;

/* Work-group reduction via sycl::reduction */
q.submit([&](sycl::handler &h) {
    auto reduction = sycl::reduction(d_result, sycl::plus<double>{});
    h.parallel_for(sycl::nd_range<1>{N, 256}, reduction,
                   [=](sycl::nd_item<1> item, auto &sum) {
                       sum += d_data[item.get_global_id()];
                   });
}).wait();

double local_sum = *d_result;

/* Global reduction across MPI ranks */
double global_sum;
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
```

---

## 38.9 SYCL vs CUDA vs HIP

| | CUDA | HIP | SYCL |
|---|---|---|---|
| Standard body | NVIDIA proprietary | AMD open-source | Khronos open standard |
| Primary hardware | NVIDIA | AMD (+ NVIDIA) | Intel, AMD, NVIDIA, CPU, FPGA |
| Language | C++ extension | C++ extension | Standard C++ |
| Single source | Yes | Yes | Yes |
| MPI-aware | CUDA-aware MPI (all vendors) | ROCm-aware MPI | Intel MPI (Level Zero) |
| Porting tool | — | `hipify` (from CUDA) | — |
| Compiler | `nvcc` / `nvc++` | `hipcc` | `icpx -fsycl` / `acpp` |
| NCCL / RCCL equiv | NCCL | RCCL | oneCCL (Intel) |

Choose SYCL when:
- Target hardware is Intel GPU (Ponte Vecchio, Gaudi, Xe) — SYCL is the native model
- Portability across Intel / AMD / NVIDIA is required
- Using Intel oneAPI MKL, oneDNN, or oneCCL in the same application

Choose CUDA when NVIDIA hardware is the sole target and maximum ecosystem support
(cuBLAS, cuDNN, NCCL) is needed. See Chapter 30.

Choose HIP when targeting AMD hardware with occasional NVIDIA fallback. See Chapter 32.

---

## Summary

| Topic | Key Points |
|---|---|
| SYCL programming model | `sycl::queue` + `parallel_for`; single-source C++; Khronos standard |
| USM memory kinds | `malloc_device` (GPU-only), `malloc_shared` (migratable), `malloc_host` (pinned) |
| Standard MPI pattern | `q.memcpy` device→`malloc_host` → MPI send/recv → `q.memcpy` host→device |
| SYCL-aware MPI | Intel MPI 2021.3+ with `I_MPI_OFFLOAD=1`; pass USM device pointers directly |
| Overlap pattern | `MPI_Isend/Irecv` + `q.parallel_for(interior)` in parallel; `MPI_Waitall` + `q.wait()` |
| Multi-GPU | One rank per GPU; `sycl::platform::get_devices(gpu)`; round-robin or SLURM binding |
| Group reductions | `sycl::reduction` + `MPI_Allreduce` for distributed reduce |
| Compiler | `icpx -fsycl` (Intel oneAPI); `acpp --acpp-targets=...` (AdaptiveCpp) |
| vs CUDA/HIP | SYCL for Intel hardware or cross-vendor portability; CUDA for NVIDIA ecosystem depth |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
