# Chapter 30: MPI with CUDA and NCCL

## 30.1 The GPU Communication Landscape

GPU-accelerated MPI programs face a fundamental challenge: MPI traditionally operates
on CPU-accessible (host) memory, while computation happens in GPU (device) memory.
Naively, this requires an extra copy at every communication boundary:

```
Without GPU-aware MPI:
  GPU kernel → cudaMemcpy to host → MPI_Send → MPI_Recv → cudaMemcpy to GPU

With GPU-aware MPI:
  GPU kernel → MPI_Send(device_ptr) → MPI_Recv(device_ptr) → GPU kernel
```

Two technologies eliminate the extra copies:

1. **GPU-aware MPI**: the MPI implementation understands CUDA device pointers and
   uses GDR (GPUDirect RDMA) to transfer directly between GPU memory and the network.

2. **NCCL (NVIDIA Collective Communications Library)**: NVIDIA's alternative
   collective library, optimized for GPU topology. Often faster than MPI collectives
   on GPU clusters; does not replace MPI for point-to-point.

---

## 30.2 GPU-Aware MPI

### Checking Support at Runtime

```c
#include <mpi.h>
#include <mpi-ext.h>   /* CUDA-aware query — OpenMPI specific */

void check_gpu_aware_mpi(void)
{
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    if (MPIX_Query_cuda_support()) {
        printf("MPI is CUDA-aware (compile-time check passed)\n");
    } else {
        fprintf(stderr, "MPI is NOT CUDA-aware — using host copies\n");
    }
#else
    fprintf(stderr, "OMPI_HAVE_MPI_EXT_CUDA not defined — unknown\n");
#endif
}
```

Enable GPU-aware MPI at runtime (Open MPI):

```bash
# Enable GPUDirect
export OMPI_MCA_btl_openib_want_cuda_gdr=1
export OMPI_MCA_opal_cuda_support=1

# Or with UCX transport (preferred for modern systems)
export UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy
```

For MPICH / Cray MPICH:
```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_RDMA_ENABLED_CUDA=1
```

### Sending Device Memory

```cuda
#include <mpi.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Set the GPU for this rank (one GPU per MPI rank is the standard pattern) */
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    cudaSetDevice(rank % num_gpus);

    const int N = 1 << 20;   /* 1M elements */
    double *d_buf;
    cudaMalloc(&d_buf, N * sizeof(double));

    /* Fill buffer with a GPU kernel */
    fill_kernel<<<(N+255)/256, 256>>>(d_buf, N, rank);
    cudaDeviceSynchronize();

    /* Send device pointer directly — GPU-aware MPI handles the transfer */
    if (rank == 0) {
        MPI_Send(d_buf, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(d_buf, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* d_buf on rank 1 now contains rank 0's data — no host copies */
    }

    cudaFree(d_buf);
    MPI_Finalize();
}
```

Compile:
```bash
nvcc -O2 -o mpi_gpu mpi_gpu.cu $(mpicc -showme:compile) $(mpicc -showme:link)
# Or use mpic++ with CUDA linkage:
mpic++ -O2 -o mpi_gpu mpi_gpu.cu -lcudart
```

### Non-Blocking with CUDA Streams

```cuda
/* Overlap GPU computation with MPI communication using streams */

cudaStream_t compute_stream, comm_stream;
cudaStreamCreate(&compute_stream);
cudaStreamCreate(&comm_stream);

double *d_interior, *d_halo_send, *d_halo_recv;
/* ... allocate ... */

/* Step 1: launch interior computation on compute_stream */
interior_kernel<<<grid, block, 0, compute_stream>>>(d_interior, N);

/* Step 2: launch halo computation on comm_stream (runs concurrently) */
pack_halo_kernel<<<halo_grid, halo_block, 0, comm_stream>>>(d_halo_send, d_data);
cudaStreamSynchronize(comm_stream);   /* ensure halo is packed */

/* Step 3: MPI exchange using device pointers */
MPI_Request reqs[2];
MPI_Isend(d_halo_send, HALO_SIZE, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &reqs[0]);
MPI_Irecv(d_halo_recv, HALO_SIZE, MPI_DOUBLE, left,  0, MPI_COMM_WORLD, &reqs[1]);

/* Step 4: wait for interior to finish and halo exchange to complete */
cudaStreamSynchronize(compute_stream);
MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

/* Step 5: apply received halos */
apply_halo_kernel<<<halo_grid, halo_block>>>(d_data, d_halo_recv);
```

---

## 30.3 One-GPU-Per-Rank vs. Multi-GPU-Per-Rank

### One GPU per MPI Rank (Most Common)

```c
/* Assign GPU based on local rank (within node) */
MPI_Comm node_comm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                    MPI_INFO_NULL, &node_comm);
int local_rank;
MPI_Comm_rank(node_comm, &local_rank);
cudaSetDevice(local_rank);
MPI_Comm_free(&node_comm);
```

### Multiple GPUs per Rank

```cuda
/* One rank manages N GPUs; use CUDA multi-GPU patterns */
int rank_ngpus;
cudaGetDeviceCount(&rank_ngpus);

/* Launch kernels on each GPU in round-robin */
for (int g = 0; g < rank_ngpus; g++) {
    cudaSetDevice(g);
    kernel<<<grid, block, 0, streams[g]>>>(d_bufs[g], N);
}

/* Reduce across GPUs within rank using peer access */
cudaSetDevice(0);
for (int g = 1; g < rank_ngpus; g++) {
    cudaMemcpyPeerAsync(d_bufs[0], 0, d_bufs[g], g, N*sizeof(double), streams[0]);
}
cudaStreamSynchronize(streams[0]);

/* Now MPI reduce across ranks using d_bufs[0] on GPU 0 */
MPI_Allreduce(MPI_IN_PLACE, d_bufs[0], N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
```

---

## 30.4 NCCL: NVIDIA Collective Communications Library

NCCL provides GPU-native collective operations optimized for NVIDIA interconnects
(NVLink, NVSwitch, InfiniBand with GPUDirect). NCCL is the backbone of distributed
deep learning frameworks (PyTorch `torch.distributed`, Horovod).

### When to Use NCCL vs. MPI Collectives

| | MPI Allreduce | NCCL Allreduce |
|---|---|---|
| Data location | Host or device | Device only |
| Topology awareness | Network-level | NVLink + network |
| Latency (small) | Lower (CPU-driven) | Higher (GPU-driven) |
| Bandwidth (large) | Good | Better on NVLink systems |
| Integration effort | Zero (already using MPI) | Additional library and init |
| Best for | General HPC | Deep learning, GPU-dense nodes |

### NCCL Initialization

```cuda
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

/* Exchange NCCL unique IDs via MPI, then initialize communicators */
ncclComm_t init_nccl(MPI_Comm mpi_comm)
{
    int rank, size;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    /* Rank 0 generates the unique ID and broadcasts it */
    ncclUniqueId nccl_id;
    if (rank == 0)
        ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, mpi_comm);

    /* Each rank creates its NCCL communicator */
    ncclComm_t nccl_comm;
    ncclCommInitRank(&nccl_comm, size, nccl_id, rank);
    return nccl_comm;
}
```

### NCCL Allreduce

```cuda
void nccl_allreduce_example(ncclComm_t nccl_comm, cudaStream_t stream)
{
    const int N = 1 << 22;   /* 4M doubles */
    double *d_buf;
    cudaMalloc(&d_buf, N * sizeof(double));
    fill_kernel<<<(N+255)/256, 256, 0, stream>>>(d_buf, N);

    /* In-place Allreduce across all GPUs */
    ncclAllReduce(d_buf, d_buf, N, ncclDouble, ncclSum, nccl_comm, stream);

    /* NCCL is async — synchronize when needed */
    cudaStreamSynchronize(stream);
    /* d_buf now holds the global sum on all ranks */

    cudaFree(d_buf);
}
```

### Full NCCL Collective API

| Function | MPI Equivalent |
|---|---|
| `ncclAllReduce` | `MPI_Allreduce` |
| `ncclBroadcast` | `MPI_Bcast` |
| `ncclReduce` | `MPI_Reduce` |
| `ncclAllGather` | `MPI_Allgather` |
| `ncclReduceScatter` | `MPI_Reduce_scatter_block` |
| `ncclSend` | `MPI_Send` (via NCCL group) |
| `ncclRecv` | `MPI_Recv` (via NCCL group) |

NCCL point-to-point (Send/Recv) must be called within a `ncclGroupStart`/`ncclGroupEnd` block:

```cuda
ncclGroupStart();
if (rank > 0)
    ncclSend(d_send, N, ncclDouble, rank-1, nccl_comm, stream);
if (rank < size-1)
    ncclRecv(d_recv, N, ncclDouble, rank+1, nccl_comm, stream);
ncclGroupEnd();
cudaStreamSynchronize(stream);
```

---

## 30.5 Mixing MPI and NCCL

The standard pattern for distributed deep learning: use NCCL for gradient reduction
(bandwidth-intensive, GPU data) and MPI for control flow (parameter synchronization,
checkpoint coordination):

```cuda
void training_step(ncclComm_t nccl_comm, MPI_Comm mpi_comm,
                   cudaStream_t stream, int rank)
{
    /* Forward pass + backward pass on GPU */
    forward_backward_kernel<<<...>>>(d_params, d_grads, d_data);

    /* Gradient Allreduce via NCCL (stays on GPU) */
    ncclAllReduce(d_grads, d_grads, GRAD_SIZE, ncclFloat,
                  ncclSum, nccl_comm, stream);
    cudaStreamSynchronize(stream);

    /* Scale gradients (still on GPU) */
    scale_kernel<<<...>>>(d_grads, 1.0f / size, GRAD_SIZE);

    /* Optimizer step on GPU */
    optimizer_step_kernel<<<...>>>(d_params, d_grads, lr);

    /* Periodic validation loss reduction via MPI (scalar, on host) */
    if (step % VALIDATE_EVERY == 0) {
        float host_loss;
        cudaMemcpy(&host_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        MPI_Allreduce(MPI_IN_PLACE, &host_loss, 1, MPI_FLOAT, MPI_SUM, mpi_comm);
        host_loss /= size;
        if (rank == 0) printf("Step %d loss: %.4f\n", step, host_loss);
    }
}
```

### Synchronization Between MPI and NCCL

MPI and NCCL do not share synchronization state. After a NCCL operation, call
`cudaStreamSynchronize` before any MPI operation that depends on the result:

```cuda
/* WRONG: MPI may run before NCCL completes */
ncclAllReduce(d_buf, d_buf, N, ncclFloat, ncclSum, nccl_comm, stream);
float host_val;
cudaMemcpy(&host_val, d_buf, sizeof(float), cudaMemcpyDeviceToHost); /* BUG */
MPI_Bcast(&host_val, 1, MPI_FLOAT, 0, mpi_comm);

/* CORRECT */
ncclAllReduce(d_buf, d_buf, N, ncclFloat, ncclSum, nccl_comm, stream);
cudaStreamSynchronize(stream);   /* wait for NCCL to finish */
float host_val;
cudaMemcpy(&host_val, d_buf, sizeof(float), cudaMemcpyDeviceToHost);
MPI_Bcast(&host_val, 1, MPI_FLOAT, 0, mpi_comm);
```

---

## 30.6 Error Handling

```c
/* NCCL error check macro */
#define NCCL_CHECK(call)                                            \
    do {                                                             \
        ncclResult_t _r = (call);                                   \
        if (_r != ncclSuccess) {                                    \
            fprintf(stderr, "NCCL error %s:%d: %s\n",              \
                    __FILE__, __LINE__, ncclGetErrorString(_r));    \
            MPI_Abort(MPI_COMM_WORLD, 1);                          \
        }                                                            \
    } while (0)

/* CUDA error check macro */
#define CUDA_CHECK(call)                                            \
    do {                                                             \
        cudaError_t _e = (call);                                    \
        if (_e != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(_e));    \
            MPI_Abort(MPI_COMM_WORLD, 1);                          \
        }                                                            \
    } while (0)
```

---

## 30.7 Build System Integration

### CMake

```cmake
cmake_minimum_required(VERSION 3.18)
project(MpiCudaNccl CUDA CXX)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# NCCL is not shipped with a CMake config; find manually
find_library(NCCL_LIB nccl HINTS /usr/local/cuda/lib64 ENV NCCL_HOME)
find_path(NCCL_INCLUDE nccl.h HINTS /usr/local/cuda/include ENV NCCL_HOME)

add_executable(myprogram main.cu)
target_include_directories(myprogram PRIVATE ${NCCL_INCLUDE})
target_link_libraries(myprogram
    MPI::MPI_CXX
    CUDA::cudart
    ${NCCL_LIB}
)
set_target_properties(myprogram PROPERTIES CUDA_ARCHITECTURES "80;90")
```

### SLURM Launch with GPU Binding

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8         # 8 MPI ranks per node
#SBATCH --gpus-per-task=1           # 1 GPU per rank
#SBATCH --cpus-per-task=4           # 4 CPU threads per rank

module load cuda/12.3 openmpi/4.1.5-cuda nccl/2.19

export NCCL_DEBUG=INFO              # NCCL diagnostic logging
export NCCL_IB_DISABLE=0           # enable InfiniBand for inter-node
export OMPI_MCA_opal_cuda_support=1

srun --gpu-bind=per_task ./myprogram
```

---

## Summary

| Topic | Key Points |
|---|---|
| GPU-aware MPI | Pass device pointers to MPI; set `OMPI_MCA_opal_cuda_support=1` |
| GPU assignment | `cudaSetDevice(local_rank)` using `MPI_COMM_TYPE_SHARED` local rank |
| NCCL init | Rank 0 gets `ncclUniqueId`; broadcast via MPI; all call `ncclCommInitRank` |
| NCCL vs MPI collectives | NCCL better for large GPU-resident reductions; MPI for control flow |
| Stream synchronization | Always `cudaStreamSynchronize` between NCCL and MPI operations |
| NCCL P2P | Must use `ncclGroupStart` / `ncclGroupEnd` |
| Error handling | Check both `ncclResult_t` and `cudaError_t`; abort on failure |
