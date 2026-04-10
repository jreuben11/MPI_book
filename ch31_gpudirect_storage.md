# Chapter 31: GPUDirect Storage

## 31.1 What GPUDirect Storage Is

Traditional I/O requires data to pass through CPU memory even when the application
uses GPU buffers:

```
Without GDS:
  Storage → DMA → CPU DRAM → cudaMemcpy → GPU VRAM → kernel

With GDS (GPUDirect Storage):
  Storage → DMA → GPU VRAM → kernel
```

**GPUDirect Storage (GDS)** allows the storage subsystem (NVMe SSD, Lustre, GPFS)
to DMA data directly into GPU memory, bypassing CPU DRAM entirely. NVIDIA introduced
GDS via the **cuFile API** (CUDA 11.4+), and parallel filesystems have added GDS
support on recent versions.

Benefits:
- **Zero CPU-buffer copies**: eliminates one full copy for every I/O operation.
- **Higher aggregate bandwidth**: CPU DRAM is no longer a bottleneck.
- **Lower CPU overhead**: the CPU is not in the data path; it only issues DMA commands.

Hardware requirements:
- NVIDIA GPU (Volta or newer, with BAR1 support for direct PCIe DMA)
- GDS-compatible storage: NVMe via PCIe P2P, or Lustre/GPFS with GDS plugin
- NVIDIA driver ≥ 450.80.02 and CUDA ≥ 11.4

---

## 31.2 cuFile API Basics

The cuFile API is NVIDIA's interface to GDS. It mirrors POSIX file I/O but operates
on CUDA device memory buffers.

### Initialization

```c
#include <cufile.h>
#include <cuda_runtime.h>

/* Initialize the cuFile driver (once per process) */
CUfileError_t status = cuFileDriverOpen();
if (status.err != CU_FILE_SUCCESS) {
    fprintf(stderr, "cuFileDriverOpen failed: %s\n",
            cuFileGetErrorString(status));
    exit(1);
}

/* At program exit */
cuFileDriverClose();
```

### Opening a File and Registering It

```c
CUfileHandle_t cf_handle;
CUfileDescr_t  cf_descr;
int fd;

/* Open with O_DIRECT — required for GDS */
fd = open("data.bin", O_RDWR | O_CREAT | O_DIRECT, 0644);
if (fd < 0) { perror("open"); exit(1); }

memset(&cf_descr, 0, sizeof(cf_descr));
cf_descr.handle.fd = fd;
cf_descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

status = cuFileHandleRegister(&cf_handle, &cf_descr);
if (status.err != CU_FILE_SUCCESS) {
    fprintf(stderr, "cuFileHandleRegister: %s\n",
            cuFileGetErrorString(status));
}
```

### Registering GPU Memory

GPU buffers must be registered with cuFile before use in I/O:

```c
double *d_buf;
size_t buf_size = N * sizeof(double);
cudaMalloc(&d_buf, buf_size);

/* Register buffer with cuFile driver */
status = cuFileBufRegister(d_buf, buf_size, 0);
if (status.err != CU_FILE_SUCCESS) {
    fprintf(stderr, "cuFileBufRegister: %s\n",
            cuFileGetErrorString(status));
}
```

### Reading and Writing

```c
/* Write from GPU memory directly to storage */
ssize_t bytes_written = cuFileWrite(
    cf_handle,       /* file handle */
    d_buf,           /* GPU device pointer */
    buf_size,        /* number of bytes */
    file_offset,     /* file offset in bytes */
    0                /* buffer offset */
);

if (bytes_written < 0)
    fprintf(stderr, "cuFileWrite failed: %zd\n", bytes_written);

/* Read from storage directly into GPU memory */
ssize_t bytes_read = cuFileRead(
    cf_handle,
    d_buf,
    buf_size,
    file_offset,
    0
);
```

### Cleanup

```c
cuFileBufDeregister(d_buf);
cuFileHandleDeregister(cf_handle);
close(fd);
cudaFree(d_buf);
```

---

## 31.3 MPI-IO + GPUDirect Storage

Combining MPI-IO (Chapter 19–20) with GDS requires a two-layer approach: MPI-IO
manages the parallel file layout (which rank writes where), while cuFile performs
the actual I/O from GPU buffers.

### Pattern: MPI Coordinates, cuFile Transfers

```c
#include <mpi.h>
#include <cufile.h>
#include <cuda_runtime.h>
#include <fcntl.h>

void parallel_gpu_write(MPI_Comm comm, const char *filename,
                         double *d_buf, int local_n, int global_n)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Open file in parallel — POSIX open with O_DIRECT */
    int fd = open(filename, O_WRONLY | O_CREAT | O_DIRECT, 0644);

    /* Register with cuFile */
    CUfileHandle_t cf_handle;
    CUfileDescr_t  cf_descr = {};
    cf_descr.handle.fd = fd;
    cf_descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    cuFileHandleRegister(&cf_handle, &cf_descr);
    cuFileBufRegister(d_buf, local_n * sizeof(double), 0);

    /* Compute file offset for this rank */
    MPI_Offset offset = (MPI_Offset)rank * local_n * sizeof(double);

    /* Write from GPU memory directly to file */
    ssize_t written = cuFileWrite(cf_handle, d_buf,
                                   local_n * sizeof(double),
                                   offset, 0);

    /* Barrier to ensure all ranks have written before any reads */
    MPI_Barrier(comm);

    cuFileBufDeregister(d_buf);
    cuFileHandleDeregister(cf_handle);
    close(fd);
}
```

### Pattern: cuFile Async with MPI Coordination

GDS supports asynchronous I/O via the cuFile Stream API (CUDA 12.x).
The stream-based async functions take pointer arguments so the runtime can
write the result back asynchronously:

```cuda
#include <cufile.h>

/*
 * cuFileWriteAsync signature (CUDA 12.x cuFile Stream API):
 *   CUfileError_t cuFileWriteAsync(CUFileHandle_t fh,
 *                                   void          *devPtr_base,
 *                                   size_t        *size_p,
 *                                   off_t         *file_offset_p,
 *                                   off_t         *devPtr_offset_p,
 *                                   ssize_t       *bytes_written_p,
 *                                   CUstream       stream)
 *
 * All pointer arguments must remain valid until the stream is synchronized.
 */
void async_gpu_write(CUfileHandle_t cf_handle, double *d_buf,
                     size_t nbytes, off_t file_offset, cudaStream_t stream)
{
    size_t  io_size      = nbytes;
    off_t   io_offset    = file_offset;
    off_t   dev_offset   = 0;       /* write from start of d_buf */
    ssize_t bytes_written = -1;

    /* Submit async write — returns immediately */
    CUfileError_t err = cuFileWriteAsync(cf_handle, d_buf,
                                          &io_size, &io_offset,
                                          &dev_offset, &bytes_written,
                                          stream);
    if (err.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "cuFileWriteAsync failed to submit\n");
        return;
    }

    /* bytes_written is valid only after the stream completes */
    cudaStreamSynchronize(stream);

    if (bytes_written != (ssize_t)nbytes)
        fprintf(stderr, "Short write: expected %zu, got %zd\n",
                nbytes, bytes_written);
}
```

---

## 31.4 Integrating with MPI-IO Views

For multi-dimensional data, use MPI-IO file views to define the layout, then
cuFile for the actual transfer. The key is computing the correct byte offset:

```c
/* 2D decomposition: each rank owns a local_rows × global_cols slice */
/* MPI-IO computes the correct file offset; cuFile does the transfer   */

MPI_File mpi_fh;
MPI_File_open(comm, filename,
              MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpi_fh);

/* Set view to define this rank's portion of the 2D array */
int global_sizes[2] = {global_rows, global_cols};
int local_sizes[2]  = {local_rows, global_cols};
int starts[2]       = {rank * local_rows, 0};

MPI_Datatype filetype;
MPI_Type_create_subarray(2, global_sizes, local_sizes, starts,
                          MPI_ORDER_C, MPI_DOUBLE, &filetype);
MPI_Type_commit(&filetype);
MPI_File_set_view(mpi_fh, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

/* Query what byte offset this rank's data starts at */
MPI_Offset view_disp;
MPI_File_get_position(mpi_fh, &view_disp);
MPI_Offset byte_offset = view_disp + (MPI_Offset)rank * local_rows
                         * global_cols * sizeof(double);

MPI_Type_free(&filetype);
MPI_File_close(&mpi_fh);

/* Now use cuFile with the computed offset */
int fd = open(filename, O_WRONLY | O_DIRECT);
CUfileHandle_t cf;
/* ... register fd and d_buf ... */
cuFileWrite(cf, d_local_data, local_rows * global_cols * sizeof(double),
            byte_offset, 0);
```

For simple contiguous decompositions, the offset is just
`rank * local_count * sizeof(element)` — no MPI-IO view needed.

---

## 31.5 Checkpoint/Restart with GDS

A GPU simulation checkpoint pattern that avoids all CPU memory:

```cuda
void write_gpu_checkpoint(const char *ckpt_path, int rank,
                           double *d_state, size_t state_bytes,
                           int step, MPI_Comm comm)
{
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/rank_%04d_step_%06d.bin",
             ckpt_path, rank, step);

    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);

    CUfileHandle_t cf;
    CUfileDescr_t  descr = {};
    descr.handle.fd = fd;
    descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    cuFileHandleRegister(&cf, &descr);
    cuFileBufRegister(d_state, state_bytes, 0);

    /* Write GPU state directly to NVMe — no host buffer */
    ssize_t written = cuFileWrite(cf, d_state, state_bytes, 0, 0);
    if ((size_t)written != state_bytes)
        fprintf(stderr, "Rank %d: short checkpoint write\n", rank);

    cuFileBufDeregister(d_state);
    cuFileHandleDeregister(cf);
    fsync(fd);   /* ensure durability */
    close(fd);

    /* All ranks confirm checkpoint completion */
    MPI_Barrier(comm);
    if (rank == 0)
        printf("Checkpoint step %d complete\n", step);
}

void read_gpu_checkpoint(const char *ckpt_path, int rank,
                          double *d_state, size_t state_bytes, int step)
{
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/rank_%04d_step_%06d.bin",
             ckpt_path, rank, step);

    int fd = open(filename, O_RDONLY | O_DIRECT);

    CUfileHandle_t cf;
    CUfileDescr_t  descr = {};
    descr.handle.fd = fd;
    descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    cuFileHandleRegister(&cf, &descr);
    cuFileBufRegister(d_state, state_bytes, 0);

    ssize_t bytes_read = cuFileRead(cf, d_state, state_bytes, 0, 0);
    if ((size_t)bytes_read != state_bytes)
        fprintf(stderr, "Rank %d: short checkpoint read\n", rank);

    cuFileBufDeregister(d_state);
    cuFileHandleDeregister(cf);
    close(fd);
}
```

---

## 31.6 Combining GDS with Partitioned Communication

GDS and partitioned communication (Chapter 23) naturally complement each other
in streaming pipelines:

```
[Storage] → cuFile read (partition 0) → MPI_Pready(0) → network
           → cuFile read (partition 1) → MPI_Pready(1) → network
           → cuFile read (partition 2) → MPI_Pready(2) → network
```

```cuda
/* Producer rank: reads GPU partitions from storage, signals readiness */

double *d_buf;  /* pinned symmetric GPU allocation */
const int NPART = 4;
const size_t PART_BYTES = TOTAL_BYTES / NPART;

MPI_Request send_req;
MPI_Psend_init(d_buf, NPART, PART_BYTES, MPI_BYTE,
               consumer_rank, 0, comm, MPI_INFO_NULL, &send_req);

CUfileHandle_t cf;
/* ... open and register file ... */
cuFileBufRegister(d_buf, TOTAL_BYTES, 0);

MPI_Start(&send_req);

for (int p = 0; p < NPART; p++) {
    /* Read partition p from storage directly into GPU buffer */
    cuFileRead(cf, (char*)d_buf + p * PART_BYTES, PART_BYTES,
               p * PART_BYTES, 0);
    /* Signal: partition p is ready to send */
    MPI_Pready(p, &send_req);
}

MPI_Wait(&send_req, MPI_STATUS_IGNORE);
cuFileBufDeregister(d_buf);
MPI_Request_free(&send_req);
```

---

## 31.7 Performance Considerations

### O_DIRECT Requirement

cuFile requires files opened with `O_DIRECT`. This bypasses the Linux page cache
and allows DMA directly to GPU memory. Implications:

- **Buffer alignment**: I/O buffers must be aligned to 4096 bytes.
  Use `cudaMalloc` (always aligned) or `posix_memalign` for host buffers.
- **Transfer size alignment**: transfers should be multiples of the sector size
  (typically 512 or 4096 bytes).
- **No page cache**: repeated reads of the same data are not cached by the OS.
  Use application-level caching in GPU memory.

```cuda
/* Correct: cudaMalloc is always suitably aligned */
double *d_buf;
cudaMalloc(&d_buf, ALIGNED_SIZE);   /* ALIGNED_SIZE must be multiple of 4096 */

/* Check alignment explicitly */
assert(((uintptr_t)d_buf % 4096) == 0);
```

### Measuring GDS Benefit

GDS is most beneficial when:
1. I/O volume > available CPU DRAM bandwidth × transfer time
2. GPU-to-storage path (PCIe P2P) bandwidth > CPU DRAM bandwidth
3. The CPU is already busy with MPI coordination or compute

Benchmark GDS vs. standard I/O:

```c
/* Standard I/O baseline: host buffer intermediary */
double t0 = MPI_Wtime();
for (int i = 0; i < NITERS; i++) {
    read(fd, h_buf, nbytes);                         /* storage → host */
    cudaMemcpy(d_buf, h_buf, nbytes, cudaMemcpyHtoD); /* host → GPU */
}
double t_standard = (MPI_Wtime() - t0) / NITERS;

/* GDS path: storage → GPU directly */
t0 = MPI_Wtime();
for (int i = 0; i < NITERS; i++)
    cuFileRead(cf, d_buf, nbytes, 0, 0);
double t_gds = (MPI_Wtime() - t0) / NITERS;

if (rank == 0)
    printf("Standard: %.3f s  GDS: %.3f s  Speedup: %.2fx\n",
           t_standard, t_gds, t_standard / t_gds);
```

### Filesystem Support Matrix (2025)

| Filesystem | GDS Support | Notes |
|---|---|---|
| Local NVMe (PCIe) | Full | Native GPUDirect P2P |
| NVMe-oF (RDMA) | Full | Remote NVMe over InfiniBand |
| Lustre | 2.15+ with GDS plugin | Site-dependent; requires compatible OSTs |
| GPFS / SpectrumScale | 5.1.3+ | IBM GDS integration |
| BeeGFS | 7.3+ | Production (community edition) |
| PVFS2 / OrangeFS | No | Not supported |
| NFS | No | No RDMA P2P path |

---

## 31.8 Error Handling

```c
/* cuFile error codes are in CUfileError_t */
CUfileError_t err;

err = cuFileHandleRegister(&cf, &descr);
if (err.err != CU_FILE_SUCCESS) {
    /* err.err is the cuFile error code */
    /* err.cu_err is the underlying CUDA error (if applicable) */
    fprintf(stderr, "cuFile error: %s (CUDA: %s)\n",
            cuFileGetErrorString(err),
            err.cu_err != CUDA_SUCCESS ?
                cudaGetErrorString((cudaError_t)err.cu_err) : "none");
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/* cuFileRead/Write return negative values on error */
ssize_t n = cuFileRead(cf, d_buf, nbytes, 0, 0);
if (n < 0) {
    fprintf(stderr, "cuFileRead failed: %zd\n", n);
    /* n encodes the cuFile error code as a negative value */
    MPI_Abort(MPI_COMM_WORLD, 1);
}
```

---

## 31.9 Build and Runtime Setup

### CMake

```cmake
find_package(CUDAToolkit REQUIRED)
find_library(CUFILE_LIB cufile
             HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
find_path(CUFILE_INCLUDE cufile.h
          HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/local/cuda/include)

target_include_directories(myprogram PRIVATE ${CUFILE_INCLUDE})
target_link_libraries(myprogram CUDA::cudart MPI::MPI_C ${CUFILE_LIB})
```

### Runtime Configuration

```bash
# Check GDS driver status
nvidia-smi gds --status

# Enable GDS compatibility mode (for systems without native NVMe P2P)
export CUFILE_ENV_PATH_JSON=/etc/cufile.json

# cufile.json: tune buffer sizes and compatibility options
cat /etc/cufile.json
# {
#   "execution": { "max_io_queue_depth": 128, "max_batch_io_size": 128 },
#   "properties": { "use_compat_mode": false, "gds_rdma_write_support": true }
# }
```

---

## Summary

| Topic | Key Points |
|---|---|
| GDS benefit | Eliminates host-buffer copy for GPU I/O; requires NVMe/network with P2P |
| `cuFileDriverOpen` | Initialize once per process; `cuFileDriverClose` at exit |
| `O_DIRECT` | Required; bypasses page cache; buffers must be 4096-byte aligned |
| `cuFileBufRegister` | Register GPU buffer before I/O; deregister after |
| `cuFileRead/Write` | Synchronous; negative return = error |
| MPI + GDS | MPI manages layout/coordination; cuFile does actual transfer |
| Partitioned comm | cuFile reads trigger `MPI_Pready`; natural pipeline |
| Checkpoint | Per-rank files with cuFile; `MPI_Barrier` for collective sync |
| Filesystem | Lustre 2.15+, GPFS 5.1.3+, local NVMe; no NFS |
