# Chapter 35: Application-Level Checkpointing — SCR and VeloC

## 35.1 Why Application-Level Checkpointing?

Large HPC jobs run for hours or days. Node failures, scheduler preemption, and
time-limit hits are inevitable. Two strategies exist:

```
System-level (DMTCP, CRIU):
  - OS snapshots all process memory + file descriptors
  - Zero application changes
  - Large checkpoint files; MPI socket state hard to capture
  - Limited support on modern HPC clusters

Application-level (SCR, VeloC):
  - Application explicitly saves its logical state
  - Smaller, portable checkpoints
  - Multi-level: node-local SSD → burst buffer → parallel filesystem
  - Works reliably with MPI; the standard choice for large jobs
```

---

## 35.2 Multi-Level Checkpointing Architecture

```
Level 0: Compute Node NVME / RAM (partner copy)
    │  Fast write (~10 GB/s), volatile, lost on node failure
    │
Level 1: Burst Buffer (DDN IME, Cray DataWarp, Vast)
    │  Medium write (~100 GB/s aggregate), survives node failure
    │
Level 2: Parallel Filesystem (Lustre, GPFS, DAOS)
       Slow write (~10 GB/s aggregate), persistent, survives system failure
```

SCR (Scalable Checkpoint/Restart) and VeloC (Very Low Overhead Checkpointing) both
implement this hierarchy. SCR is older and simpler; VeloC is modular and supports
async flush pipelines.

---

## 35.3 SCR (Scalable Checkpoint/Restart)

SCR is developed at LLNL. It intercepts checkpoint writes and manages multi-level
redundancy transparently.

### Installing SCR

```bash
# From source
git clone https://github.com/LLNL/scr.git
cd scr
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/scr \
         -DWITH_MPI_PREFIX=$(dirname $(which mpicc))/..
make -j4 install
```

### SCR API

```c
#include <scr.h>

/* Initialize SCR (call after MPI_Init) */
SCR_Init();

/* Check if a restart is available from a previous run */
int have_restart;
char restart_name[SCR_MAX_FILENAME];
SCR_Have_restart(&have_restart, restart_name);

/* Start a restart — reads from best available checkpoint level */
SCR_Start_restart(restart_name);
/* ... read checkpoint files ... */
SCR_Complete_restart(1 /* success */);

/* ---- Main time loop ---- */

while (!done) {
    /* Do computation ... */

    /* Ask SCR if we should checkpoint now
       (SCR considers time, redundancy needs, job time remaining) */
    int need_checkpoint;
    SCR_Need_checkpoint(&need_checkpoint);

    if (need_checkpoint) {
        SCR_Start_checkpoint();

        /* Each rank writes its own checkpoint file */
        char filename[SCR_MAX_FILENAME];
        SCR_Route_file("checkpoint.dat", filename);
        /* filename is a local path SCR has routed to the appropriate level */

        FILE *fp = fopen(filename, "w");
        /* ... write rank's state to fp ... */
        fclose(fp);

        SCR_Complete_checkpoint(1 /* success */);
    }
}

/* Finalize SCR (call before MPI_Finalize) */
SCR_Finalize();
```

### Full SCR Example: Iterative Solver

```c
#include <mpi.h>
#include <scr.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1000000   /* local array size per rank */
#define MAX_ITER 10000

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    SCR_Init();

    double *x = malloc(N * sizeof(double));
    int start_iter = 0;

    /* Check for restart */
    int have_restart;
    char restart_name[SCR_MAX_FILENAME];
    SCR_Have_restart(&have_restart, restart_name);

    if (have_restart) {
        SCR_Start_restart(restart_name);

        /* Route the checkpoint file for this rank */
        char ckpt_file[256];
        snprintf(ckpt_file, sizeof(ckpt_file), "rank_%d.ckpt", rank);
        char routed[SCR_MAX_FILENAME];
        SCR_Route_file(ckpt_file, routed);

        FILE *fp = fopen(routed, "r");
        int read_ok = 0;
        if (fp) {
            read_ok  = (fread(&start_iter, sizeof(int),    1, fp) == 1);
            read_ok &= (fread(x,           sizeof(double), N, fp) == (size_t)N);
            fclose(fp);
            if (rank == 0) printf("Restarting from iteration %d\n", start_iter);
        }
        /* Use read_ok (not fp != NULL after fclose) — fp is non-NULL but dangling */
        SCR_Complete_restart(read_ok);
    } else {
        /* Fresh start */
        for (int i = 0; i < N; i++) x[i] = rank + 1.0;
    }

    /* Main iteration */
    for (int iter = start_iter; iter < MAX_ITER; iter++) {
        /* Computation */
        for (int i = 0; i < N; i++) x[i] *= 0.999;

        /* Check if SCR wants a checkpoint */
        int need_ckpt;
        SCR_Need_checkpoint(&need_ckpt);

        if (need_ckpt) {
            SCR_Start_checkpoint();

            char ckpt_file[256];
            snprintf(ckpt_file, sizeof(ckpt_file), "rank_%d.ckpt", rank);
            char routed[SCR_MAX_FILENAME];
            SCR_Route_file(ckpt_file, routed);

            FILE *fp = fopen(routed, "w");
            int success = 0;
            if (fp) {
                int next_iter = iter + 1;
                fwrite(&next_iter, sizeof(int), 1, fp);
                fwrite(x, sizeof(double), N, fp);
                fclose(fp);
                success = 1;
            }

            SCR_Complete_checkpoint(success);

            if (rank == 0) printf("Checkpoint at iteration %d\n", iter);
        }
    }

    free(x);
    SCR_Finalize();
    MPI_Finalize();
    return 0;
}
```

### Building with SCR

```bash
mpicc -O2 -o solver solver.c \
    -I$HOME/scr/include \
    -L$HOME/scr/lib -lscr -Wl,-rpath,$HOME/scr/lib
```

### SCR Configuration

```bash
# SCR configuration file (SCR_CONF_FILE or ~/.scrconf)
cat > ~/.scrconf << 'EOF'
SCR_PREFIX=/scratch/myproject/checkpoints
SCR_CACHE_SIZE=2             # keep 2 checkpoints in cache
SCR_CHECKPOINT_INTERVAL=300  # checkpoint every 300 seconds
SCR_DISTRIBUTE=1             # spread checkpoint to partner node

# Redundancy scheme: XOR (tolerates 1 failure), PARTNER (mirror copy)
SCR_COPY_TYPE=XOR
SCR_SET_SIZE=8               # XOR set size
EOF

export SCR_CONF_FILE=~/.scrconf
```

### SLURM Integration

```bash
#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00

# SCR uses SLURM env vars automatically
export SCR_PREFIX=$SCRATCH/checkpoints/$SLURM_JOB_ID

# Wrap with scr_run for automatic restart on failure
scr_run srun ./solver

# scr_run re-submits the job if there's a checkpoint available
# (requires scr_srun wrapper scripts from SCR installation)
```

---

## 35.4 VeloC (Very Low Overhead Checkpointing)

VeloC is designed for async checkpoint pipelines and modular backend storage.
It separates the checkpoint API from the storage backend (POSIX, HDF5, DAOS).

### Installing VeloC

```bash
git clone https://github.com/ECP-VeloC/VELOC.git
cd VELOC
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/veloc \
         -DWITH_AXL=ON \    # async transfer library
         -DWITH_ER=ON        # encoding/redundancy
make -j4 install
```

### VeloC API

```c
#include <veloc.h>

/* Initialize VeloC */
VELOC_Init(MPI_COMM_WORLD, "veloc.cfg");

/* Register memory regions to checkpoint */
int   iter = 0;
double *x   = malloc(N * sizeof(double));

VELOC_Mem_protect(0, &iter, 1,    sizeof(int));    /* id=0: iteration counter */
VELOC_Mem_protect(1, x,    N,    sizeof(double));  /* id=1: array x */

/* Check for restart */
int latest_version;
if (VELOC_Restart_test("solver", &latest_version) == VELOC_SUCCESS) {
    VELOC_Restart("solver", latest_version);
    /* iter and x now restored */
}

/* Main loop */
for (; iter < MAX_ITER; iter++) {
    /* ... computation ... */

    if (iter % CHECKPOINT_FREQ == 0) {
        VELOC_Checkpoint("solver", iter);
        /* Returns immediately; VeloC flushes asynchronously */
    }
}

/* Wait for any pending async flush */
VELOC_Checkpoint_wait();

VELOC_Finalize(0);
```

### VeloC Configuration File

```ini
# veloc.cfg
scratch = /dev/shm/veloc          # node-local fast storage
persistent = /scratch/checkpoints # parallel filesystem

mode = async                       # async (recommended) or sync
max_versions = 3                   # keep last 3 checkpoints

# Backend: posix (default), hdf5, DAOS
backend = posix
```

### Full VeloC Example

```c
#include <mpi.h>
#include <veloc.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define MAX_ITER 10000
#define CKPT_FREQ 100

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VELOC_Init(MPI_COMM_WORLD, "veloc.cfg");

    int iter = 0;
    double *x = malloc(N * sizeof(double));

    /* Register checkpoint regions */
    VELOC_Mem_protect(0, &iter, 1, sizeof(int));
    VELOC_Mem_protect(1, x,    N, sizeof(double));

    /* Restart if possible */
    int version;
    if (VELOC_Restart_test("myapp", &version) == VELOC_SUCCESS) {
        if (VELOC_Restart("myapp", version) == VELOC_SUCCESS) {
            if (rank == 0)
                printf("Restarted from checkpoint version %d (iter=%d)\n",
                       version, iter);
        }
    } else {
        /* Initialize fresh */
        for (int i = 0; i < N; i++) x[i] = (double)(rank + 1);
    }

    for (; iter < MAX_ITER; iter++) {
        /* Computation */
        for (int i = 0; i < N; i++) x[i] += 0.001;

        if (iter % CKPT_FREQ == 0) {
            int rc = VELOC_Checkpoint("myapp", iter);
            if (rc != VELOC_SUCCESS && rank == 0)
                fprintf(stderr, "Checkpoint failed at iter %d\n", iter);
        }
    }

    VELOC_Checkpoint_wait();   /* drain async flushes */
    VELOC_Finalize(0);
    free(x);
    MPI_Finalize();
    return 0;
}
```

### Building with VeloC

```bash
mpicc -O2 -o solver solver.c \
    -I$HOME/veloc/include \
    -L$HOME/veloc/lib -lveloc -Wl,-rpath,$HOME/veloc/lib
```

---

## 35.5 Manual Checkpoint with MPI-IO

When SCR and VeloC are unavailable, write checkpoints directly using MPI-IO:

```c
#include <mpi.h>
#include <stdio.h>

void checkpoint_write(MPI_Comm comm, int rank, int size,
                      double *x, int N, int iter,
                      const char *dir)
{
    /* Each rank writes to a separate file using MPI-IO */
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/ckpt_%05d_rank_%05d.bin",
             dir, iter, rank);

    MPI_File fh;
    MPI_File_open(MPI_COMM_SELF, filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);

    MPI_File_write_at(fh, 0, &iter, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_write_at(fh, sizeof(int), x, N, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    /* Collective sync to ensure all ranks finished */
    MPI_Barrier(comm);

    /* Rank 0 writes a marker file indicating checkpoint is complete */
    if (rank == 0) {
        char marker[256];
        snprintf(marker, sizeof(marker), "%s/ckpt_%05d.done", dir, iter);
        FILE *fp = fopen(marker, "w");
        if (fp) { fprintf(fp, "iter=%d nranks=%d\n", iter, size); fclose(fp); }
    }
    MPI_Barrier(comm);
}

int checkpoint_restart(MPI_Comm comm, int rank, double *x, int N,
                       int *iter, const char *dir)
{
    /* Rank 0 finds the latest complete checkpoint */
    int latest = -1;
    if (rank == 0) {
        /* Scan for .done marker files — simplified */
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "ls %s/ckpt_*.done 2>/dev/null | tail -1", dir);
        FILE *p = popen(cmd, "r");
        if (p) {
            char line[256];
            if (fgets(line, sizeof(line), p))
                sscanf(line, "%*[^0-9]%d", &latest);
            pclose(p);
        }
    }
    MPI_Bcast(&latest, 1, MPI_INT, 0, comm);

    if (latest < 0) return 0;   /* no checkpoint found */

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/ckpt_%05d_rank_%05d.bin",
             dir, latest, rank);

    MPI_File fh;
    int rc = MPI_File_open(MPI_COMM_SELF, filename,
                           MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS) return 0;

    MPI_File_read_at(fh, 0, iter, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_read_at(fh, sizeof(int), x, N, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    return 1;
}
```

---

## 35.6 SLURM Job Arrays for Automatic Restart

```bash
#!/bin/bash
# restart_array.sh — automatically restarts if time limit hit
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=32
#SBATCH --time=02:00:00
#SBATCH --array=1-20%1         # at most 1 running at a time

CKPT_DIR=$SCRATCH/checkpoints/$SLURM_ARRAY_JOB_ID

# If a checkpoint exists, restart from it; otherwise fresh start
if [ -d "$CKPT_DIR" ]; then
    echo "Restarting from checkpoint in $CKPT_DIR"
fi

srun ./solver --ckpt-dir=$CKPT_DIR

# Exit code 0 = success (done); non-zero = did not finish
# SLURM will launch the next array element if this one times out
```

---

## 35.7 Choosing Between SCR, VeloC, and Manual

| | SCR | VeloC | Manual MPI-IO |
|---|---|---|---|
| Setup effort | Medium | Medium | Low |
| Async flush | No | Yes | No |
| Multi-level | Yes (XOR, partner) | Yes (pluggable) | Manual |
| Redundancy | Yes | Via ER module | Manual |
| Portability | LLNL clusters, others | Any MPI cluster | Any MPI cluster |
| HDF5 backend | No | Yes | No |
| Suitable for | Most HPC workloads | Large deep learning | Simple use cases |

---

## 35.8 io_uring for Local Checkpoint Writes

For Level 0 node-local checkpoints to NVMe (the fastest tier in the multi-level
hierarchy), bypassing the kernel page cache with `O_DIRECT` plus Linux's
**io_uring** interface (kernel 5.1+) delivers maximum throughput and avoids stalling
compute threads on I/O system calls.

io_uring works through two lock-free ring buffers in shared memory between kernel
and user space:
- **Submission Queue (SQ)**: application posts I/O requests without a syscall per I/O
- **Completion Queue (CQ)**: kernel posts completions; application polls or waits

This allows batching dozens of writes into a single `io_uring_submit()` syscall and
overlapping I/O with computation — ideal for streaming checkpoint data to local NVMe
while MPI continues the next iteration.

### liburing Setup

```bash
sudo apt install liburing-dev    # Ubuntu 20.04+ / kernel 5.4+
sudo dnf install liburing-devel  # Fedora/RHEL
```

### Fixed-Buffer O_DIRECT Pattern

```c
#include <liburing.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define QUEUE_DEPTH  64
#define BLOCK_SIZE   (1UL << 20)   /* 1 MiB chunks */

/* O_DIRECT requires 512-byte-aligned I/O and page-aligned buffers */
static int open_direct(const char *path)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) { perror("open"); exit(1); }
    return fd;
}

void uring_checkpoint(int fd, void *data, size_t nbytes)
{
    struct io_uring ring;
    io_uring_queue_init(QUEUE_DEPTH, &ring, 0);

    /* Register buffer once — avoids per-write memory pinning overhead */
    struct iovec iov = { .iov_base = data, .iov_len = nbytes };
    io_uring_register_buffers(&ring, &iov, 1);

    size_t offset = 0;
    int in_flight  = 0;

    while (offset < nbytes || in_flight > 0) {
        /* Fill SQ with new writes */
        while (offset < nbytes && in_flight < QUEUE_DEPTH) {
            size_t chunk = (nbytes - offset < BLOCK_SIZE) ? nbytes - offset
                                                          : BLOCK_SIZE;
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            /* WRITE_FIXED: uses pre-registered buffer — no re-pinning per call */
            io_uring_prep_write_fixed(sqe, fd, (char *)data + offset,
                                      chunk, (off_t)offset, 0 /* buf_index */);
            sqe->user_data = (uint64_t)offset;
            offset += chunk;
            in_flight++;
        }

        io_uring_submit(&ring);   /* one syscall for all queued writes */

        /* Wait for at least one completion before refilling */
        struct io_uring_cqe *cqe;
        io_uring_wait_cqe(&ring, &cqe);
        if (cqe->res < 0)
            fprintf(stderr, "write error at offset %zu: %s\n",
                    (size_t)cqe->user_data, strerror(-cqe->res));
        io_uring_cqe_seen(&ring, cqe);
        in_flight--;
    }

    io_uring_unregister_buffers(&ring);
    io_uring_queue_exit(&ring);
}
```

### Integration with MPI Checkpointing

```c
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const size_t N = 10000000;

    /* posix_memalign: 4 KiB alignment satisfies O_DIRECT requirements */
    double *x;
    posix_memalign((void **)&x, 4096, N * sizeof(double));

    for (int iter = 0; iter < MAX_ITER; iter++) {
        compute(x, N);

        if (iter % CKPT_FREQ == 0) {
            /* Each rank writes independently to node-local NVMe */
            char path[256];
            snprintf(path, sizeof(path), "/local/nvme/ckpt_%05d_rank_%05d.bin",
                     iter, rank);
            int fd = open_direct(path);

            uring_checkpoint(fd, x, N * sizeof(double));
            fsync(fd);   /* flush write-back cache — required before marking done */
            close(fd);

            /* Collective fence: all ranks done before continuing */
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    free(x);
    MPI_Finalize();
}
```

```bash
mpicc -O2 -o solver solver.c -luring
```

### When to Use io_uring vs MPI-IO

| | `write()` / `fwrite()` | MPI-IO (`File_write_at_all`) | io_uring + O_DIRECT |
|---|---|---|---|
| Target storage | Any | Parallel FS (Lustre, GPFS) | Node-local NVMe, burst buffer |
| Syscall cost | One per call | Library-managed | Batched — 1 `submit` per N writes |
| Page cache | Yes | Yes (unless `O_DIRECT` hint) | Bypassed |
| Collective optimization | No | Two-phase I/O, striping | N/A (local only) |
| CPU stall | Blocks | Blocks | Overlap with compute |
| Complexity | Low | Medium | Medium |

**Guidance**: Use MPI-IO for all writes to parallel filesystems — ROMIO's collective
buffering and Lustre striping optimizations already do the right thing there. Use
io_uring for the **Level 0** tier only: direct writes to node-local NVMe or
`/dev/shm` → burst buffer where you control the I/O path and want to pipeline
checkpoint writes alongside ongoing computation. SCR and VeloC handle this tier
internally; io_uring is relevant only when implementing your own Level 0 writer
or augmenting the manual MPI-IO pattern from Section 35.5.

---

## Summary

| Topic | Key Points |
|---|---|
| Multi-level checkpointing | Node-local (fast) → burst buffer → PFS (persistent) |
| SCR | `SCR_Need_checkpoint`, `SCR_Route_file`, `SCR_Start/Complete_checkpoint`; config via `~/.scrconf` |
| VeloC | `VELOC_Mem_protect` registers regions; `VELOC_Checkpoint` writes asynchronously; `VELOC_Restart_test` finds latest |
| Manual MPI-IO | `MPI_File_open(MPI_COMM_SELF, ...)` per rank; `MPI_Barrier` after all ranks complete; marker file from rank 0 |
| Restart | `SCR_Have_restart` / `VELOC_Restart_test` → read saved state → resume iteration |
| SLURM | Job arrays (`--array=1-N%1`) with time limits provide automatic restart mechanism |
| I/O tips | Checkpoint to `/dev/shm` or burst buffer first; SCR/VeloC flush to PFS in background |
| io_uring (Level 0) | `io_uring_queue_init` + `io_uring_register_buffers` + `O_DIRECT`; batch writes to local NVMe; overlap with compute; use only for app-managed local I/O, not parallel FS |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
