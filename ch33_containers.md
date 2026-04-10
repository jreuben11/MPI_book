# Chapter 33: MPI in Containers — Singularity/Apptainer + PMIx

## 33.1 Why Containers for MPI?

Containers solve the reproducibility and portability problem in HPC:

- **Reproducibility**: freeze the entire software stack (MPI, libraries, application) in an image
- **Portability**: move a working application across clusters without recompiling
- **Dependency isolation**: avoid conflicts between system MPI and application MPI requirements

The challenge: MPI startup requires tight integration with the cluster's job scheduler
(PMIx), network fabric (InfiniBand, Slingshot), and process binding — all of which live
*outside* the container.

### The Two MPI Models for Containers

```
Model 1: Host MPI (Hybrid)
  - Host launches MPI processes via srun/mpiexec
  - MPI libraries inside container must ABI-match host MPI
  - Container gets MPI from host via bind-mount or symlink
  - Common: OpenMPI 4.x on host → OpenMPI 4.x inside container

Model 2: PMIx-Only (Preferred Modern Approach)
  - Host provides only PMIx for process startup/coordination
  - Container ships its own complete MPI stack
  - srun launches container processes; PMIx socket passed in
  - Best for portability; requires MPI built with PMIx support
```

---

## 33.2 Apptainer (formerly Singularity)

Apptainer is the dominant container runtime in HPC. Unlike Docker:
- Runs as the invoking user (no root daemon)
- Bind-mounts host paths by default
- Works with SLURM and PMIx natively

### Installing Apptainer

```bash
# Ubuntu/Debian (from apt)
sudo apt install apptainer

# Or from source (see https://apptainer.org/docs)
```

### Basic MPI Container Definition

```singularity
# mpi_app.def — Apptainer definition file
Bootstrap: docker
From: ubuntu:22.04

%post
    apt-get update && apt-get install -y \
        build-essential \
        openmpi-bin \
        libopenmpi-dev \
        wget

    # Build your application inside the container
    cd /tmp
    wget https://example.com/myapp.tar.gz
    tar xzf myapp.tar.gz
    cd myapp && make -j4
    cp myapp /usr/local/bin/

%environment
    export PATH=/usr/local/bin:$PATH
    export OMPI_MCA_btl_vader_single_copy_mechanism=none

%runscript
    exec /usr/local/bin/myapp "$@"

%labels
    Author myname
    Version 1.0
```

Build the container:

```bash
# Build requires root or --fakeroot
apptainer build mpi_app.sif mpi_app.def
# Or with fakeroot (configured in /etc/subuid):
apptainer build --fakeroot mpi_app.sif mpi_app.def
```

---

## 33.3 PMIx Integration

PMIx (Process Management Interface — Exascale) is the modern process startup protocol.
SLURM communicates with MPI_Init via a PMIx server socket.

### How PMIx Bootstrap Works

```
slurmstepd
    │
    ├── sets PMIX_SERVER_URI env var
    │   (socket path like /tmp/pmix-12345/agent.1234.0)
    │
    └── launches container processes via srun
            │
            └── MPI_Init() inside container
                    │
                    └── connects to PMIX_SERVER_URI socket
                        (socket is accessible from container
                         because it's bind-mounted from host /tmp)
```

The key: `/tmp` is typically bind-mounted into Apptainer containers automatically.
The PMIx socket in `/tmp/pmix-*/` is thus reachable from inside the container.

### Running MPI in Apptainer via SLURM

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# PMIx launch — container's MPI_Init connects to slurmstepd's PMIx server
srun --mpi=pmix apptainer exec mpi_app.sif /usr/local/bin/myapp
```

```bash
# Verify PMIx is available on the host
srun --mpi=list
# Should show: pmix, pmix_v2, pmix_v3, etc.
```

### Passing Environment Variables into the Container

```bash
# Apptainer inherits the host environment by default
# Pass specific variables:
srun --mpi=pmix apptainer exec \
    --env OMPI_MCA_opal_cuda_support=1 \
    --env UCX_TLS=rc,cuda_copy,gdr_copy \
    mpi_gpu.sif /usr/local/bin/mpi_gpu_app
```

---

## 33.4 Bind-Mounting for MPI

### Binding the Host MPI (Hybrid Model)

When the container's MPI ABI must match the host:

```bash
# Bind-mount host OpenMPI over the container's MPI
apptainer exec \
    --bind /usr/lib/x86_64-linux-gnu/openmpi:/usr/lib/x86_64-linux-gnu/openmpi \
    --bind /usr/bin/mpiexec:/usr/bin/mpiexec \
    mpi_app.sif /usr/local/bin/myapp

# Or use --nv for NVIDIA GPU support (also binds CUDA libraries)
apptainer exec --nv mpi_gpu.sif /usr/local/bin/mpi_gpu_app
```

### Binding Network Fabrics

InfiniBand and other fabrics require the host verbs library:

```bash
# InfiniBand
apptainer exec \
    --bind /usr/lib/libibverbs.so.1 \
    --bind /dev/infiniband \
    mpi_app.sif myapp

# Or set in /etc/apptainer/apptainer.conf:
# bind path = /dev/infiniband
```

### GPU Bind-Mount (NVIDIA)

```bash
# --nv automatically binds CUDA libraries and /dev/nvidia*
apptainer exec --nv mpi_gpu.sif myapp

# AMD ROCm
apptainer exec --rocm mpi_hip_app.sif myapp

# Verify inside:
apptainer exec --nv mpi_gpu.sif nvidia-smi
apptainer exec --rocm mpi_hip_app.sif rocm-smi
```

---

## 33.5 Container-Aware MPI Program

Your application code itself needs no container-specific changes. The process view
from inside the container looks identical to a bare-metal launch:

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* This works identically in a container — MPI_Init connected via PMIx */
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("Rank %d/%d on host %s\n", rank, size, hostname);

    MPI_Finalize();
    return 0;
}
```

```bash
# Build inside container (during %post) or outside with matching compiler
mpicc -O2 -o myapp myapp.c

# Launch 64 ranks across 4 nodes, 16 per node
srun --nodes=4 --ntasks-per-node=16 --mpi=pmix \
    apptainer exec mpi_app.sif /usr/local/bin/myapp
```

---

## 33.6 Process Binding in Containers

SLURM's CPU binding (`--cpu-bind`) operates at the host level and is respected by
Apptainer. The container process sees a restricted CPU mask.

```bash
# Bind each rank to its own core set
srun --ntasks-per-node=16 --cpus-per-task=4 \
     --cpu-bind=cores \
     --mpi=pmix \
     apptainer exec mpi_app.sif myapp

# Verify inside the container:
apptainer exec mpi_app.sif taskset -c -p $$
# Shows the CPU mask set by SLURM
```

### Checking Binding from Inside MPI

```c
#include <mpi.h>
#include <sched.h>
#include <stdio.h>

void print_cpu_binding(int rank)
{
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);

    int count = CPU_COUNT(&cpuset);
    printf("Rank %d: bound to %d CPUs\n", rank, count);
}
```

---

## 33.7 Shared Memory Inside Containers

Apptainer creates a separate namespace by default. Shared memory (`/dev/shm`) is
separate from the host — which matters for `MPI_Win_allocate_shared` and
`MPI_COMM_TYPE_SHARED` (intra-node communicator).

```bash
# Share /dev/shm with host (needed for shared memory windows on some MPI impls)
apptainer exec --bind /dev/shm mpi_app.sif myapp

# Or use --ipc to give the container its own isolated IPC namespace:
apptainer exec --ipc mpi_app.sif myapp
```

```c
/* This pattern works in containers — MPI_COMM_TYPE_SHARED
   splits into node-local groups; the underlying shared memory
   is allocated by MPI (not /dev/shm directly) */
MPI_Comm node_comm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                    MPI_INFO_NULL, &node_comm);
int local_rank;
MPI_Comm_rank(node_comm, &local_rank);
```

---

## 33.8 Multi-Container Workflows

### Different Containers per Role

```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks=64

# Ranks 0-31: compute container
srun --ntasks=32 --mpi=pmix \
    apptainer exec compute.sif /usr/local/bin/solver &

# Ranks 32-63: I/O container (different software stack)
srun --ntasks=32 --mpi=pmix \
    apptainer exec io_server.sif /usr/local/bin/io_server &

wait
```

Note: this heterogeneous pattern requires careful PMIx coordination and is
implementation-specific. The simpler approach is one container for all ranks.

---

## 33.9 OCI Containers (Docker-Compatible)

Apptainer supports running OCI images (Docker Hub, container registries) directly:

```bash
# Pull from Docker Hub and convert to SIF
apptainer pull docker://nvcr.io/nvidia/hpc-benchmarks:23.10

# Run with GPU support
srun --ntasks=4 --gpus-per-task=1 --mpi=pmix \
    apptainer exec --nv hpc-benchmarks_23.10.sif /workspace/hpl.sh
```

### Building from a Dockerfile (Apptainer 1.2+)

```singularity
# hybrid.def — bootstrap from Docker image and add MPI
Bootstrap: docker
From: nvidia/cuda:12.3.0-devel-ubuntu22.04

%post
    apt-get update && apt-get install -y \
        libopenmpi-dev openmpi-bin

    # PMIx support
    apt-get install -y libpmix-dev pmix

    # Your application
    cd /build && make install
```

---

## 33.10 Debugging MPI in Containers

### Common Failure Modes

| Symptom | Likely Cause | Fix |
|---|---|---|
| `MPI_Init` hangs | PMIx socket not reachable | Check `--bind /tmp`, verify `$PMIX_SERVER_URI` |
| Ranks can't find each other | MPI ABI mismatch | Match host and container MPI version |
| Slow bandwidth | Network fabric not bound | Bind `/dev/infiniband`, use `--nv` or host verbs |
| Shared memory fails | `/dev/shm` namespace isolation | Add `--bind /dev/shm` or `--ipc` |
| GPU not visible | Missing `--nv`/`--rocm` flag | Add the appropriate flag |

### Diagnostic Commands

```bash
# Check PMIx environment inside container
srun --ntasks=1 --mpi=pmix \
    apptainer exec mpi_app.sif env | grep -E 'PMIX|OMPI|UCX|SLURM'

# Check MPI version mismatch
apptainer exec mpi_app.sif mpirun --version
mpirun --version   # host

# Test with simple hello world first
srun --ntasks=4 --mpi=pmix \
    apptainer exec mpi_app.sif /usr/local/bin/hello_mpi

# Enable MPI debugging
srun --ntasks=4 --mpi=pmix \
    apptainer exec \
    --env OMPI_MCA_ras_base_verbose=1 \
    --env PMIX_DEBUG=1 \
    mpi_app.sif myapp
```

---

## 33.11 Reproducible MPI Environments with Spack

Spack inside a container ensures reproducible MPI builds:

```singularity
# spack_mpi.def
Bootstrap: docker
From: ubuntu:22.04

%post
    apt-get update && apt-get install -y \
        git python3 build-essential gfortran

    git clone https://github.com/spack/spack.git /opt/spack
    . /opt/spack/share/spack/setup-env.sh

    # Install OpenMPI with PMIx support
    spack install openmpi@4.1.6 +pmix +ucx fabrics=ucx

    # Install your application with the Spack-managed MPI
    spack install myapp ^openmpi@4.1.6

%environment
    . /opt/spack/share/spack/setup-env.sh
    spack load myapp

%runscript
    exec myapp "$@"
```

---

## Summary

| Topic | Key Points |
|---|---|
| Container models | Host MPI (ABI match) vs PMIx-only (preferred for portability) |
| PMIx socket | Lives in `/tmp/pmix-*/`; accessible from container via default `/tmp` bind |
| SLURM launch | `srun --mpi=pmix apptainer exec image.sif myapp` |
| GPU support | `--nv` for NVIDIA, `--rocm` for AMD; auto-binds device libraries |
| Network fabric | Bind `/dev/infiniband`; set in `/etc/apptainer/apptainer.conf` |
| Shared memory | Add `--bind /dev/shm` or `--ipc` if MPI shared memory windows fail |
| CPU binding | SLURM `--cpu-bind` works through containers transparently |
| Debugging | Check `PMIX_SERVER_URI`, verify MPI version match, enable `PMIX_DEBUG=1` |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
