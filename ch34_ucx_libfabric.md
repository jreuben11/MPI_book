# Chapter 34: UCX and libfabric — Network Transport Layers

## 34.1 The Transport Layer Stack

Modern MPI implementations do not talk to the network hardware directly. Instead they
use a transport layer that abstracts the fabric:

```
Application
    │
    └── MPI (Open MPI, MPICH, Cray MPICH)
            │
            ├── UCX (Unified Communication X)
            │       │
            │       ├── InfiniBand (verbs/RoCE)
            │       ├── CUDA IPC / ROCm IPC
            │       ├── Shared memory (POSIX, CMA)
            │       └── TCP (fallback)
            │
            └── libfabric / OFI (Open Fabrics Interface)
                    │
                    ├── verbs (InfiniBand)
                    ├── psm2 (Intel Omni-Path)
                    ├── cxi (Slingshot)
                    └── tcp (fallback)
```

Understanding these layers lets you diagnose performance problems and tune for your
cluster's network hardware without changing application code.

---

## 34.2 UCX (Unified Communication X)

UCX is the primary transport for Open MPI 4.x+ and MPICH on InfiniBand/RoCE clusters.

### Key Concepts

| Term | Meaning |
|---|---|
| Transport (`tl`) | Low-level driver: `rc` (reliable connected), `ud` (unreliable datagram), `dc` (dynamic connection), `shm` (shared memory), `self` (loopback) |
| Memory type | Where the buffer lives: host, CUDA device, ROCm device |
| `UCX_TLS` | Comma-separated list of enabled transports |
| Endpoint | A one-to-one connection handle between two processes |

### Checking UCX Configuration

```bash
# What UCX was built with
ucx_info -v

# Available transports on this machine
ucx_info -d

# What Open MPI will use
ompi_info --param btl all
mpirun -np 1 --mca pml ucx ucx_info -d

# Show all UCX config variables with current values
UCX_LOG_LEVEL=info mpirun -np 2 ./myapp 2>&1 | grep UCX
```

### UCX_TLS Selection

```bash
# Use all available (UCX chooses best per message)
export UCX_TLS=all

# InfiniBand reliable connected + GPU direct copy
export UCX_TLS=rc,cuda_copy,gdr_copy

# InfiniBand + shared memory + GPU IPC (within node)
export UCX_TLS=rc,shm,cuda_ipc,cuda_copy,gdr_copy

# Force TCP only (debugging — no RDMA)
export UCX_TLS=tcp

# Slingshot (HPE Cray EX) — Cray MPICH uses libfabric, not UCX
# But UCX on Slingshot: (rarely used)
export UCX_TLS=dc_x,shm

# Omni-Path (Intel) — legacy; prefer libfabric/psm2
export UCX_TLS=ud,shm
```

Common transport abbreviations:

| TL | Description | When to use |
|---|---|---|
| `rc` | InfiniBand Reliable Connected | Default for IB; good for point-to-point |
| `rc_x` | RC with accelerated verbs | Mellanox CX-5+ |
| `dc_x` | Dynamic Connection (scalable RC) | Very large job counts (>10k ranks) |
| `ud_x` | Unreliable Datagram | Broadcast-heavy workloads |
| `shm` | Shared memory (CMA/POSIX) | Intra-node; auto-selected |
| `cuda_ipc` | CUDA IPC (intra-node GPU) | Same-node GPU-to-GPU |
| `cuda_copy` | CUDA host-copy | Cross-node GPU data |
| `gdr_copy` | GPUDirect RDMA | Cross-node GPU; requires GDR driver |
| `rocm_copy` | ROCm host-copy | Cross-node AMD GPU |
| `tcp` | TCP sockets | Fallback; no RDMA |

### UCX Memory Registration

UCX registers memory regions with the network hardware. For large buffers this has overhead:

```bash
# Pre-register a memory pool (reduces registration overhead)
export UCX_RNDV_THRESH=16384    # use rendezvous protocol for messages > 16KB
export UCX_MEMTYPE_CACHE=n      # disable memtype cache if causing issues with CUDA
export UCX_IB_REG_METHODS=odp   # On-demand paging (avoids eager registration)
```

### UCX Rendezvous Threshold

```bash
# Tune the eager/rendezvous boundary (default varies by implementation)
# Messages below threshold: eager (sender pushes data immediately)
# Messages above threshold: rendezvous (sender waits for receiver ready)

export UCX_RNDV_THRESH=65536    # 64 KB threshold
export UCX_RNDV_SCHEME=put_zcopy  # sender writes directly to receiver buffer
# Alternatives: get_zcopy (receiver pulls), am (active message)
```

---

## 34.3 Diagnosing UCX Transport Issues

### Enable UCX Logging

```bash
# Verbose transport selection
export UCX_LOG_LEVEL=info
mpirun -np 4 ./myapp 2>&1 | grep -E 'UCX|tl|transport'

# Debug level (very verbose)
export UCX_LOG_LEVEL=debug

# Show which transport is selected per peer
export UCX_LOG_LEVEL=diag
```

### UCX Performance Test

```bash
# Test raw UCX bandwidth between two nodes
# (ucx_perftest is part of ucx-utils package)
# On node1:
ucx_perftest -t tag_bw -s 1048576 -n 1000

# On node2:
ucx_perftest -t tag_bw -s 1048576 -n 1000 node1

# Expected output: bandwidth in MB/s for the given transport
```

### Common UCX Problems

| Problem | Symptom | Fix |
|---|---|---|
| Wrong transport selected | Low bandwidth | Set `UCX_TLS` explicitly |
| GPUDirect not working | cudaMemcpy fallback | Set `UCX_TLS=...,gdr_copy`; verify GDR kernel module |
| Memory registration error | MPI_Init fails | Try `UCX_IB_REG_METHODS=odp` or reduce registration |
| Large job scalability | Slow MPI_Init (>10k ranks) | Switch from `rc` to `dc_x` |
| Endpoint not reachable | Rank cannot communicate | Check `UCX_NET_DEVICES`, verify IB port |

```bash
# Select specific network device (useful when multiple IB HCAs present)
export UCX_NET_DEVICES=mlx5_0:1   # HCA 0, port 1
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1  # Use both HCAs

# List available devices
ucx_info -d | grep Transport
```

---

## 34.4 libfabric / OFI (Open Fabrics Interface)

libfabric (the OFI implementation) is the transport used by:
- MPICH and its derivatives (Intel MPI, Cray MPICH, MVAPICH2)
- Open MPI 5.x (shifting from UCX to OFI)
- Cray MPICH on HPE Cray EX / Frontier / Perlmutter (Slingshot fabric)

### Key Concepts

| Term | Meaning |
|---|---|
| Provider | Fabric-specific plugin: `verbs`, `psm2`, `cxi` (Slingshot), `tcp`, `shm` |
| `FI_PROVIDER` | Select which provider to use |
| Endpoint type | `FI_EP_RDM` (reliable datagram, most common), `FI_EP_MSG` (connected) |
| Domain | Network interface abstraction |

### Checking libfabric Configuration

```bash
# List available providers
fi_info

# Detail a specific provider
fi_info -p verbs
fi_info -p cxi       # HPE Slingshot
fi_info -p psm2      # Intel Omni-Path

# Check MPICH libfabric selection
FI_LOG_LEVEL=info mpirun -np 2 ./myapp 2>&1 | grep -E 'FI_|provider'
```

### FI_PROVIDER Selection

```bash
# Force specific provider
export FI_PROVIDER=verbs          # InfiniBand
export FI_PROVIDER=cxi            # HPE Slingshot (Frontier, Perlmutter)
export FI_PROVIDER=psm2           # Intel Omni-Path
export FI_PROVIDER=tcp            # Fallback (debugging)
export FI_PROVIDER=shm            # Shared memory only

# MPICH-specific (maps to FI_PROVIDER)
export MPIR_CVAR_OFI_USE_PROVIDER=verbs
```

### Slingshot (HPE Cray EX) Configuration

Frontier and Perlmutter use the Cray Slingshot network with the `cxi` provider:

```bash
# Cray MPICH on Frontier/Perlmutter
export MPICH_OFI_NIC_POLICY=NUMA     # select NIC closest to CPU
export FI_CXI_RX_MATCH_MODE=software # or hardware; hardware is default
export FI_CXI_RDZV_THRESHOLD=16384   # rendezvous threshold in bytes

# GPU-aware MPI on Frontier (AMD ROCm + Cray MPICH)
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_VERBOSE=1       # log NIC selection

# Module setup on Frontier
module load PrgEnv-amd
module load craype-accel-amd-gfx90a  # MI250X
module load rocm
module load cray-mpich
```

### Intel Omni-Path (PSM2) Configuration

```bash
# Intel Omni-Path via PSM2 provider
export FI_PROVIDER=psm2
export PSM2_TRACEMASK=0x101          # verbose PSM2 logging
export PSM2_SHAREDCONTEXTS=0         # disable shared contexts if instability
export HFI_UNIT=0                    # select OPA HFI device 0
```

---

## 34.5 libfabric Performance Tuning

### Rendezvous Threshold

```bash
# MPICH / libfabric rendezvous threshold
export MPIR_CVAR_CH4_OFI_ENABLE_RNDV=1
# Per-collective eager thresholds (collective-specific CVARs):
export MPIR_CVAR_BCAST_SHORT_MSG_SIZE=131072     # bcast eager limit
export MPIR_CVAR_ALLREDUCE_SHORT_MSG_SIZE=131072  # allreduce short-message limit

# Intel MPI (uses libfabric internally)
export I_MPI_FABRICS=shm:ofi
export I_MPI_OFI_PROVIDER=verbs
export I_MPI_RDMA_THRESHOLD=65536
```

### libfabric Performance Test

```bash
# fabtest bandwidth benchmark
# On node1:
fi_rdm_tagged_bw -e rdm -p verbs -s 1048576

# On node2:
fi_rdm_tagged_bw -e rdm -p verbs -s 1048576 node1

# Latency test
fi_rdm_pingpong -e rdm -p verbs
```

---

## 34.6 Choosing Between UCX and libfabric

| | UCX | libfabric |
|---|---|---|
| Primary MPI | Open MPI 4.x | MPICH, Cray MPICH, Intel MPI |
| InfiniBand | Excellent | Good (verbs provider) |
| Slingshot | Not typical | Native (cxi provider) |
| Omni-Path | Partial | Native (psm2 provider) |
| GPU direct | Excellent (cuda_ipc, gdr_copy) | Good (mpich cuda support) |
| Tuning vars | `UCX_*` | `FI_*`, `MPIR_CVAR_*` |
| Diagnostic tools | `ucx_info`, `ucx_perftest` | `fi_info`, `fi_rdm_tagged_bw` |

The choice is usually dictated by which MPI implementation your cluster provides.

---

## 34.7 MPI Runtime Detection of Transport

```c
#include <mpi.h>
#include <stdio.h>

/* Print transport info via MPI_T interface */
void print_transport_info(void)
{
    int provided;   /* receives the thread support level actually provided */
    MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);

    /* Search for UCX or OFI provider cvar */
    int num = 0;
    MPI_T_cvar_get_num(&num);
    for (int i = 0; i < num; i++) {
        char name[256]; int namelen = 256;
        char desc[1024]; int desclen = 1024;
        int verb, mt_bind, mt_scope;   /* bind before scope per MPI standard */
        MPI_Datatype dtype; MPI_T_enum enumtype;
        MPI_T_cvar_get_info(i, name, &namelen, &verb, &dtype,
                            &enumtype, desc, &desclen,
                            &mt_bind, &mt_scope);   /* bind is 9th, scope is 10th */
        /* Print variables related to transport selection */
        if (strstr(name, "UCX") || strstr(name, "OFI") ||
            strstr(name, "PROVIDER") || strstr(name, "FABRICS")) {
            printf("CVAR: %s\n", name);
        }
    }
    MPI_T_finalize();
}
```

---

## 34.8 Environment-Based Configuration Reference

### Open MPI + UCX (InfiniBand)

```bash
# Full recommended setup for Open MPI on InfiniBand cluster
export OMPI_MCA_pml=ucx
export UCX_TLS=rc_x,shm,self
export UCX_NET_DEVICES=mlx5_0:1
export UCX_RNDV_THRESH=65536
export UCX_RNDV_SCHEME=put_zcopy
export OMPI_MCA_btl=^vader,tcp,uct  # disable slower BTLs
```

### Open MPI + UCX (GPU cluster)

```bash
export OMPI_MCA_pml=ucx
export UCX_TLS=rc_x,cuda_ipc,cuda_copy,gdr_copy,shm,self
export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_opal_cuda_support=1
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID  # set in SLURM job
```

### Cray MPICH on Slingshot (Frontier)

```bash
export MPICH_OFI_NIC_POLICY=NUMA
export FI_CXI_RDZV_THRESHOLD=16384
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_MR_CACHE_MONITOR=kdreg2  # kernel memory monitoring (best on RHEL8+)
```

### MPICH on InfiniBand

```bash
export MPIR_CVAR_OFI_USE_PROVIDER=verbs
export MPIR_CVAR_CH4_OFI_ENABLE_RNDV=1
export FI_VERBS_IFACE=ib0           # select IB interface
```

---

## 34.9 Diagnosing Network Performance Problems

### Step-by-Step Diagnosis

```bash
# Step 1: Verify the correct transport is being used
UCX_LOG_LEVEL=info mpirun -np 2 ./myapp 2>&1 | grep -i transport

# Step 2: Run a bandwidth benchmark (OSU microbenchmarks)
mpirun -np 2 osu_bw
# Expected: 90+ GB/s for NVLink, 25+ GB/s for HDR InfiniBand, <1 GB/s for TCP

# Step 3: Check latency
mpirun -np 2 osu_latency
# Expected: <1 µs for NVLink, ~1 µs for InfiniBand, >50 µs for TCP

# Step 4: If bandwidth is low — check transport selection
mpirun -np 2 --map-by node osu_bw  # inter-node
mpirun -np 2 osu_bw               # intra-node (should use shm)
```

### OSU Microbenchmarks

```bash
# Install
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.3.tar.gz
tar xzf osu-micro-benchmarks-7.3.tar.gz
cd osu-micro-benchmarks-7.3
./configure CC=mpicc CXX=mpic++ --prefix=$HOME/osu
make -j4 install

# Run point-to-point benchmarks
mpirun -np 2 --map-by node $HOME/osu/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw
mpirun -np 2 --map-by node $HOME/osu/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency

# Collective benchmarks
mpirun -np 16 $HOME/osu/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce
```

---

## Summary

| Topic | Key Points |
|---|---|
| UCX | Used by Open MPI; `UCX_TLS` selects transports; `ucx_info -d` shows capabilities |
| UCX GPU | `gdr_copy` for cross-node GPU; `cuda_ipc` for intra-node; `UCX_MEMTYPE_CACHE=n` for CUDA |
| libfabric | Used by MPICH/Intel MPI/Cray MPICH; `FI_PROVIDER` selects driver |
| Slingshot | `FI_PROVIDER=cxi`; Cray MPICH on Frontier/Perlmutter |
| Omni-Path | `FI_PROVIDER=psm2`; legacy Intel fabric |
| InfiniBand | `UCX_TLS=rc_x` (UCX) or `FI_PROVIDER=verbs` (libfabric) |
| Rendezvous | Tune threshold with `UCX_RNDV_THRESH` or `MPIR_CVAR_CH4_OFI_ENABLE_RNDV` |
| Diagnosis | OSU microbenchmarks; `fi_info`; `UCX_LOG_LEVEL=info`; `fi_rdm_tagged_bw` |
