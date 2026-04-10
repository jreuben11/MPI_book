# Chapter 28: MPI with SLURM

## 28.1 The Role of SLURM

SLURM (Simple Linux Utility for Resource Management) is the dominant job scheduler
on HPC clusters. It allocates nodes, launches MPI processes, enforces time and memory
limits, and provides process management through PMIx.

The key relationship: SLURM and MPI are not independent. SLURM creates the process
management environment (PMIx or PMI2) that the MPI runtime uses to bootstrap — to
exchange endpoint addresses, set up collectives, and coordinate startup. Understanding
this relationship is essential for running MPI jobs correctly and efficiently.

---

## 28.2 srun vs. mpiexec

On SLURM clusters, **use `srun` instead of `mpiexec`**:

```bash
# Correct on SLURM
srun -n 128 ./myprogram

# Works but bypasses SLURM PMI integration on many clusters
mpiexec -n 128 ./myprogram
```

`srun` is SLURM's built-in parallel launcher. It communicates with SLURM's PMIx
daemon to set up process addresses — the same daemon that `MPI_Init` talks to.
`mpiexec` may work but relies on fallback mechanisms (TCP bootstrapping) that are
slower and less reliable than the native PMIx path.

---

## 28.3 Batch Job Script Structure

```bash
#!/bin/bash
#SBATCH --job-name=mpi_job          # Job name (appears in squeue)
#SBATCH --nodes=8                   # Number of nodes
#SBATCH --ntasks-per-node=32        # MPI ranks per node
#SBATCH --cpus-per-task=1           # Threads per MPI rank (set >1 for hybrid)
#SBATCH --time=02:00:00             # Wall time limit (HH:MM:SS)
#SBATCH --partition=compute         # Queue name
#SBATCH --account=proj123           # Allocation account
#SBATCH --output=job_%j.out         # stdout (%j = job ID)
#SBATCH --error=job_%j.err          # stderr
#SBATCH --mem-per-cpu=4G            # Memory per CPU

# Load required modules
module purge
module load gcc/12.3 openmpi/4.1.5

# Print environment for debugging
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Tasks: $SLURM_NTASKS"

# Run — srun reads ntasks and ntasks-per-node from SBATCH headers
srun ./myprogram --input data.bin --output result.bin
```

Submit with:
```bash
sbatch myjob.sh
squeue -u $USER          # check status
scancel $SLURM_JOB_ID   # cancel if needed
```

---

## 28.4 SLURM Environment Variables

SLURM sets environment variables that MPI programs can read to understand their
execution context:

| Variable | Meaning |
|---|---|
| `SLURM_JOB_ID` | Job ID |
| `SLURM_NTASKS` | Total number of MPI tasks |
| `SLURM_NPROCS` | Alias for `SLURM_NTASKS` |
| `SLURM_NTASKS_PER_NODE` | Tasks per node |
| `SLURM_PROCID` | This process's MPI rank (0-indexed) |
| `SLURM_LOCALID` | This process's rank within its node (0-indexed) |
| `SLURM_NODEID` | This process's node index (0-indexed) |
| `SLURM_NODELIST` | Comma-separated list of nodes |
| `SLURM_JOB_NUM_NODES` | Number of nodes allocated |
| `SLURM_CPUS_PER_TASK` | CPUs (threads) per task |
| `SLURM_MEM_PER_CPU` | Memory in MB per CPU |
| `SLURM_SUBMIT_DIR` | Directory from which sbatch was called |
| `SLURM_ARRAY_TASK_ID` | Task ID within a job array (if applicable) |

Using these in a program:

```c
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Cross-check MPI rank against SLURM's view */
    const char *slurm_procid = getenv("SLURM_PROCID");
    const char *slurm_localid = getenv("SLURM_LOCALID");

    if (rank == 0 && slurm_procid) {
        printf("SLURM_NTASKS=%s SLURM_JOB_NUM_NODES=%s\n",
               getenv("SLURM_NTASKS"), getenv("SLURM_JOB_NUM_NODES"));
    }

    /* Use SLURM_LOCALID as the within-node rank for NUMA decisions */
    int local_rank = slurm_localid ? atoi(slurm_localid) : 0;
    set_numa_affinity(local_rank);

    MPI_Finalize();
}
```

**Warning**: `SLURM_PROCID` and `MPI_Comm_rank(MPI_COMM_WORLD)` should agree, but
this is not guaranteed if MPI reorders processes. Query MPI directly for rank.
Use `SLURM_LOCALID` for node-local identity.

---

## 28.5 Process Binding and Topology

Binding MPI processes to specific CPU cores prevents OS migration and improves
cache locality. SLURM handles binding through `--cpu-bind`:

```bash
# Bind each task to a specific core
srun --cpu-bind=cores -n 64 ./myprogram

# Bind to sockets (NUMA nodes)
srun --cpu-bind=sockets -n 32 ./myprogram

# Verbose binding report (prints which core each process is on)
srun --cpu-bind=verbose,cores -n 8 ./myprogram
```

For hybrid MPI+OpenMP, set `--cpus-per-task` and use OpenMP's thread binding:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2       # 2 MPI ranks per node
#SBATCH --cpus-per-task=16        # 16 OpenMP threads per rank

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close        # bind OpenMP threads near MPI rank

srun --cpu-bind=sockets ./hybrid_program
```

### Inspecting Actual Placement

```c
#include <sched.h>   /* glibc */
#include <stdio.h>

void report_binding(int rank)
{
    cpu_set_t set;
    sched_getaffinity(0, sizeof(set), &set);

    char buf[256] = {0};
    int pos = 0;
    for (int i = 0; i < CPU_SETSIZE && pos < 250; i++)
        if (CPU_ISSET(i, &set))
            pos += sprintf(buf + pos, "%d ", i);

    printf("Rank %d bound to CPUs: %s\n", rank, buf);
}
```

---

## 28.6 PMIx: The MPI-SLURM Interface

PMIx (Process Management Interface — Exascale) is the protocol between SLURM's
slurmstepd and the MPI runtime. When `MPI_Init` is called:

1. MPI calls PMIx to register itself with the job's PMIx server.
2. PMIx exchanges endpoint information (network addresses, port numbers) between
   all MPI ranks.
3. MPI builds its internal routing tables from this information.

```
SLURM slurmstepd
    │
    ├─ PMIx server (one per node)
    │   ├─ MPI rank 0 (connects, gets endpoints)
    │   ├─ MPI rank 1
    │   └─ MPI rank 2
    │
    └─ PMIx server (next node)
        ├─ MPI rank 3
        └─ ...
```

You typically do not interact with PMIx directly. However, some advanced use cases
require PMIx:

- **Dynamic spawning**: `MPI_Comm_spawn` uses PMIx to launch new processes.
- **Process sets**: the Sessions model (`MPI_Session_init`) uses PMIx process set
  names (like `"mpi://WORLD"`).
- **Fault notification**: PMIx delivers fault events to ULFM-enabled programs.

To force a specific PMI interface (useful when debugging):

```bash
# Force PMIx (default for modern Open MPI on SLURM)
srun --mpi=pmix ./myprogram

# Force PMI2 (legacy)
srun --mpi=pmi2 ./myprogram

# List available PMI types on this system
srun --mpi=list
```

---

## 28.7 SLURM Job Arrays

Job arrays run the same script with different `SLURM_ARRAY_TASK_ID` values.
Each array element is an independent SLURM job with its own node allocation.

```bash
#!/bin/bash
#SBATCH --array=0-15               # 16 parameter sets (IDs 0..15)
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32

# Use SLURM_ARRAY_TASK_ID to select input parameters
PARAM_FILE="params/config_${SLURM_ARRAY_TASK_ID}.json"
OUTPUT_DIR="results/run_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTPUT_DIR"

srun ./myprogram --config "$PARAM_FILE" --outdir "$OUTPUT_DIR"
```

Inside your program, read the task ID for output naming:

```c
const char *task_id = getenv("SLURM_ARRAY_TASK_ID");
char outfile[256];
snprintf(outfile, sizeof(outfile), "results/run_%s/output.bin",
         task_id ? task_id : "0");
```

Limit concurrent jobs to avoid overwhelming the scheduler:

```bash
#SBATCH --array=0-99%16   # 100 jobs, max 16 running simultaneously
```

---

## 28.8 Interactive Jobs

For debugging, launch an interactive shell with resources:

```bash
# Interactive allocation: 2 nodes, 16 tasks, 1 hour
salloc --nodes=2 --ntasks-per-node=8 --time=01:00:00

# Within the interactive shell:
srun ./myprogram    # uses the allocated nodes

# Or run with GDB on rank 0:
srun --ntasks=1 gdb ./myprogram
```

For debugging all ranks simultaneously:

```bash
# Launch each rank in its own xterm
srun --ntasks=4 xterm -e gdb -ex run --args ./myprogram

# Or use mpirun for a direct (non-srun) launch in the allocation
mpirun -n 4 xterm -e gdb ./myprogram
```

---

## 28.9 I/O Considerations

### Parallel Filesystem Setup

Lustre configuration for optimal MPI-IO performance:

```bash
# Set striping on output directory before running
lfs setstripe -c 32 -S 4m /scratch/myproject/output/

# Check striping
lfs getstripe /scratch/myproject/output/result.bin
```

### Per-Process I/O Paths

Avoid having all MPI ranks write to the same filesystem path simultaneously using
stdio — use MPI-IO for shared files (Chapter 19) or direct each rank to a separate
file in a per-rank subdirectory:

```bash
# Create per-rank output directories in the batch script
for i in $(seq 0 $((SLURM_NTASKS - 1))); do
    mkdir -p "output/rank_$i"
done
```

### SLURM Tmpdir

Each node has a fast local scratch (`$TMPDIR` or `/tmp`) not visible to other nodes.
Use it for temporary per-rank files:

```c
const char *tmpdir = getenv("TMPDIR");
if (!tmpdir) tmpdir = "/tmp";

char localfile[512];
snprintf(localfile, sizeof(localfile), "%s/rank_%d_tmp.bin", tmpdir, rank);
/* Write to local disk, then MPI-IO aggregate to parallel filesystem */
```

---

## 28.10 Diagnosing SLURM + MPI Failures

### Common Failure Modes

| Symptom | Likely Cause |
|---|---|
| `srun: error: PMI2 init failed` | Wrong `--mpi` mode or PMIx version mismatch |
| Processes launch but hang at `MPI_Init` | Network fabric issue; check `ibstat` and OpenFabrics |
| `Out Of Memory` OOM kill | `--mem-per-cpu` too low; add `--mem=0` to use all available |
| `DUE TO TIME LIMIT` in output | Job exceeded `--time`; profile to find bottleneck |
| Inconsistent node counts | Node exclusion; check `sinfo -R` for drained nodes |

### Diagnostic Commands

```bash
# View job output in real-time
tail -f job_12345.out

# Check node health
sinfo -N -l | grep -v idle

# Inspect allocated nodes
scontrol show job $SLURM_JOB_ID | grep NodeList

# Check MPI network connectivity (Open MPI)
srun --ntasks=2 --nodes=2 mpi_hello   # minimal 2-node test

# Show OpenFabrics device status on all nodes
srun --ntasks-per-node=1 ibstat | grep -A2 State
```

---

## Summary

| Topic | Key Points |
|---|---|
| `srun` vs `mpiexec` | Use `srun` on SLURM for native PMIx integration |
| `--ntasks-per-node` | Primary knob for MPI rank layout |
| `--cpus-per-task` | Set to OMP_NUM_THREADS for hybrid jobs |
| `SLURM_PROCID` / `SLURM_LOCALID` | Node-local rank for NUMA/binding decisions |
| `--cpu-bind=cores` | Prevents OS migration; essential for performance |
| `--mpi=pmix` | Explicit PMIx selection; avoids fallback to TCP bootstrap |
| Job arrays | Parameter sweeps with independent allocations |
| Lustre striping | Set before job runs with `lfs setstripe` |
| `salloc` | Interactive allocation for debugging |
