# Chapter 26: Profiling & Debugging

## 26.1 The PMPI Profiling Layer

MPI defines a standardized profiling interface: every MPI function `MPI_Foo` has
a corresponding `PMPI_Foo` symbol that goes directly to the implementation. By
intercepting `MPI_Foo` and delegating to `PMPI_Foo`, you can wrap every MPI call
without modifying application code.

```c
/* A minimal timing interceptor using the PMPI layer */

#include <mpi.h>
#include <stdio.h>
#include <time.h>

static double total_send_time = 0.0;
static long   total_send_bytes = 0;

int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm)
{
    int size;
    MPI_Type_size(datatype, &size);

    double t0 = MPI_Wtime();
    int rc = PMPI_Send(buf, count, datatype, dest, tag, comm);
    double t1 = MPI_Wtime();

    total_send_time  += (t1 - t0);
    total_send_bytes += (long)count * size;
    return rc;
}

int MPI_Finalize(void)
{
    int rank;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[Rank %d] Total MPI_Send: %.3f s, %ld bytes\n",
           rank, total_send_time, total_send_bytes);
    return PMPI_Finalize();
}
```

Compile as a shared library and `LD_PRELOAD` it, or link before `libmpi`:

```bash
mpicc -shared -fPIC -o libmpi_profile.so mpi_profile.c
LD_PRELOAD=./libmpi_profile.so mpiexec -n 4 ./myprogram
```

The PMPI layer is the foundation of all major MPI profiling tools.

---

## 26.2 The MPI_T Interface

MPI 3.0 introduced the **MPI_T** interface — a standardized way to query and
control MPI implementation internals. It exposes two categories:

- **Performance Variables (PVars)**: read-only metrics (message counts, bytes,
  algorithm selection, network counters).
- **Control Variables (CVars)**: read-write knobs (buffer sizes, algorithm selection,
  threshold values).

### Initializing MPI_T

```c
int provided;   /* receives the thread support level actually provided */
MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
/* ... use MPI_T ... */
MPI_T_finalize();
```

`MPI_T_init_thread` and `MPI_T_finalize` are independent of `MPI_Init`/`MPI_Finalize`.

### Querying Control Variables

```c
int num_cvars;
MPI_T_cvar_get_num(&num_cvars);

for (int i = 0; i < num_cvars; i++) {
    int namelen = 0, verbosity, binding, scope;
    MPI_Datatype datatype;
    MPI_T_enum enumtype;
    char name[256], desc[1024];
    int desclen = sizeof(desc);

    namelen = sizeof(name);
    MPI_T_cvar_get_info(i, name, &namelen, &verbosity,
                         &datatype, &enumtype, desc, &desclen,
                         &binding, &scope);
    printf("CVAR %d: %s\n", i, name);
}
```

### Reading and Setting a Control Variable

```c
/* Example: change the eager limit */
int cvar_index;
MPI_T_cvar_handle handle;

/* Find the eager limit variable by name */
MPI_T_cvar_get_index("MPIR_CVAR_CH4_OFI_EAGER_MAX_MSG_SIZE", &cvar_index);
MPI_T_cvar_handle_alloc(cvar_index, NULL, &handle, NULL);

/* Read current value */
int current_eager;
MPI_T_cvar_read(handle, &current_eager);
printf("Eager limit: %d bytes\n", current_eager);

/* Set new value */
int new_eager = 65536;
MPI_T_cvar_write(handle, &new_eager);

MPI_T_cvar_handle_free(&handle);
```

CVar names are implementation-specific. Query `MPI_T_cvar_get_info` for the list
available in your MPI build.

### Reading Performance Variables

```c
int num_pvars;
MPI_T_pvar_get_num(&num_pvars);

/* Find a pvar by name, allocate a session and handle */
MPI_T_pvar_session session;
MPI_T_pvar_session_create(&session);

int pvar_index;
MPI_T_pvar_get_index("MPI_Allreduce_count", MPI_T_PVAR_CLASS_COUNTER, &pvar_index);

MPI_T_pvar_handle phandle;
int count;
MPI_T_pvar_handle_alloc(session, pvar_index, NULL, &phandle, &count);

MPI_T_pvar_start(session, phandle);  /* start counting */

/* ... run application code ... */

MPI_T_pvar_stop(session, phandle);

unsigned long long allreduce_count;
MPI_T_pvar_read(session, phandle, &allreduce_count);
printf("MPI_Allreduce calls: %llu\n", allreduce_count);

MPI_T_pvar_handle_free(session, &phandle);
MPI_T_pvar_session_free(&session);
```

---

## 26.3 External Profiling Tools

### Score-P

Score-P is the most widely used open-source MPI profiling framework on HPC systems.
It intercepts MPI calls via the PMPI layer and records traces.

```bash
# Compile with Score-P instrumentation
scorep mpicc -O2 -o myprogram myprogram.c

# Run
mpiexec -n 64 ./myprogram

# Analyze with Cube (GUI) or OTF2 trace
cube scorep-*/profile.cubex
```

Score-P produces:
- **Flat profiles**: time per function (callpath aggregated)
- **OTF2 traces**: full event timeline, viewable in Vampir

### TAU (Tuning and Analysis Utilities)

TAU provides measurement, analysis, and visualization of parallel performance.
Strong support for hybrid MPI+OpenMP+CUDA.

```bash
# Instrument and run
tau_cc.sh -O2 -o myprogram myprogram.c
mpiexec -n 64 ./myprogram
# Results in tauprofile/
paraprof  # launch GUI analyzer
```

### mpiP

Lightweight MPI profiling — minimal overhead (< 1%), focused on MPI-only statistics.

```bash
gcc -O2 -o myprogram myprogram.c -lmpiP -lmpi -lbfd
mpiexec -n 64 ./myprogram
# Creates mpiP.*.mpiP file
```

Reports: total MPI time, per-call statistics, top hotspots. No trace — very low
overhead suitable for production runs.

### Intel Trace Analyzer (ITAC)

Commercial tool integrated with Intel MPI. Provides timeline visualization and
correctness checking (detects deadlocks, incorrect usage).

```bash
mpiexec -n 64 -trace ./myprogram
traceanalyzer *.stf  # launch GUI
```

### Vampir

Timeline visualization for OTF2 traces produced by Score-P. Shows per-process
event timelines, communication patterns, load imbalance.

---

## 26.4 Common Bug Checklist

### Deadlock

**Symptoms**: program hangs; all processes blocked waiting.

**Diagnosis**:
```bash
# Attach GDB to a running process
gdb -p $(pgrep -f myprogram | head -1)
# Look at the backtrace — usually inside an MPI function
bt
```

**Common causes**:
- `MPI_Send` before `MPI_Recv` on every process (Section 5.6)
- Missing `MPI_Wait` for a non-blocking operation
- Collective call missing from one process
- Asymmetric tag or communicator usage

### Message Mismatch

**Symptoms**: wrong results; `MPI_ERR_TRUNCATE`; random data.

**Diagnosis**: add `MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN)` and check
return codes. Print send/recv parameters with rank and tag.

**Common causes**:
- Tag mismatch between sender and receiver
- Count or datatype mismatch
- Wrong `MPI_ANY_TAG` usage allowing a wrong message to be matched

### Buffer Safety Violation

**Symptoms**: intermittent wrong results; results change with process count or
message size; race conditions visible only under large loads.

```bash
# MPICH: enable strict buffer checking
export MPICH_NO_LOCAL=1  # force network path even for shared memory
```

**Common causes**:
- Reusing `buf` before `MPI_Wait` completes a non-blocking operation
- Reusing sendbuf during an active `MPI_Win_fence` epoch
- Stack-allocated buffer passed to `MPI_Isend` goes out of scope

### Type Mismatch

**Symptoms**: garbled data at receiver; platform-specific failures.

**Cause**: `MPI_LONG` is 4 bytes on Windows, 8 bytes on Linux. Use `MPI_INT32_T`
/ `MPI_INT64_T` / `MPI_UINT64_T` for portable cross-platform code.

### Unbalanced Collective

**Symptoms**: hang at a collective call; only some processes call it.

```c
/* Wrong: conditional collective */
if (rank == 0 || some_condition) {
    MPI_Barrier(comm);  /* other processes skip this! */
}
```

Fix: ensure all processes in the communicator call the collective unconditionally.
If you need conditional behavior, compute the condition first, then call the
collective on all processes.

---

## 26.5 Systematic Debugging Approach

1. **Enable error checking**: `MPI_ERRORS_RETURN` + check every return code.

2. **Replace `MPI_Send` with `MPI_Ssend`**: converts potential deadlocks into
   guaranteed deadlocks, making them reproducible (Section 7.4).

3. **Run with 1 process first**: many bugs appear with single-process runs.

4. **Run with 2, then 4, then P processes**: deadlocks often first appear at specific
   counts.

5. **Use `MPI_Wtime` bracketing**: bracket suspected sections with timers. A section
   that takes longer than expected is probably waiting.

6. **GDB with `mpiexec`**:
```bash
mpiexec -n 4 xterm -e gdb ./myprogram
# Or for headless
mpiexec -n 4 gdb -ex run --args ./myprogram
```

7. **Valgrind for memory errors** (single-process first, then multi-process):
```bash
mpiexec -n 4 valgrind --leak-check=full ./myprogram
```

---

## Summary

| Tool/Technique | Purpose |
|---|---|
| PMPI layer | Intercept MPI calls; build custom profilers |
| `MPI_T` CVars | Read/write implementation tuning parameters |
| `MPI_T` PVars | Read implementation performance counters |
| Score-P | Full tracing + profiling; OTF2 output for Vampir |
| TAU | Multi-paradigm profiling (MPI+OpenMP+CUDA) |
| mpiP | Lightweight MPI-only statistics; < 1% overhead |
| ITAC | Intel MPI timeline + correctness checker |
| `MPI_Ssend` debugging | Force deadlock detection regardless of message size |
| GDB attach | Diagnose hangs; inspect blocked process stacks |
| `MPI_ERRORS_RETURN` | Enable recoverable error checking |

**Debugging priority**:
1. Correct results on 1 process
2. Correct results on 2 processes
3. Correct results on P processes
4. Profile and optimize

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
