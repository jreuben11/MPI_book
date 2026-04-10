# Chapter 2: Setup & First Programs

## 2.1 Installing MPI

### Linux (Package Manager)

```bash
# Ubuntu / Debian — Open MPI
sudo apt install libopenmpi-dev openmpi-bin

# Ubuntu / Debian — MPICH
sudo apt install libmpich-dev mpich

# RHEL / Fedora
sudo dnf install openmpi-devel   # then: module load mpi/openmpi-x86_64
sudo dnf install mpich-devel
```

### macOS

```bash
brew install open-mpi
# or
brew install mpich
```

### Cluster / HPC Systems

On clusters, MPI is almost always installed as an **environment module**:

```bash
module avail mpi          # list available MPI modules
module load mpi/openmpi   # load one
module list               # confirm
```

Never build MPI from source on a managed cluster unless you have a specific reason —
the system MPI is tuned for the interconnect hardware.

### Verifying the Installation

```bash
mpicc --version
mpiexec --version
mpicc -show          # shows the underlying compiler command and flags
```

`mpicc -show` is your best diagnostic tool: it reveals what compiler, include paths,
and link flags the wrapper uses.

---

## 2.2 Compiler Wrappers

MPI provides wrapper scripts around your system compiler. They inject the correct
include paths and link flags automatically.

| Wrapper | Language |
|---|---|
| `mpicc` | C |
| `mpic++` or `mpicxx` | C++ |
| `mpifort` | Fortran (not covered here) |

You pass all normal compiler flags through the wrapper:

```bash
mpicc   -O2 -Wall -o myprogram myprogram.c
mpic++  -O2 -std=c++17 -o myprogram myprogram.cpp
```

### ABI Considerations

Open MPI and MPICH have **incompatible ABIs**. A binary compiled against Open MPI
cannot load MPICH's `libmpi.so` at runtime, and vice versa. On clusters with multiple
MPI installations, always load the module before compiling, and use the same module
when running.

If you are writing a library that will be used by downstream MPI programs:
- Do not link `libmpi` into your static library — let the application link it.
- Document which MPI implementation and version you tested against.

### CMake Integration

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyMPIProject CXX)

find_package(MPI REQUIRED)

add_executable(myprogram main.cpp)
target_link_libraries(myprogram MPI::MPI_CXX)
```

CMake's `FindMPI` module sets `MPI::MPI_C` and `MPI::MPI_CXX` imported targets that
carry the correct include paths and link libraries. Do not manually set `-I` or `-l`
flags — let CMake do it.

---

## 2.3 Hello World

```c
/* hello.c */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello from rank %d of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
```

```bash
mpicc -O2 -o hello hello.c
mpiexec -n 4 ./hello
```

Expected output (order is non-deterministic):

```
Hello from rank 2 of 4
Hello from rank 0 of 4
Hello from rank 3 of 4
Hello from rank 1 of 4
```

The same program in C++:

```cpp
// hello.cpp
#include <cstdio>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::printf("Hello from rank %d of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
```

```bash
mpic++ -O2 -std=c++17 -o hello hello.cpp
mpiexec -n 4 ./hello
```

Note: there are no special C++ MPI classes. The C++ bindings that MPI 2.0 introduced
(`MPI::COMM_WORLD.Send(...)`) were deprecated in MPI 2.2 and removed in MPI 3.0.
All C++ MPI code uses the C API (`MPI_Send(...)`) with the same `mpi.h` header.

---

## 2.4 Basic Program Structure

Every MPI program follows this skeleton:

```c
#include <mpi.h>

int main(int argc, char *argv[])
{
    /* --- 1. Initialize MPI --- */
    MPI_Init(&argc, &argv);
    // After this point: MPI functions may be called.
    // Before this point: only MPI_Initialized and MPI_Finalized are safe.

    /* --- 2. Get identity --- */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* --- 3. Do work, communicate --- */
    // ... your code ...

    /* --- 4. Finalize MPI --- */
    MPI_Finalize();
    // After this point: MPI functions must NOT be called (except MPI_Finalized).
    return 0;
}
```

### MPI_Init

```c
int MPI_Init(int *argc, char ***argv);
```

- Pass pointers to `argc` and `argv` from `main`. MPI may consume MPI-specific
  arguments from the command line (though most implementations do not).
- Passing `NULL, NULL` is legal but not recommended.
- Must be called exactly once, before any other MPI function.
- Is **not** thread-safe: call it from the main thread only. If you need thread
  support, use `MPI_Init_thread` instead (Chapter 21).

### MPI_Finalize

```c
int MPI_Finalize(void);
```

- Must be called exactly once, by all processes, after all MPI communication is
  complete.
- All pending non-blocking operations must be completed before calling `MPI_Finalize`.
- After `MPI_Finalize` returns, no MPI function may be called (except `MPI_Finalized`).

### Checking Init/Finalize State

```c
int flag;
MPI_Initialized(&flag);  // flag = 1 if MPI_Init has been called
MPI_Finalized(&flag);    // flag = 1 if MPI_Finalize has been called
```

These are the only two MPI functions safe to call before `MPI_Init` or after
`MPI_Finalize`. Libraries use them to avoid re-initializing MPI.

---

## 2.5 Running MPI Programs

### mpiexec / mpirun

`mpiexec` is the standard launcher (defined by MPI). `mpirun` is an alias provided by
most implementations. Prefer `mpiexec` for portability.

```bash
# Run 8 processes
mpiexec -n 8 ./myprogram

# Run with arguments passed to the program
mpiexec -n 4 ./myprogram --input data.bin --output result.bin

# Run on specific hosts
mpiexec -n 4 -host node01,node02 ./myprogram

# Run 2 processes per node across 4 nodes
mpiexec -n 8 --map-by node --npernode 2 ./myprogram

# Bind processes to cores (Open MPI)
mpiexec -n 4 --bind-to core ./myprogram
```

### PMIx and Job Schedulers

On clusters, `mpiexec` is typically not called directly. Instead, the job scheduler
(SLURM, PBS, LSF) calls MPI's process management interface:

```bash
# SLURM: srun directly replaces mpiexec
srun -n 128 --ntasks-per-node 32 ./myprogram

# SLURM: batch script
#!/bin/bash
#SBATCH -N 4
#SBATCH --ntasks-per-node 32
#SBATCH --time 01:00:00
srun ./myprogram
```

PMIx (Process Management Interface — Exascale) is the protocol between the job
scheduler and MPI runtime. Open MPI 4.x and MPICH 4.x both use PMIx natively.
You generally do not interact with PMIx directly.

### Hostfiles

For running across multiple nodes without a scheduler:

```bash
# hostfile content:
# node01 slots=8
# node02 slots=8

mpiexec -n 16 --hostfile myhosts ./myprogram
```

### Environment Variables

Useful environment variables (Open MPI-specific, but most have MPICH equivalents):

```bash
OMPI_MCA_btl=^openib          # disable a transport
OMPI_MCA_coll_tuned_use_dynamic_rules=1   # enable tuned collective rules
MPICH_ASYNC_PROGRESS=1        # enable async progress thread (MPICH)
```

Use `ompi_info` or `mpichversion` to see available parameters.

---

## 2.6 A More Complete Example: Ping-Pong Benchmark

This program measures round-trip latency between rank 0 and rank 1:

```c
/* pingpong.c */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WARMUP_ITERS  100
#define BENCH_ITERS   1000
#define MAX_SIZE      (1 << 22)   /* 4 MB */

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    char *buf = malloc(MAX_SIZE);
    if (!buf) { MPI_Abort(MPI_COMM_WORLD, 1); }

    if (rank == 0)
        printf("%12s  %12s  %12s\n", "Bytes", "Latency(us)", "Bandwidth(MB/s)");

    for (int sz = 1; sz <= MAX_SIZE; sz *= 2) {

        /* Warmup */
        for (int i = 0; i < WARMUP_ITERS; i++) {
            if (rank == 0) {
                MPI_Send(buf, sz, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, sz, MPI_BYTE, 1, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(buf, sz, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(buf, sz, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }

        /* Benchmark */
        double t0 = MPI_Wtime();
        for (int i = 0; i < BENCH_ITERS; i++) {
            if (rank == 0) {
                MPI_Send(buf, sz, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, sz, MPI_BYTE, 1, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(buf, sz, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                MPI_Send(buf, sz, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }
        double t1 = MPI_Wtime();

        if (rank == 0) {
            double lat_us = (t1 - t0) / (2.0 * BENCH_ITERS) * 1e6;
            double bw_mbs = (sz / 1e6) / ((t1 - t0) / (2.0 * BENCH_ITERS));
            printf("%12d  %12.2f  %12.2f\n", sz, lat_us, bw_mbs);
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
```

Key points illustrated by this example:

- `MPI_Wtime()` returns wall-clock time in seconds as a `double`. It is the standard
  MPI timing function. Its resolution can be queried with `MPI_Wtick()`.
- `MPI_STATUS_IGNORE` tells MPI you do not need the status information from a receive.
  This is more efficient than passing a real `MPI_Status` variable when you do not
  intend to inspect it.
- `MPI_Abort(MPI_COMM_WORLD, errorcode)` terminates all processes immediately. Use it
  only for unrecoverable errors; prefer clean `MPI_Finalize` when possible.

---

## 2.7 Querying the Environment

```c
/* env_query.c — print MPI environment info */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Processor name (usually hostname) */
    char name[MPI_MAX_PROCESSOR_NAME];
    int namelen;
    MPI_Get_processor_name(name, &namelen);

    /* MPI library version string */
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    int verlen;
    MPI_Get_library_version(version, &verlen);

    /* MPI standard version numbers */
    int ver, subver;
    MPI_Get_version(&ver, &subver);

    if (rank == 0) {
        printf("MPI standard version:  %d.%d\n", ver, subver);
        printf("Library version:\n%s\n", version);
    }

    printf("Rank %d running on %s\n", rank, name);

    MPI_Finalize();
    return 0;
}
```

Useful constants:

| Constant | Value | Meaning |
|---|---|---|
| `MPI_MAX_PROCESSOR_NAME` | ≥128 | Max length of processor name string |
| `MPI_MAX_ERROR_STRING` | ≥256 | Max length of error string |
| `MPI_MAX_LIBRARY_VERSION_STRING` | ≥8192 | Max length of library version string |
| `MPI_MAX_OBJECT_NAME` | ≥128 | Max length of communicator/window/file name |

---

## 2.8 C++ Wrapper Pattern

Since MPI has no official C++ API, production C++ code often wraps the init/finalize
lifecycle in an RAII guard:

```cpp
// mpi_guard.hpp
#pragma once
#include <mpi.h>
#include <stdexcept>

class MPIGuard {
public:
    MPIGuard(int &argc, char **&argv) {
        int already;
        MPI_Initialized(&already);
        if (!already) {
            if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
                throw std::runtime_error("MPI_Init failed");
        }
    }

    ~MPIGuard() {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized)
            MPI_Finalize();
    }

    // Non-copyable, non-movable
    MPIGuard(const MPIGuard &) = delete;
    MPIGuard &operator=(const MPIGuard &) = delete;

    int rank() const {
        int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r;
    }
    int size() const {
        int s; MPI_Comm_size(MPI_COMM_WORLD, &s); return s;
    }
};
```

```cpp
// main.cpp
#include "mpi_guard.hpp"
#include <cstdio>

int main(int argc, char *argv[])
{
    MPIGuard mpi(argc, argv);
    std::printf("Rank %d of %d\n", mpi.rank(), mpi.size());
    return 0;
}   // MPIGuard destructor calls MPI_Finalize
```

This pattern ensures `MPI_Finalize` is called even if an exception is thrown, and
prevents double-initialization when multiple libraries all try to call `MPI_Init`.
The Sessions model (Chapter 22) offers a more principled solution to the
multi-library problem.

---

## Summary

| Topic | Key Points |
|---|---|
| Wrappers | Use `mpicc`/`mpic++`; pass through normal compiler flags |
| CMake | `find_package(MPI REQUIRED)` + `MPI::MPI_CXX` target |
| `MPI_Init` | Must be first; pass `&argc, &argv`; call once from main thread |
| `MPI_Finalize` | Must be last; all processes must call it; all requests must be done |
| `mpiexec -n N` | Launches N processes; `srun` on SLURM clusters |
| `MPI_Wtime()` | Wall-clock timer in seconds; use for benchmarking |
| `MPI_STATUS_IGNORE` | Use when you do not need receive status metadata |
| `MPI_Abort` | Emergency termination; not a substitute for clean error handling |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
