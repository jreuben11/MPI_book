# Preface

## Why This Book

The MPI standard is vast. The printed specification for MPI 4.0 runs to over a
thousand pages; MPI 5.0 adds more. Yet most HPC programs use a small, stable
subset of that surface area — blocking point-to-point, a handful of collectives,
and occasionally one-sided operations or parallel I/O.

This guide takes a different approach. It targets the practical 20% of MPI that
appears in 80% of real programs, covers it deeply with working C/C++ examples,
and then extends into the modern HPC ecosystem: GPU-direct communication, AMD
ROCm, container runtimes, network transport tuning, fault-tolerant checkpointing,
and the Python `mpi4py` bindings. It is organized as a programming guide, not a
reference manual.

## Audience

This book assumes you:

- Are comfortable writing C (C99 or later) or C++ (C++17 or later).
- Have a basic understanding of parallel computing — processes, shared vs.
  distributed memory, and why inter-process communication is necessary.
- Are working on, or will soon work on, an HPC cluster or a multi-node GPU
  system.

You do **not** need prior MPI experience. Chapter 1 builds the conceptual model
from scratch; Chapter 2 gets a working program running in under ten minutes.

Experienced MPI programmers can skip to Chapter 11 (non-blocking and persistent
collectives), Chapter 22 (the MPI 4.0 Sessions model), or Chapter 23
(partitioned communication) to pick up what is new in MPI 4.0/5.0.

## MPI Standard Coverage

This guide covers **MPI 4.0** completely and **MPI 5.0** as it has been
standardized (ULFM fault tolerance, RMA clarifications, additional `MPI_Info`
keys). Every code example compiles against Open MPI 5.x or MPICH 4.x.

The following are deliberately excluded:

| Excluded | Reason |
|---|---|
| Fortran bindings | Separate dialect; C API is the universal interop layer |
| Full `MPI_T` specification | Ch. 26 covers the practical subset |
| `MPI_Comm_spawn` / dynamic processes | Sessions model is the modern replacement |
| Formal compliance requirements | Implementation concern, not user concern |
| C++ bindings (`MPI::COMM_WORLD`, etc.) | Deprecated in MPI 2.2, removed in MPI 3.0 |

## How to Use This Book

**Linear reading**: Parts I–IV (Chapters 1–13) form the core. Read these in
order the first time through. They build on each other: you cannot understand
collective behavior without understanding communicators, and you cannot tune
performance without understanding eager vs. rendezvous protocols.

**Reference use**: The appendices (Appendix A: function signatures, Appendix B:
constants and types, Appendix C: migration notes) are designed for daily lookup.
Appendix A groups functions by category, not alphabetically, so related functions
appear together.

**Topic jumping**: Parts V–X (Chapters 14–36) are largely independent. A
developer working on GPU clusters can read Part X chapters (30–32) immediately
after Part I without loss of continuity. A developer tuning collective
performance can go directly to Chapter 25.

**Integration chapters** (27–36) each stand alone: they require only the core
P2P and collective knowledge from Parts I–III.

## Code Examples

All code examples are complete, compilable, and verified against the API
signatures in the MPI 4.0 and MPI 5.0 standards. They follow these conventions:

- **C is used by default**. C++ is used where a C++17/20 feature adds
  meaningful clarity (Chapters 27, and C++ compilation notes throughout).
- **Error checking**: Most examples use `MPI_ERRORS_RETURN` and check return
  codes. Production code should always check errors; tutorial code sometimes
  omits checks for brevity — this is noted explicitly where it occurs.
- **No deprecated API**: The C++ bindings (`MPI::` namespace) are never used.
  Old-style functions (`MPI_Type_struct`, `MPI_Address`, etc.) appear only in
  migration examples showing what *not* to write.

## A Note on Implementation Differences

MPI implementations — Open MPI, MPICH, Intel MPI, Cray MPICH, HPC-X — conform
to the same standard but differ in:

- Default buffer sizes and eager/rendezvous thresholds
- Available network transports (UCX vs. libfabric; see Chapter 34)
- Degree of MPI 4.0/5.0 feature support
- Environment variable names for tuning

Where behavior differs significantly between implementations, the relevant
chapter notes the difference and provides implementation-specific configuration.
Use `MPI_Get_library_version` to query the running implementation at runtime.

## Building the Examples

```bash
# Install Open MPI (Ubuntu / Debian)
sudo apt install libopenmpi-dev openmpi-bin

# Install MPICH
sudo apt install libmpich-dev mpich

# Compile C
mpicc -O2 -Wall -o myprogram myprogram.c

# Compile C++17
mpic++ -O2 -std=c++17 -Wall -o myprogram myprogram.cpp

# Run on 4 processes
mpiexec -n 4 ./myprogram

# Run with SLURM (Chapter 28)
srun -n 64 ./myprogram
```

For GPU examples (Chapters 30–32), see the build instructions in each chapter.
CUDA examples use `nvcc`; HIP examples use `hipcc` or CMake with the HIP
language support. All GPU chapters include CMake `CMakeLists.txt` snippets.

## Typographic Conventions

| Convention | Meaning |
|---|---|
| `MPI_Send` | MPI function name, constant, or type |
| `UCX_TLS` | Shell environment variable |
| `mpicc` | Command-line tool |
| *rank* | Emphasized term on first introduction |
| `/* comment */` | Explanatory annotation in code |

---

*MPI 5.0 Programming Guide — C/C++*
*Based on the MPI 4.0 standard (2021) and MPI 5.0 standard (2025)*
