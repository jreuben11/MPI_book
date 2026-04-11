# Chapter 4: Error Handling

## 4.1 The Default Behavior

Out of the box, MPI aborts on any error. Every function returns an integer error code,
but the default error handler (`MPI_ERRORS_ARE_FATAL`) terminates all processes before
your code can inspect the return value.

```c
/* This will abort on error — you never see the return value */
MPI_Send(buf, count, MPI_INT, dest, tag, comm);
```

This default is intentional for application code: HPC jobs running on thousands of
nodes should not silently continue after a communication failure. The "fail-stop"
model keeps programs correct at the cost of availability.

For library writers and programs that want to attempt recovery, you must explicitly
switch the error handler.

---

## 4.2 Error Codes and Error Classes

MPI defines **error codes** (implementation-specific integers) and **error classes**
(portable categories):

```c
/* Get a human-readable error string from a code */
char errstr[MPI_MAX_ERROR_STRING];
int errlen;
MPI_Error_string(errorcode, errstr, &errlen);
printf("MPI error: %s\n", errstr);

/* Get the portable class from an implementation-specific code */
int errclass;
MPI_Error_class(errorcode, &errclass);
```

Standard error classes (selected):

| Constant | Meaning |
|---|---|
| `MPI_SUCCESS` | No error |
| `MPI_ERR_COMM` | Invalid communicator |
| `MPI_ERR_COUNT` | Invalid count argument |
| `MPI_ERR_TYPE` | Invalid datatype |
| `MPI_ERR_TAG` | Invalid tag |
| `MPI_ERR_RANK` | Invalid rank |
| `MPI_ERR_BUFFER` | Invalid buffer pointer |
| `MPI_ERR_ROOT` | Invalid root rank |
| `MPI_ERR_OP` | Invalid operation |
| `MPI_ERR_TRUNCATE` | Receive buffer too small |
| `MPI_ERR_IN_STATUS` | Error in `MPI_Waitall`/`MPI_Testall` — check individual statuses |
| `MPI_ERR_PENDING` | Request not yet complete |
| `MPI_ERR_OTHER` | Other error (see string for details) |

`MPI_SUCCESS` is always 0. All other error codes are positive integers.

---

## 4.3 Error Handlers

An **error handler** is a callback invoked when an MPI function encounters an error.
Error handlers are attached to communicators, windows, and files.

### Predefined Error Handlers

| Handler | Behavior |
|---|---|
| `MPI_ERRORS_ARE_FATAL` | Default: terminate all processes immediately on any error |
| `MPI_ERRORS_ABORT` | MPI 4.0: terminate processes in the communicator scope; distinct from `MPI_ERRORS_ARE_FATAL` |
| `MPI_ERRORS_RETURN` | Return the error code to the caller; do not abort |

Note: `MPI_ERRORS_ARE_FATAL` is the default and has been since MPI 1.0 — it aborts all
connected processes. `MPI_ERRORS_ABORT` is a separate handler added in MPI 4.0 that
aborts only the processes in the scope of the communicator/window/file/session it is
set on, making it more targeted. They are not aliases of each other.

### Switching to Return-on-Error

```c
/* Switch the default communicator to return errors */
MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

int rc = MPI_Send(buf, count, MPI_INT, dest, tag, MPI_COMM_WORLD);
if (rc != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int errlen;
    MPI_Error_string(rc, errstr, &errlen);
    fprintf(stderr, "Rank %d: MPI_Send failed: %s\n", rank, errstr);
    MPI_Abort(MPI_COMM_WORLD, rc);
}
```

Error handlers are communicator-scoped. Changing `MPI_COMM_WORLD`'s handler does not
affect windows or files, and does not affect communicators created by libraries.

```c
/* Set error handler for an MPI-IO file */
MPI_File_set_errhandler(fh, MPI_ERRORS_RETURN);

/* Set error handler for an RMA window */
MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);
```

### Custom Error Handlers

```c
void my_error_handler(MPI_Comm *comm, int *errcode, ...)
{
    char errstr[MPI_MAX_ERROR_STRING];
    int errlen;
    MPI_Error_string(*errcode, errstr, &errlen);

    int rank;
    MPI_Comm_rank(*comm, &rank);
    fprintf(stderr, "Rank %d: MPI error %d: %s\n", rank, *errcode, errstr);

    /* Log, flush I/O, then abort */
    fflush(stderr);
    MPI_Abort(*comm, *errcode);
}

/* Register the custom handler */
MPI_Errhandler handler;
MPI_Comm_create_errhandler(my_error_handler, &handler);
MPI_Comm_set_errhandler(MPI_COMM_WORLD, handler);
MPI_Errhandler_free(&handler);  /* Can free after attaching */
```

The `...` variadic parameter in the handler signature is required by the standard but
the content is implementation-defined. Do not read it.

---

## 4.4 MPI_Abort

```c
int MPI_Abort(MPI_Comm comm, int errorcode);
```

`MPI_Abort` terminates all processes in the job (not just those in `comm` — the
`comm` argument is advisory in most implementations). The `errorcode` is passed back
to the job scheduler as the process exit code.

Use `MPI_Abort` for unrecoverable errors only:

```c
if (rank == 0 && fopen("input.dat", "r") == NULL) {
    fprintf(stderr, "Cannot open input file\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}
```

`MPI_Abort` does not call `MPI_Finalize`. It is not a clean shutdown — it is a
controlled crash. Do not rely on destructors or `atexit` handlers running after it.

---

## 4.5 Defensive Error Checking Pattern

For development and debugging, wrap every MPI call with error checking. A macro
makes this practical:

```c
#define MPI_CHECK(call)                                              \
    do {                                                             \
        int _rc = (call);                                            \
        if (_rc != MPI_SUCCESS) {                                    \
            char _errstr[MPI_MAX_ERROR_STRING];                      \
            int  _errlen;                                            \
            MPI_Error_string(_rc, _errstr, &_errlen);                \
            fprintf(stderr, "%s:%d: MPI error: %s\n",               \
                    __FILE__, __LINE__, _errstr);                    \
            MPI_Abort(MPI_COMM_WORLD, _rc);                         \
        }                                                            \
    } while (0)
```

Usage:

```c
MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

MPI_CHECK(MPI_Send(buf, count, MPI_INT, dest, tag, comm));
MPI_CHECK(MPI_Recv(buf, count, MPI_INT, src,  tag, comm, MPI_STATUS_IGNORE));
```

In production builds, you can strip the checks with a compile flag if profiling shows
the overhead matters — but usually it does not.

The equivalent in C++:

```cpp
inline void mpi_check(int rc, const char *file, int line)
{
    if (rc != MPI_SUCCESS) {
        char errstr[MPI_MAX_ERROR_STRING];
        int errlen;
        MPI_Error_string(rc, errstr, &errlen);
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            ": MPI error: " + errstr);
    }
}
#define MPI_CHECK(call) mpi_check((call), __FILE__, __LINE__)
```

In C++, throwing from an error handler or check macro is acceptable, but remember that
MPI is still in an inconsistent state if the error was a real communication failure.
Catch the exception at the top level, finalize or abort cleanly.

---

## 4.6 MPI_Waitall Error Handling

When `MPI_Waitall` detects an error in any request, it returns `MPI_ERR_IN_STATUS`
and sets the `MPI_ERROR` field in each individual status:

```c
MPI_Status statuses[N];
int rc = MPI_Waitall(N, requests, statuses);
if (rc == MPI_ERR_IN_STATUS) {
    for (int i = 0; i < N; i++) {
        if (statuses[i].MPI_ERROR != MPI_SUCCESS) {
            char errstr[MPI_MAX_ERROR_STRING];
            int errlen;
            MPI_Error_string(statuses[i].MPI_ERROR, errstr, &errlen);
            fprintf(stderr, "Request %d failed: %s\n", i, errstr);
        }
    }
    MPI_Abort(MPI_COMM_WORLD, rc);
}
```

Never pass `MPI_STATUSES_IGNORE` if you need to diagnose which request failed.

---

## 4.7 Fault Tolerance and the MPIX_ Extension Namespace

### The MPIX_ Namespace

`MPIX_` is the reserved prefix for MPI extensions that have not yet been incorporated
into the standard. These are:

- **Experimental features** under active standardization debate
- **Implementation-specific extensions** adopted by multiple vendors as de facto standards
- **Post-deadline additions** to a ratified standard that will be merged in the next version

All `MPIX_` functions require `#include <mpi-ext.h>`. They may disappear, be renamed,
or gain an `MPI_` prefix in a future standard version. Code using them should guard
availability with compile-time checks.

---

### User Level Failure Mitigation (ULFM)

Standard MPI offered no mechanism to recover from process failures — `MPI_ERRORS_ARE_FATAL`
was the only real option. **ULFM** adds fault-tolerant operations; it remains in the
`MPIX_` extension namespace despite decades of development.

#### New Error Codes

ULFM introduces three new error codes:

| Code | Meaning |
|---|---|
| `MPIX_ERR_PROC_FAILED` | A remote process has failed; a P2P or collective operation cannot complete |
| `MPIX_ERR_PROC_FAILED_PENDING` | A process may have failed while a non-blocking operation was in flight |
| `MPIX_ERR_REVOKED` | The operation was issued on a communicator that has been revoked |

These codes are returned (not fatal-abort) only when `MPI_ERRORS_RETURN` is active.

#### ULFM Functions

```c
#include <mpi-ext.h>   /* required for all MPIX_ extensions */

/* Step 1: Revoke — broadcast "something went wrong" to all processes.
   All pending and future operations on comm return MPI_ERR_REVOKED.
   Non-collective: any process may call this unilaterally. */
int MPIX_Comm_revoke(MPI_Comm comm);

/* Step 2: Shrink — collective across all *surviving* processes.
   Creates a new communicator that excludes every process that has failed.
   Like MPI_Comm_create but failure-aware; survivors must all call this. */
int MPIX_Comm_shrink(MPI_Comm comm, MPI_Comm *newcomm);

/* Step 3: Acknowledge failures — clears the "unacknowledged failure" state
   on comm so that subsequent operations no longer return MPIX_ERR_PROC_FAILED.
   Must be called before reusing an old communicator after failure. */
int MPIX_Comm_failure_ack(MPI_Comm comm);

/* Step 4: Query acknowledged failures — returns the MPI_Group of all
   processes that have been acknowledged as failed via MPIX_Comm_failure_ack. */
int MPIX_Comm_failure_get_acked(MPI_Comm comm, MPI_Group *failedgrp);

/* Agreement — collective AND-reduction of flag across surviving processes,
   tolerant of concurrent failures. Used to agree on whether recovery succeeded.
   flag is ANDed: if any survivor passes 0, all see 0 on return. */
int MPIX_Comm_agree(MPI_Comm comm, int *flag);

/* Non-blocking agreement — initiates agreement, complete with MPI_Wait. */
int MPIX_Comm_iagree(MPI_Comm comm, int *flag, MPI_Request *request);
```

**Why `MPIX_Comm_shrink` matters**: it is the only standard-track way to create a
valid MPI communicator from a process set where some members have died. Unlike
`MPI_Comm_create`, which requires all processes in the old group to participate,
`MPIX_Comm_shrink` proceeds with whoever is still alive. It uses an internal
consensus protocol to determine exactly which ranks failed before building the new
communicator's rank mapping.

#### Typical ULFM Recovery Pattern

```c
#include <mpi.h>
#include <mpi-ext.h>

MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

/* --- application loop --- */
int rc = MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, world_comm);

if (rc == MPIX_ERR_PROC_FAILED || rc == MPIX_ERR_REVOKED) {

    /* Step 1: revoke so all survivors see a consistent error state */
    MPIX_Comm_revoke(world_comm);

    /* Step 2: build a communicator from survivors */
    MPI_Comm survivor_comm;
    MPIX_Comm_shrink(world_comm, &survivor_comm);

    /* Step 3: agree — did all survivors detect the failure? */
    int all_agreed = 1;
    MPIX_Comm_agree(survivor_comm, &all_agreed);

    if (all_agreed) {
        /* Step 4: free the old communicator; continue with survivors */
        MPI_Comm_free(&world_comm);
        world_comm = survivor_comm;
        /* redistribute data, reload checkpoint, continue ... */
    } else {
        MPI_Abort(survivor_comm, 1);
    }
}
```

Full fault tolerance requires checkpoint protocols, data redistribution across the
reduced process set, and replaying partial work. For most HPC codes, application-level
checkpoint/restart (Chapter 35) remains the practical strategy. ULFM enables automated
recovery without re-queuing the job, but requires implementation support (Open MPI 5.x
with `--with-ft=ulfm`; experimental in MPICH).

---

### GPU Support Queries

Several MPI implementations expose GPU-awareness queries under `MPIX_`:

```c
#include <mpi-ext.h>

/* Open MPI: returns 1 if the library was built with CUDA-aware support */
int cuda_ok = MPIX_Query_cuda_support();

/* Open MPI: returns 1 if built with ROCm/HIP-aware support */
int rocm_ok = MPIX_Query_rocm_support();

/* Open MPI: returns 1 if built with oneAPI Level Zero (Intel GPU) support */
int ze_ok = MPIX_Query_ze_support();
```

Use these at startup to gate code paths that pass device pointers directly to MPI.
Passing a GPU pointer to a non-GPU-aware MPI results in a segfault or silent
data corruption, not a catchable error.

```c
if (!MPIX_Query_cuda_support()) {
    fprintf(stderr, "MPI library is not CUDA-aware — cannot use device pointers\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}
/* safe to call MPI_Send with a cudaMalloc'd pointer */
```

These functions are Open MPI-specific; MPICH exposes the same information through
`MPIR_CVAR_CH4_OFI_ENABLE_GPU` environment inspection or compile-time defines.

---

### Other MPIX_ Extensions

| Extension | Description |
|---|---|
| `MPIX_Isendrecv` | Non-blocking `MPI_Sendrecv`; not yet standardized but provided by Open MPI and MPICH as an extension |
| `MPIX_Comm_get_failed` | Like `MPIX_Comm_failure_get_acked` but without requiring prior `ack`; returns the current set of failed processes |
| `MPIX_Grequest_start` / `MPIX_Grequest_class_*` | Extended generalized requests (MPICH extension) for integrating non-MPI async operations into `MPI_Wait` |
| `MPIX_Status_f082f` / `MPIX_Status_f082c` | Fortran 2008 status conversion helpers (rarely needed in C/C++) |

---

## 4.8 Error Handling in Libraries

If you are writing an MPI-based library:

1. **Save and restore the error handler** on any communicator you receive from the
   caller. Never permanently change the caller's error handler.

```c
MPI_Errhandler saved;
MPI_Comm_get_errhandler(user_comm, &saved);
MPI_Comm_set_errhandler(user_comm, MPI_ERRORS_RETURN);

/* ... library internals ... */

MPI_Comm_set_errhandler(user_comm, saved);
MPI_Errhandler_free(&saved);
```

2. **Prefer `MPI_Comm_dup`** so your library communicates on its own context, not the
   user's communicator. This also insulates your error handler from the user's.

3. **Document your error behavior**: does your library abort on error, return codes, or
   throw exceptions?

---

## Summary

| Topic | Key Points |
|---|---|
| Default handler | `MPI_ERRORS_ARE_FATAL` — terminates all processes on any error |
| `MPI_ERRORS_RETURN` | Switches to return-code mode; you must check return values |
| Error string | `MPI_Error_string(rc, str, &len)` gives a human-readable message |
| Error class | `MPI_Error_class(rc, &class)` gives a portable category |
| `MPI_Abort` | Emergency stop; not a clean shutdown; does not run destructors |
| `MPI_Waitall` errors | Returns `MPI_ERR_IN_STATUS`; check each `status.MPI_ERROR` |
| ULFM error codes | `MPIX_ERR_PROC_FAILED`, `MPIX_ERR_PROC_FAILED_PENDING`, `MPIX_ERR_REVOKED` |
| ULFM revoke | `MPIX_Comm_revoke` — unilateral; signals all pending ops to return `MPI_ERR_REVOKED` |
| ULFM shrink | `MPIX_Comm_shrink` — collective over survivors; builds a new communicator excluding failed ranks |
| ULFM ack/agree | `MPIX_Comm_failure_ack` / `MPIX_Comm_failure_get_acked` / `MPIX_Comm_agree` |
| GPU queries | `MPIX_Query_cuda_support()`, `MPIX_Query_rocm_support()`, `MPIX_Query_ze_support()` (Open MPI) |
| Library rule | Save/restore caller's error handler; use `MPI_Comm_dup` |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
