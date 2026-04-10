# Chapter 4: Error Handling

## 4.1 The Default Behavior

Out of the box, MPI aborts on any error. Every function returns an integer error code,
but the default error handler (`MPI_ERRORS_ABORT`) terminates all processes before
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
| `MPI_ERRORS_ARE_FATAL` | Legacy name (pre-MPI 4.0): terminate all processes immediately (default) |
| `MPI_ERRORS_ABORT` | MPI 4.0+ preferred name; same abort-on-error behavior |
| `MPI_ERRORS_RETURN` | Return the error code to the caller; do not abort |

Note: `MPI_ERRORS_ARE_FATAL` is the name used in MPI 1.x–3.x implementations.
`MPI_ERRORS_ABORT` was introduced in MPI 4.0 as the new preferred name. Both refer
to the same behavior; prefer `MPI_ERRORS_ARE_FATAL` for portability with pre-4.0 systems.

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

## 4.7 Fault Tolerance — MPI 5.0

Standard MPI (before 5.0) offered no mechanism to recover from process failures.
`MPI_ERRORS_ABORT` was the only real option. MPI 4.0 introduced **User Level Failure
Mitigation (ULFM)** as an experimental annex; MPI 5.0 standardized it.

The three core ULFM operations:

```c
/* Revoke a communicator — all pending operations return MPI_ERR_REVOKED */
MPI_Comm_revoke(MPI_COMM_WORLD);

/* Create a new communicator excluding failed processes */
MPI_Comm survivor_comm;
MPI_Comm_shrink(MPI_COMM_WORLD, &survivor_comm);

/* Reach collective agreement on a value across surviving processes */
int failed = 1;
MPI_Comm_agree(survivor_comm, &failed);
```

Full fault tolerance requires rethinking your entire communication pattern — checkpoint
protocols, data redistribution across reduced process sets, and replaying work. This is
an advanced topic beyond the scope of this guide. The key takeaway for MPI 5.0 users:

1. The ULFM API is now standard — you can use it without implementation-specific flags.
2. For most HPC codes, checkpoint/restart at the application level remains the
   practical fault tolerance strategy. ULFM enables automated restart without
   re-spawning the job.

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
| Default handler | `MPI_ERRORS_ABORT` — terminates all processes on any error |
| `MPI_ERRORS_RETURN` | Switches to return-code mode; you must check return values |
| Error string | `MPI_Error_string(rc, str, &len)` gives a human-readable message |
| Error class | `MPI_Error_class(rc, &class)` gives a portable category |
| `MPI_Abort` | Emergency stop; not a clean shutdown; does not run destructors |
| `MPI_Waitall` errors | Returns `MPI_ERR_IN_STATUS`; check each `status.MPI_ERROR` |
| ULFM (MPI 5.0) | `Revoke` / `Shrink` / `Agree` for fault-tolerant recovery |
| Library rule | Save/restore caller's error handler; use `MPI_Comm_dup` |
