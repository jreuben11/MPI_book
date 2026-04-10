# Chapter 22: The Sessions Model — MPI 4.0

## 22.1 The Problem with MPI_COMM_WORLD

The traditional MPI programming model has one fundamental design limitation: all
library and application code shares `MPI_COMM_WORLD`. This creates several problems:

1. **Library initialization conflict**: if two libraries both call `MPI_Init`, the
   second call returns an error (or is silently ignored). Libraries cannot safely
   call `MPI_Init` without knowing whether someone else already did.

2. **Process set assumptions**: `MPI_COMM_WORLD` implicitly includes all processes
   in the job. Libraries cannot safely assume which processes exist or how many.

3. **Finalization ordering**: `MPI_Finalize` must be called by all processes
   simultaneously. A library that calls `MPI_Finalize` early disrupts the application.

4. **Dynamic process sets**: as HPC moves toward dynamic/elastic jobs, the set of
   processes changes at runtime. `MPI_COMM_WORLD` is static — once created, it never
   changes.

The **Sessions model** (MPI 4.0) addresses all of these by letting each independent
component of a program have its own MPI lifecycle, separate from `MPI_COMM_WORLD`.

---

## 22.2 Sessions Overview

```
Traditional model:              Sessions model:
                                
MPI_Init                        MPI_Session_init (App)
   |                               |
   MPI_COMM_WORLD                  Session A
   [all code shares this]          ├── process sets
   |                               └── communicators
MPI_Finalize                    MPI_Session_finalize (App)

                                MPI_Session_init (Library)
                                   |
                                   Session B (independent)
                                   ├── process sets
                                   └── communicators
                                MPI_Session_finalize (Library)
```

Each session is independent. A library can initialize its own session without
knowing anything about the application's session. Both sessions can coexist.

---

## 22.3 Creating a Session

```c
int MPI_Session_init(MPI_Info info, MPI_Errhandler errhandler,
                     MPI_Session *session);
```

```c
MPI_Session session;
MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &session);
```

The `info` argument accepts hints about the session's requirements:

```c
MPI_Info info;
MPI_Info_create(&info);
MPI_Info_set(info, "thread_level", "MPI_THREAD_MULTIPLE");
MPI_Session_init(info, MPI_ERRORS_RETURN, &session);
MPI_Info_free(&info);
```

Sessions are **local** — the call returns immediately. No collective operation is
required to create a session. Different processes may create sessions at different
times.

---

## 22.4 Process Sets

Within a session, processes are organized into **process sets** identified by name.
The most important predefined process set is `"mpi://WORLD"`, which corresponds
to all processes in the job (equivalent to `MPI_COMM_WORLD`'s group).

### Querying Available Process Sets

```c
/* How many process sets are available? */
int npsets;
MPI_Session_get_num_psets(session, MPI_INFO_NULL, &npsets);

/* Get the name of each process set */
for (int i = 0; i < npsets; i++) {
    int namelen = 0;
    MPI_Session_get_nth_pset(session, MPI_INFO_NULL, i, &namelen, NULL);
    /* namelen now includes the null terminator — allocate exactly namelen bytes */
    char *name = malloc(namelen);
    MPI_Session_get_nth_pset(session, MPI_INFO_NULL, i, &namelen, name);
    printf("  Process set %d: %s\n", i, name);
    free(name);
}
```

### Creating a Group from a Process Set

```c
MPI_Group world_group;
/* MPI_Session_get_pset_info: 3-arg function — returns info about the named pset */
MPI_Info pset_info;
MPI_Session_get_pset_info(session, "mpi://WORLD", &pset_info);
MPI_Info_free(&pset_info);   /* free info when done */
MPI_Group_from_session_pset(session, "mpi://WORLD", &world_group);
```

---

## 22.5 Creating Communicators from Sessions

Once you have a group, create a communicator:

```c
MPI_Comm comm;
MPI_Comm_create_from_group(world_group, "my_app_tag", MPI_INFO_NULL,
                            MPI_ERRORS_RETURN, &comm);
MPI_Group_free(&world_group);
```

The `tag` argument (a string) ensures that communicator creation operations can be
matched correctly when multiple independent libraries create communicators
simultaneously from different sessions.

Now `comm` behaves like any normal MPI communicator:

```c
int rank, size;
MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &size);

MPI_Bcast(data, N, MPI_DOUBLE, 0, comm);

MPI_Comm_free(&comm);
```

---

## 22.6 Finalizing a Session

```c
int MPI_Session_finalize(MPI_Session *session);
```

Like `MPI_Session_init`, this is **local** — no collective required. A process can
finalize its session independently. After finalization, the session handle is set
to `MPI_SESSION_NULL`.

All communicators created from this session must be freed before the session is
finalized.

```c
MPI_Comm_free(&comm);
MPI_Session_finalize(&session);
```

---

## 22.7 Full Sessions Example

A complete program using only the Sessions model:

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    /* --- Sessions model: no MPI_Init --- */
    MPI_Session session;
    MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_ABORT, &session);

    /* Get the world group */
    MPI_Group world_group;
    MPI_Group_from_session_pset(session, "mpi://WORLD", &world_group);

    /* Create a communicator */
    MPI_Comm comm;
    MPI_Comm_create_from_group(world_group, "hello_sessions",
                                MPI_INFO_NULL, MPI_ERRORS_ABORT, &comm);
    MPI_Group_free(&world_group);

    /* Normal MPI usage */
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    printf("Hello from rank %d of %d (Sessions model)\n", rank, size);

    /* Allreduce works normally */
    double val = (double)rank;
    MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (rank == 0) printf("Sum of ranks: %.0f\n", val);

    /* Cleanup */
    MPI_Comm_free(&comm);
    MPI_Session_finalize(&session);
    /* --- No MPI_Finalize --- */
    return 0;
}
```

---

## 22.8 Library Isolation with Sessions

This is the primary motivation for Sessions. A library can have its own complete
MPI lifecycle:

```c
/* solver_library.c */

typedef struct SolverContext {
    MPI_Session session;
    MPI_Comm    comm;
} SolverContext;

SolverContext *solver_init(MPI_Comm user_comm)
{
    SolverContext *ctx = malloc(sizeof(SolverContext));

    /* Independent session — does not touch the user's MPI state */
    MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &ctx->session);

    /* Create group and communicator matching the user's communicator */
    MPI_Group user_group, lib_group;
    MPI_Comm_group(user_comm, &user_group);
    /* ... (get same set of processes from session) ... */

    MPI_Comm_create_from_group(lib_group, "solver_lib_v2",
                                MPI_INFO_NULL, MPI_ERRORS_RETURN,
                                &ctx->comm);
    MPI_Group_free(&user_group);
    return ctx;
}

void solver_finalize(SolverContext *ctx)
{
    MPI_Comm_free(&ctx->comm);
    MPI_Session_finalize(&ctx->session);
    free(ctx);
}
```

The library's session is completely independent. If the user's session or
`MPI_COMM_WORLD` fails, the library can detect and handle it. If the library
finalizes before the user does, no global state is affected.

---

## 22.9 Interoperability with MPI_COMM_WORLD

Sessions and the traditional `MPI_COMM_WORLD` model are interoperable. A program
can use both:

```c
/* Traditional initialization */
MPI_Init(&argc, &argv);

/* Also create a session for a library */
MPI_Session lib_session;
MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN, &lib_session);

/* Use MPI_COMM_WORLD for application code */
MPI_Bcast(data, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

/* Use session-based comm for library */
library_function(lib_comm, data);

/* Cleanup */
MPI_Session_finalize(&lib_session);
MPI_Finalize();
```

When `MPI_Init` is called, MPI implicitly creates an internal session and creates
`MPI_COMM_WORLD` from it. This session is finalized when `MPI_Finalize` is called.

---

## Summary

| Function | Notes |
|---|---|
| `MPI_Session_init` | Local (non-collective); creates independent MPI context |
| `MPI_Session_finalize` | Local; cleans up session |
| `MPI_Session_get_num_psets` | Query available process sets |
| `MPI_Session_get_nth_pset` | Get process set name by index |
| `MPI_Group_from_session_pset` | Get group from named process set |
| `MPI_Comm_create_from_group` | Create communicator from group + tag |

**Key benefits**:
- Libraries can initialize and finalize MPI independently
- No dependency on `MPI_COMM_WORLD`
- Enables dynamic process management and fault tolerance
- Safe for use in multiple independent libraries within one application
