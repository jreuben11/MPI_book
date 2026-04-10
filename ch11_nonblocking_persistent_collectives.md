# Chapter 11: Non-Blocking & Persistent Collectives

## 11.1 Motivation

Standard blocking collectives (`MPI_Bcast`, `MPI_Allreduce`, etc.) force every process
to wait until the collective is complete before proceeding. For large communicators or
slow networks, this can be a significant bottleneck.

Non-blocking collectives (MPI 3.0) allow computation to overlap with collective
communication — the same principle as non-blocking point-to-point from Chapter 6.

Persistent collectives (MPI 4.0) extend persistent requests (Chapter 8) to collectives,
reducing per-call overhead for repeated patterns.

---

## 11.2 Non-Blocking Collectives

Every blocking collective has a non-blocking counterpart prefixed with `I`:

| Blocking | Non-Blocking |
|---|---|
| `MPI_Bcast` | `MPI_Ibcast` |
| `MPI_Scatter` | `MPI_Iscatter` |
| `MPI_Gather` | `MPI_Igather` |
| `MPI_Allgather` | `MPI_Iallgather` |
| `MPI_Reduce` | `MPI_Ireduce` |
| `MPI_Allreduce` | `MPI_Iallreduce` |
| `MPI_Alltoall` | `MPI_Ialltoall` |
| `MPI_Barrier` | `MPI_Ibarrier` |
| `MPI_Scan` | `MPI_Iscan` |
| `MPI_Exscan` | `MPI_Iexscan` |
| `MPI_Scatterv` | `MPI_Iscatterv` |
| `MPI_Gatherv` | `MPI_Igatherv` |
| `MPI_Allgatherv` | `MPI_Iallgatherv` |
| `MPI_Reduce_scatter` | `MPI_Ireduce_scatter` |
| `MPI_Reduce_scatter_block` | `MPI_Ireduce_scatter_block` |

All follow the same pattern: the non-blocking version takes the same arguments as
the blocking version plus an `MPI_Request *` at the end, and returns immediately.

```c
int MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count,
                   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                   MPI_Request *request);
```

---

## 11.3 Usage Pattern

```c
double local_error, global_max_error;
MPI_Request coll_req;

/* Start the reduction while computing the next step */
MPI_Iallreduce(&local_error, &global_max_error, 1, MPI_DOUBLE,
               MPI_MAX, MPI_COMM_WORLD, &coll_req);

/* Overlap: compute next iteration's local_error (for step+2)
   using the PREVIOUS global_max_error */
compute_next_step(data);                   /* does not touch local_error */

/* Wait for the reduction from step+1 */
MPI_Wait(&coll_req, MPI_STATUS_IGNORE);

if (global_max_error < TOLERANCE) break;  /* convergence check */
```

This is a **pipelined convergence check**: the reduction for step `k+1` overlaps with
the computation for step `k+2`. The convergence decision lags by one step, which is
acceptable for most iterative solvers.

---

## 11.4 Non-Blocking Collective Rules

The rules are stricter than for blocking collectives:

1. **Non-blocking collectives on the same communicator must be matched consistently**
   across all processes: same sequence, same collective type, one at a time. The
   "active" period is from the `MPI_I*` call to the `MPI_Wait`. Starting a second
   non-blocking collective on the same communicator before the first completes is
   erroneous.

2. **Mixing blocking and non-blocking collectives on the same communicator is prohibited**
   while a non-blocking collective is active (MPI-3.1 §5.12.5). The blocking collective
   must use a *different* communicator to be correct — this is a hard requirement, not
   just a performance hint.

3. **Buffer rules are identical to non-blocking point-to-point**: do not touch
   `sendbuf` or `recvbuf` between the `MPI_I*` call and `MPI_Wait`.

4. `MPI_Ibarrier` is useful for **split-phase barrier**: post the barrier while
   finishing up work, then wait when you genuinely need synchronization.

```c
/* Split-phase barrier */
MPI_Request barrier_req;
MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
flush_output_buffers();          /* do cleanup work while barrier progresses */
MPI_Wait(&barrier_req, MPI_STATUS_IGNORE);
/* Now all processes have passed the barrier */
```

---

## 11.5 Persistent Collectives (MPI 4.0)

MPI 4.0 introduced persistent variants of all collective operations. They follow the
same init/start/wait/free lifecycle as persistent point-to-point (Chapter 8).

```c
/* Init: create the persistent collective request */
int MPI_Allreduce_init(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                       MPI_Info info, MPI_Request *request);
```

The `MPI_Info` argument allows implementation-specific hints (algorithm selection,
buffer placement). Pass `MPI_INFO_NULL` for default behavior.

All persistent collective initiators follow this naming: `MPI_<Name>_init`.

| Blocking | Non-Blocking | Persistent (MPI 4.0) |
|---|---|---|
| `MPI_Bcast` | `MPI_Ibcast` | `MPI_Bcast_init` |
| `MPI_Scatter` | `MPI_Iscatter` | `MPI_Scatter_init` |
| `MPI_Gather` | `MPI_Igather` | `MPI_Gather_init` |
| `MPI_Allgather` | `MPI_Iallgather` | `MPI_Allgather_init` |
| `MPI_Reduce` | `MPI_Ireduce` | `MPI_Reduce_init` |
| `MPI_Allreduce` | `MPI_Iallreduce` | `MPI_Allreduce_init` |
| `MPI_Alltoall` | `MPI_Ialltoall` | `MPI_Alltoall_init` |
| `MPI_Barrier` | `MPI_Ibarrier` | `MPI_Barrier_init` |

---

## 11.6 Persistent Collective Example

```c
double local_val, global_result;
MPI_Request allred_req;

/* Initialize once outside the loop */
MPI_Allreduce_init(&local_val, &global_result, 1, MPI_DOUBLE,
                   MPI_SUM, MPI_COMM_WORLD, MPI_INFO_NULL, &allred_req);

for (int step = 0; step < NSTEPS; step++) {
    local_val = compute_local_contribution(step);

    MPI_Start(&allred_req);

    /* Overlap with computation that does not need global_result yet */
    setup_next_step(step);

    MPI_Wait(&allred_req, MPI_STATUS_IGNORE);
    /* allred_req is now INACTIVE, ready for next Start */

    use_global_result(global_result);
}

MPI_Request_free(&allred_req);
```

After `MPI_Wait`, the persistent collective request is **inactive** (not null) and
can be restarted with `MPI_Start` in the next iteration. This is identical to the
persistent point-to-point lifecycle.

---

## 11.7 When to Use Each Form

```
Blocking collective      — default; simplest code; use unless profiling shows a bottleneck
Non-blocking collective  — overlap computation with one-off collective; varying patterns
Persistent collective    — repeated identical collective in a tight loop with MPI 4.0+
```

The performance difference between blocking and non-blocking collectives is only
realized if:
1. There is genuinely independent work to do during the collective.
2. The MPI implementation supports asynchronous progress for collectives.
3. The communication is large enough that the overlap savings exceed the overhead.

Measure before optimizing. Replacing all `MPI_Allreduce` calls with `MPI_Iallreduce`
adds code complexity; do it only where profiler data shows the collective dominates
wall time.

---

## 11.8 Checking MPI 4.0 Availability

```c
#include <mpi.h>

#if MPI_VERSION < 4
#error "Persistent collectives require MPI 4.0 or later"
#endif
```

Or at runtime:

```c
int ver, subver;
MPI_Get_version(&ver, &subver);
if (ver < 4) {
    fprintf(stderr, "Need MPI 4.0, got %d.%d\n", ver, subver);
    MPI_Abort(MPI_COMM_WORLD, 1);
}
```

As of 2025, Open MPI 5.x and MPICH 4.x both implement MPI 4.0 persistent collectives.

---

## Summary

| Form | Overhead | Overlap | Repeat efficiency |
|---|---|---|---|
| Blocking | Per call | None | — |
| Non-blocking (`MPI_I*`) | Per call + request | Yes | Same as blocking |
| Persistent (`MPI_*_init`) | Init once + start/wait | Yes | Lower per-iteration cost |

**Key rules for non-blocking collectives**:
- Same ordering requirement as blocking collectives on the same communicator
- No buffer touches between `MPI_I*` and `MPI_Wait`
- `MPI_Ibarrier` enables split-phase synchronization

**Key rules for persistent collectives (MPI 4.0)**:
- `_init` called once; `MPI_Start` + `MPI_Wait` per iteration
- After `MPI_Wait`, request is inactive (not null) — ready to restart
- Free with `MPI_Request_free` when done

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
