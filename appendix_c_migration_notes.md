# Appendix C: MPI 4.0 → 5.0 Migration Notes

## C.1 MPI 3.x → 4.0 Changes

### New in MPI 4.0

| Feature | Summary | Chapter |
|---|---|---|
| Sessions model | `MPI_Session_init`, independent MPI lifecycles | 22 |
| Partitioned communication | `MPI_Psend_init`, `MPI_Precv_init`, per-partition readiness | 23 |
| Persistent collectives | `MPI_Allreduce_init` etc., `_init` suffix | 11 |
| Large count (`_c` variants) | `MPI_Send_c`, `MPI_Count`, `MPI_Type_size_x` | 12, 24 |
| Standardized `MPI_Info` keys | Portable hints for comm, win, file | 3, 24 |
| `MPI_Comm_dup_with_info` | Dup with hints | 14 |
| `MPI_Group_from_session_pset` | Group from named process set | 22 |
| `MPI_Comm_create_from_group` | Comm from group + tag string | 22 |

### Deprecated in MPI 4.0

These functions still exist in MPI 4.0 implementations but should not be used in
new code:

| Deprecated | Replacement |
|---|---|
| `MPI_Type_hindexed` | `MPI_Type_create_hindexed` |
| `MPI_Type_hvector` | `MPI_Type_create_hvector` |
| `MPI_Type_struct` | `MPI_Type_create_struct` |
| `MPI_Type_extent` | `MPI_Type_get_extent` |
| `MPI_Type_lb` | `MPI_Type_get_extent` (read lb field) |
| `MPI_Type_ub` | `MPI_Type_get_extent` (read extent field) |
| `MPI_Attr_get` | `MPI_Comm_get_attr` |
| `MPI_Attr_put` | `MPI_Comm_set_attr` |
| `MPI_Attr_delete` | `MPI_Comm_delete_attr` |
| `MPI_Keyval_create` | `MPI_Comm_create_keyval` |
| `MPI_Keyval_free` | `MPI_Comm_free_keyval` |
| `MPI_Errhandler_create` | `MPI_Comm_create_errhandler` |
| `MPI_Errhandler_get` | `MPI_Comm_get_errhandler` |
| `MPI_Errhandler_set` | `MPI_Comm_set_errhandler` |
| `MPI_Handler_function` typedef | `MPI_Comm_errhandler_function` |
| `MPI_Copy_function` typedef | `MPI_Comm_copy_attr_function` |
| `MPI_Delete_function` typedef | `MPI_Comm_delete_attr_function` |
| `MPI_NULL_COPY_FN` | `MPI_COMM_NULL_COPY_FN` |
| `MPI_NULL_DELETE_FN` | `MPI_COMM_NULL_DELETE_FN` |
| `MPI_DUP_FN` | `MPI_COMM_DUP_FN` |

### Removed in MPI 3.0

These were deprecated in MPI 2.x and removed in MPI 3.0:

| Removed | What it was |
|---|---|
| C++ bindings (`MPI::COMM_WORLD`, etc.) | Deprecated in 2.2; use C API from C++ |
| `MPI_LB` datatype | Lower-bound marker type |
| `MPI_UB` datatype | Upper-bound marker type |
| `MPI_COMBINER_HVECTOR_INTEGER` | Old combiner constant |
| `MPI_COMBINER_HINDEXED_INTEGER` | Old combiner constant |
| `MPI_COMBINER_STRUCT_INTEGER` | Old combiner constant |
| `MPI_Address` | Use `MPI_Get_address` |

---

## C.2 MPI 4.0 → 5.0 Changes

### New in MPI 5.0

| Feature | Summary |
|---|---|
| ULFM standardized | `MPI_Comm_revoke`, `MPI_Comm_shrink`, `MPI_Comm_agree` moved from informational annex to standard |
| Error class additions | `MPI_ERR_PROC_ABORTED`, `MPI_ERR_REVOKED` added |
| RMA model clarifications | Unified model is now the recommended default |
| `MPI_Win_flush_local` semantics | Clarified: local-only; no remote completion |
| Additional standardized `MPI_Info` keys | More keys standardized across communicators, files, windows |
| Deprecation cleanup | Several older aliases formally deprecated |

### Deprecated in MPI 5.0

| Deprecated | Replacement |
|---|---|
| Several obscure `MPI_COMBINER_*` constants | Use `MPI_Type_get_envelope` with updated constants |
| `MPI_Bsend_overhead` | (internal; applications should not reference it) |
| Certain Fortran-only constants in C headers | Cleaned up; use C-appropriate names |

---

## C.3 API Naming Conventions Summary

Understanding the naming conventions makes it easier to navigate unfamiliar functions:

### Prefix Patterns

| Pattern | Meaning |
|---|---|
| `MPI_` | Standard MPI function |
| `PMPI_` | Profiling layer equivalent |
| `MPI_T_` | Tools / profiling interface |
| `ompi_` | Open MPI internal (avoid using directly) |

### Suffix Patterns

| Suffix | Meaning |
|---|---|
| `_c` | Large count variant (MPI 4.0); uses `MPI_Count` instead of `int` |
| `_x` | Extended: returns `MPI_Aint` or `MPI_Count` (e.g., `MPI_Type_size_x`) |
| `_init` | Persistent version (collective or point-to-point) |
| `v` | Vector variant: variable counts/displacements per process |
| `w` | "Wide" variant: variable types per process (most general) |
| `I` prefix | Non-blocking (e.g., `MPI_Isend`, `MPI_Iallreduce`) |

### Object Operation Pattern

Most MPI objects follow this pattern:
```
MPI_Object_create / MPI_Object_init    — construct
MPI_Object_set_*                        — configure
MPI_Object_get_*                        — query
MPI_Object_free / MPI_Object_finalize   — destroy
```

---

## C.4 Common Porting Checklist

### Porting from C++ Bindings (MPI 2.x code)

```cpp
/* OLD (C++ bindings — removed in MPI 3.0) */
MPI::COMM_WORLD.Send(&data, 1, MPI::INT, dest, tag);
MPI::COMM_WORLD.Recv(&data, 1, MPI::INT, src, tag);

/* NEW (C API in C++) */
MPI_Send(&data, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
MPI_Recv(&data, 1, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

### Porting from Old Deprecated Functions

```c
/* OLD */
MPI_Aint extent;
MPI_Type_extent(my_type, &extent);
int lb_marker;
MPI_Type_lb(my_type, &lb_marker);

/* NEW */
MPI_Aint lb, extent;
MPI_Type_get_extent(my_type, &lb, &extent);
```

```c
/* OLD */
MPI_Type_struct(count, blocklens, displs, types, &new_type);

/* NEW */
MPI_Type_create_struct(count, blocklens, displs, types, &new_type);
```

```c
/* OLD */
MPI_Keyval_create(copy_fn, delete_fn, &keyval, extra);
MPI_Attr_put(comm, keyval, &data);
MPI_Attr_get(comm, keyval, &data, &flag);

/* NEW */
MPI_Comm_create_keyval(copy_fn, delete_fn, &keyval, extra);
MPI_Comm_set_attr(comm, keyval, &data);
MPI_Comm_get_attr(comm, keyval, &data, &flag);
```

### Checking for New Features at Compile Time

```c
#include <mpi.h>

/* Feature detection macros */
#if MPI_VERSION >= 4
    /* Sessions, partitioned comm, persistent collectives, _c variants */
    #define HAVE_MPI4 1
#endif

#if MPI_VERSION >= 5
    /* ULFM standardized, RMA clarifications */
    #define HAVE_MPI5 1
#endif

/* Write fallback code for older MPI versions */
#ifdef HAVE_MPI4
    MPI_Allreduce_init(&local, &global, 1, MPI_DOUBLE, MPI_SUM,
                       comm, MPI_INFO_NULL, &req);
    MPI_Start(&req);
    /* ... */
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    MPI_Request_free(&req);
#else
    /* Fallback: non-blocking */
    MPI_Iallreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm, &req);
    /* ... */
    MPI_Wait(&req, MPI_STATUS_IGNORE);
#endif
```

---

## C.5 MPI Standard Version History (Compact)

| Version | Year | Key Additions |
|---|---|---|
| 1.0 | 1994 | P2P, collectives, datatypes, groups, communicators |
| 1.1 | 1995 | Clarifications and errata |
| 1.2 | 1997 | More clarifications |
| 2.0 | 1997 | RMA, parallel I/O, dynamic processes, C++ bindings |
| 2.1 | 2008 | Merged 1.x+2.x documents; clarifications |
| 2.2 | 2009 | C++ deprecated; `MPI_Dist_graph_create`; bug fixes |
| 3.0 | 2012 | Non-blocking collectives; `MPI_T`; C++ removed; `MPI_Mprobe`/`MPI_Mrecv`; `MPI_Comm_idup` |
| 3.1 | 2015 | Shared memory windows improvements; clarifications |
| 4.0 | 2021 | Sessions; partitioned comm; persistent collectives; large count; deprecations |
| 5.0 | 2025 | ULFM standard; RMA clarifications; more deprecations; standardized info keys |

---

## C.6 Implementation Version Compatibility

As of early 2025:

| Implementation | MPI Standard Support |
|---|---|
| Open MPI 4.x | MPI 3.1 full; MPI 4.0 partial |
| Open MPI 5.x | MPI 4.0 full; MPI 5.0 partial |
| MPICH 3.x | MPI 3.1 full |
| MPICH 4.x | MPI 4.0 full; MPI 5.0 partial |
| Intel MPI 2021 | MPI 3.1 full; MPI 4.0 partial |
| Intel MPI 2024+ | MPI 4.0 full |
| Cray MPICH (HPE) | Tracks upstream MPICH |
| HPC-X (NVIDIA) | Open MPI based |

Always verify the specific features you rely on against your vendor's release notes.
`MPI_VERSION` and `MPI_SUBVERSION` macros give the standard the implementation claims
to implement; `MPI_Get_library_version` gives the actual library version string.
