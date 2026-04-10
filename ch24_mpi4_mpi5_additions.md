# Chapter 24: Large Counts & MPI 4.0/5.0 Additions

## 24.1 Large Count Support (MPI 4.0)

### The int Limit Problem

Standard MPI uses `int` for element counts. A 32-bit signed integer holds at most
2,147,483,647 elements (~2 billion). For a `double` array, this is about 16 GB —
a limit easily reached on modern large-memory nodes.

```c
/* This silently overflows for N > 2^31 */
int N = 3000000000;   /* 3 billion — overflow! */
MPI_Send(buf, N, MPI_DOUBLE, dest, tag, comm);  /* undefined behavior */
```

### MPI_Count and _c Variants

MPI 4.0 introduced `MPI_Count` — a 64-bit signed integer — and `_c` suffix variants
of all communication functions:

```c
/* Large count send */
MPI_Count count = 5000000000LL;  /* 5 billion doubles = 40 GB */
MPI_Send_c(buf, count, MPI_DOUBLE, dest, tag, comm);

/* Large count non-blocking */
MPI_Request req;
MPI_Isend_c(buf, count, MPI_DOUBLE, dest, tag, comm, &req);
MPI_Wait(&req, MPI_STATUS_IGNORE);

/* Large count collective */
MPI_Allreduce_c(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, comm);
```

### Large Count Datatype Functions

```c
/* Size and extent with large count return types */
MPI_Count type_size;
MPI_Type_size_x(MPI_DOUBLE, &type_size);   /* returns MPI_Count, not int */

MPI_Aint lb;
MPI_Count extent;
MPI_Type_get_extent_x(my_type, &lb, &extent);

/* Large count type constructors */
MPI_Datatype new_type;
MPI_Type_contiguous_c(count, MPI_DOUBLE, &new_type);
MPI_Type_vector_c(count, block, stride, MPI_DOUBLE, &new_type);
```

### Large Count Receive Status

```c
MPI_Status status;
MPI_Recv_c(buf, max_count, MPI_DOUBLE, src, tag, comm, &status);

MPI_Count actual_count;
MPI_Get_count_c(&status, MPI_DOUBLE, &actual_count);  /* returns MPI_Count — note: _c suffix, not _x */
```

### When to Use Large Count

- Single MPI messages with > 2^31 elements — rare, but increasingly common as node
  memory grows to 1+ TB.
- More commonly needed in datatype construction: `MPI_Type_vector_c` with large
  `stride` or `count` parameters.
- Use `MPI_Type_size_x` instead of `MPI_Type_size` when derived types may be larger
  than 2 GB.

If you are not working with multi-gigabyte messages, the standard `int`-based API
remains appropriate and has no overhead compared to `_c` variants.

---

## 24.2 MPI_Comm_idup (MPI 3.0+)

Non-blocking communicator duplication. Allows the expensive `MPI_Comm_dup` to
happen in the background while doing other setup work:

```c
MPI_Comm new_comm;
MPI_Request dup_req;
MPI_Comm_idup(MPI_COMM_WORLD, &new_comm, &dup_req);

/* Do initialization work while dup is in progress */
allocate_data_structures();
read_input_file();

/* Wait for dup to complete before using new_comm */
MPI_Wait(&dup_req, MPI_STATUS_IGNORE);
MPI_Barrier(new_comm);  /* now safe */
```

---

## 24.3 MPI_Info Updates (MPI 4.0/5.0)

### Standardized Info Keys

Prior to MPI 4.0, `MPI_Info` keys were implementation-specific. MPI 4.0 and 5.0
standardized a set of keys that all implementations must recognize:

**Communicator creation**:
- `"mpi_assert_no_any_tag"`: no `MPI_ANY_TAG` wildcards will be used
- `"mpi_assert_no_any_source"`: no `MPI_ANY_SOURCE` wildcards will be used
- `"mpi_assert_exact_length"`: all messages have exact count (no truncation)
- `"mpi_assert_allow_overtaking"`: messages may overtake each other

```c
MPI_Info info;
MPI_Info_create(&info);
MPI_Info_set(info, "mpi_assert_no_any_source", "true");
MPI_Comm new_comm;
MPI_Comm_dup_with_info(MPI_COMM_WORLD, info, &new_comm);
MPI_Info_free(&info);
```

**Collective tuning**:
- `"mpi_assert_no_collective"`: no collective operations will be called on this communicator

**Window creation**:
- `"accumulate_ordering"`: ordering guarantees for accumulate operations
- `"same_disp_unit"`: all processes use the same displacement unit
- `"same_size"`: all processes expose the same window size

---

## 24.4 MPI 4.0 Deprecations

MPI 4.0 formally deprecated several functions that had been discouraged for years.
They remain available in all current implementations but may be removed in future versions.

### Deprecated Point-to-Point Functions

| Deprecated | Replacement |
|---|---|
| `MPI_Type_hindexed` | `MPI_Type_create_hindexed` |
| `MPI_Type_hvector` | Prefer `MPI_Type_create_hvector` |
| `MPI_Type_struct` | Prefer `MPI_Type_create_struct` |
| `MPI_Type_extent` | Use `MPI_Type_get_extent` |
| `MPI_Type_lb` | Use `MPI_Type_get_extent` |
| `MPI_Type_ub` | Use `MPI_Type_get_extent` |
| `MPI_LB`, `MPI_UB` | Marker datatypes removed |

### Deprecated Utility Functions

| Deprecated | Replacement |
|---|---|
| `MPI_Attr_get` | Use `MPI_Comm_get_attr` |
| `MPI_Attr_put` | Use `MPI_Comm_set_attr` |
| `MPI_Attr_delete` | Use `MPI_Comm_delete_attr` |
| `MPI_Keyval_create` | Use `MPI_Comm_create_keyval` |
| `MPI_Keyval_free` | Use `MPI_Comm_free_keyval` |
| `MPI_Errhandler_create` | Use `MPI_Comm_create_errhandler` |
| `MPI_Errhandler_get` | Use `MPI_Comm_get_errhandler` |
| `MPI_Errhandler_set` | Use `MPI_Comm_set_errhandler` |

---

## 24.5 MPI 5.0 Highlights

### Standardized Fault Tolerance (ULFM)

The User Level Failure Mitigation (ULFM) API, previously in an informational annex,
is now part of the standard body in MPI 5.0:

```c
/* Revoke a communicator after detecting a failure */
MPI_Comm_revoke(comm);

/* Create a new communicator shrinking out failed processes */
MPI_Comm survivor_comm;
MPI_Comm_shrink(comm, &survivor_comm);

/* Collectively agree on whether failures occurred */
int flag = my_local_error_detected;
MPI_Comm_agree(survivor_comm, &flag);
/* flag == 1 if ANY process had an error */
```

See Chapter 4 for the error handling context; ULFM is the advanced fault tolerance
layer built on top.

### RMA Memory Model Clarifications

MPI 5.0 clarifies the unified memory model semantics. Key changes:
- The unified memory model is now the **default recommendation** for new
  implementations targeting systems with coherent interconnects.
- Load/store to window memory in the unified model is now explicitly defined to be
  visible across processes subject to the synchronization rules.
- `MPI_Win_flush_local` semantics are clarified: it makes local operations visible
  locally (buffer safety) but does NOT guarantee any visibility at the target.

### Additional Deprecations in MPI 5.0

MPI 5.0 continues the cleanup started in MPI 4.0:

- `MPI_COMBINER_NAMED` and some helper query functions simplified
- Several Fortran-specific constants removed from C headers
- `MPI_BOTTOM` usage clarified (still valid but documentation improved)

---

## 24.6 Checking Version at Compile and Runtime

```c
/* Compile-time check */
#if MPI_VERSION >= 4
    /* Use MPI 4.0 features */
    MPI_Allreduce_init(&local, &global, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD, MPI_INFO_NULL, &req);
#else
    /* Fallback to non-blocking */
    MPI_Iallreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM,
                   MPI_COMM_WORLD, &req);
#endif

/* Runtime check */
int ver, subver;
MPI_Get_version(&ver, &subver);
printf("MPI %d.%d\n", ver, subver);

/* Library version string */
char libver[MPI_MAX_LIBRARY_VERSION_STRING];
int libverlen;
MPI_Get_library_version(libver, &libverlen);
printf("%s\n", libver);
```

---

## 24.7 Feature Detection Summary

| Feature | Added in | How to detect |
|---|---|---|
| Non-blocking collectives | MPI 3.0 | `MPI_VERSION >= 3` |
| `MPI_T` interface | MPI 3.0 | `MPI_VERSION >= 3` |
| `MPI_Comm_idup` | MPI 3.0 | `MPI_VERSION >= 3` |
| Persistent collectives | MPI 4.0 | `MPI_VERSION >= 4` |
| Sessions model | MPI 4.0 | `MPI_VERSION >= 4` |
| Partitioned communication | MPI 4.0 | `MPI_VERSION >= 4` |
| Large count (`_c` suffix) | MPI 4.0 | `MPI_VERSION >= 4` |
| ULFM fault tolerance | MPI 5.0 | `MPI_VERSION >= 5` |

---

## Summary

| Addition | Version | Impact |
|---|---|---|
| `MPI_Count` + `_c` variants | 4.0 | > 2B element messages |
| `MPI_Type_size_x` | 4.0 | Large derived type sizes |
| `MPI_Get_count_x` | 4.0 | Large receive count query |
| `MPI_Comm_idup` | 3.0 | Async communicator duplication |
| Standardized info keys | 4.0/5.0 | Portable performance hints |
| ULFM standard | 5.0 | Fault-tolerant programming |
| RMA unified model | 5.0 | Clearer semantics for coherent hardware |
