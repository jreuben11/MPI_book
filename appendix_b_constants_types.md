# Appendix B: Predefined Constants & Types

## B.1 Predefined Communicators

| Constant | Meaning |
|---|---|
| `MPI_COMM_WORLD` | All processes in the job |
| `MPI_COMM_SELF` | Communicator containing only the calling process |
| `MPI_COMM_NULL` | Null communicator handle |

---

## B.2 Predefined Datatypes (C)

### Integer Types

| MPI Type | C Type | Size |
|---|---|---|
| `MPI_CHAR` | `signed char` | 1 |
| `MPI_UNSIGNED_CHAR` | `unsigned char` | 1 |
| `MPI_BYTE` | (raw bytes) | 1 |
| `MPI_WCHAR` | `wchar_t` | 2 or 4 |
| `MPI_SHORT` | `signed short` | 2 |
| `MPI_UNSIGNED_SHORT` | `unsigned short` | 2 |
| `MPI_INT` | `signed int` | 4 |
| `MPI_UNSIGNED` | `unsigned int` | 4 |
| `MPI_LONG` | `signed long` | 4 or 8 (platform) |
| `MPI_UNSIGNED_LONG` | `unsigned long` | 4 or 8 |
| `MPI_LONG_LONG` or `MPI_LONG_LONG_INT` | `signed long long` | 8 |
| `MPI_UNSIGNED_LONG_LONG` | `unsigned long long` | 8 |
| `MPI_INT8_T` | `int8_t` | 1 |
| `MPI_INT16_T` | `int16_t` | 2 |
| `MPI_INT32_T` | `int32_t` | 4 |
| `MPI_INT64_T` | `int64_t` | 8 |
| `MPI_UINT8_T` | `uint8_t` | 1 |
| `MPI_UINT16_T` | `uint16_t` | 2 |
| `MPI_UINT32_T` | `uint32_t` | 4 |
| `MPI_UINT64_T` | `uint64_t` | 8 |
| `MPI_C_BOOL` | `_Bool` | 1 |
| `MPI_AINT` | `MPI_Aint` | pointer-sized |
| `MPI_COUNT` | `MPI_Count` | 8 |
| `MPI_OFFSET` | `MPI_Offset` | 8 |

### Floating-Point Types

| MPI Type | C Type | Size |
|---|---|---|
| `MPI_FLOAT` | `float` | 4 |
| `MPI_DOUBLE` | `double` | 8 |
| `MPI_LONG_DOUBLE` | `long double` | 8, 10, 12, or 16 (platform) |
| `MPI_C_FLOAT_COMPLEX` | `float _Complex` | 8 |
| `MPI_C_DOUBLE_COMPLEX` | `double _Complex` | 16 |
| `MPI_C_LONG_DOUBLE_COMPLEX` | `long double _Complex` | platform |

### MAXLOC / MINLOC Pair Types

| MPI Type | C Struct |
|---|---|
| `MPI_FLOAT_INT` | `{ float val; int rank; }` |
| `MPI_DOUBLE_INT` | `{ double val; int rank; }` |
| `MPI_LONG_INT` | `{ long val; int rank; }` |
| `MPI_2INT` | `{ int val; int rank; }` |
| `MPI_SHORT_INT` | `{ short val; int rank; }` |
| `MPI_LONG_DOUBLE_INT` | `{ long double val; int rank; }` |

### Special Types

| MPI Type | Use |
|---|---|
| `MPI_PACKED` | Buffer for `MPI_Pack` / `MPI_Unpack` |
| `MPI_DATATYPE_NULL` | Null datatype handle |

---

## B.3 Predefined Reduction Operations

| Operation | Meaning | Applicable Types |
|---|---|---|
| `MPI_SUM` | Sum | Integer, Float, Complex |
| `MPI_PROD` | Product | Integer, Float, Complex |
| `MPI_MAX` | Maximum | Integer, Float |
| `MPI_MIN` | Minimum | Integer, Float |
| `MPI_MAXLOC` | Maximum value and location | Pair types |
| `MPI_MINLOC` | Minimum value and location | Pair types |
| `MPI_LAND` | Logical AND | Integer |
| `MPI_LOR` | Logical OR | Integer |
| `MPI_LXOR` | Logical XOR | Integer |
| `MPI_BAND` | Bitwise AND | Integer, `MPI_BYTE` |
| `MPI_BOR` | Bitwise OR | Integer, `MPI_BYTE` |
| `MPI_BXOR` | Bitwise XOR | Integer, `MPI_BYTE` |
| `MPI_NO_OP` | No operation (identity) | Any (for `MPI_Get_accumulate` reads) |
| `MPI_REPLACE` | Replace with origin value | Any |
| `MPI_OP_NULL` | Null op handle | — |

---

## B.4 Special Values

### Wildcard Source and Tag

| Constant | Value | Use |
|---|---|---|
| `MPI_ANY_SOURCE` | Negative (impl-defined) | Accept message from any source |
| `MPI_ANY_TAG` | Negative (impl-defined) | Accept message with any tag |
| `MPI_PROC_NULL` | Negative | Null destination/source; absorbs sends, returns empty receives |
| `MPI_ROOT` | Negative | Used in inter-communicator collectives |

### Tag Limits

| Constant | Meaning |
|---|---|
| `MPI_TAG_UB` | Attribute key; value is the maximum tag (at least 32767) |

Query: `MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &flag)`

### Status

| Constant | Use |
|---|---|
| `MPI_STATUS_IGNORE` | Discard status in receive/wait |
| `MPI_STATUSES_IGNORE` | Discard status array in `Waitall`/`Testall` |

### String Limits

| Constant | Minimum value |
|---|---|
| `MPI_MAX_PROCESSOR_NAME` | 128 |
| `MPI_MAX_ERROR_STRING` | 256 |
| `MPI_MAX_LIBRARY_VERSION_STRING` | 8192 |
| `MPI_MAX_OBJECT_NAME` | 128 |
| `MPI_MAX_PORT_NAME` | 256 |
| `MPI_MAX_INFO_KEY` | 255 |
| `MPI_MAX_INFO_VAL` | 1023 |
| `MPI_MAX_DATAREP_STRING` | 128 |

---

## B.5 Error Classes

| Constant | Meaning |
|---|---|
| `MPI_SUCCESS` | No error (0) |
| `MPI_ERR_BUFFER` | Invalid buffer pointer |
| `MPI_ERR_COUNT` | Invalid count |
| `MPI_ERR_TYPE` | Invalid datatype |
| `MPI_ERR_TAG` | Invalid tag |
| `MPI_ERR_COMM` | Invalid communicator |
| `MPI_ERR_RANK` | Invalid rank |
| `MPI_ERR_REQUEST` | Invalid request |
| `MPI_ERR_ROOT` | Invalid root |
| `MPI_ERR_GROUP` | Invalid group |
| `MPI_ERR_OP` | Invalid operation |
| `MPI_ERR_TOPOLOGY` | Invalid topology |
| `MPI_ERR_DIMS` | Invalid dimension arguments |
| `MPI_ERR_ARG` | Invalid argument |
| `MPI_ERR_UNKNOWN` | Unknown error |
| `MPI_ERR_TRUNCATE` | Message truncated |
| `MPI_ERR_OTHER` | Other error |
| `MPI_ERR_INTERN` | Internal error |
| `MPI_ERR_IN_STATUS` | Look at individual status codes |
| `MPI_ERR_PENDING` | Request not yet complete |
| `MPI_ERR_IO` | I/O error |
| `MPI_ERR_FILE` | Invalid file handle |
| `MPI_ERR_NO_SPACE` | Not enough disk space |
| `MPI_ERR_NO_SUCH_FILE` | File does not exist |
| `MPI_ERR_FILE_EXISTS` | File already exists |
| `MPI_ERR_BAD_FILE` | Invalid file name |
| `MPI_ERR_WIN` | Invalid window |
| `MPI_ERR_BASE` | Invalid base address |
| `MPI_ERR_SIZE` | Invalid size |
| `MPI_ERR_DISP` | Invalid displacement |
| `MPI_ERR_LOCKTYPE` | Invalid lock type |
| `MPI_ERR_KEYVAL` | Invalid keyval |
| `MPI_ERR_INFO_KEY` | Invalid info key |
| `MPI_ERR_INFO_VALUE` | Invalid info value |
| `MPI_ERR_INFO_NOKEY` | Info key not found |
| `MPI_ERR_SESSION` | Invalid session (MPI 4.0) |
| `MPI_ERR_PROC_ABORTED` | Process aborted (ULFM) |
| `MPI_ERR_REVOKED` | Communicator revoked (ULFM) |
| `MPI_ERR_LASTCODE` | Upper bound of error codes |

---

## B.6 Error Handlers

| Constant | Behavior |
|---|---|
| `MPI_ERRORS_ABORT` | Terminate all processes (default) |
| `MPI_ERRORS_RETURN` | Return error code to caller |
| `MPI_ERRORS_ARE_FATAL` | Alias for `MPI_ERRORS_ABORT` |
| `MPI_ERRHANDLER_NULL` | Null errhandler handle |

---

## B.7 MPI-IO Constants

### Access Mode Flags (for `MPI_File_open`)

| Constant | Meaning |
|---|---|
| `MPI_MODE_RDONLY` | Read-only |
| `MPI_MODE_WRONLY` | Write-only |
| `MPI_MODE_RDWR` | Read/write |
| `MPI_MODE_CREATE` | Create if not exists |
| `MPI_MODE_EXCL` | Fail if exists |
| `MPI_MODE_DELETE_ON_CLOSE` | Delete when all handles closed |
| `MPI_MODE_UNIQUE_OPEN` | No other MPI opens |
| `MPI_MODE_SEQUENTIAL` | Sequential access |
| `MPI_MODE_APPEND` | Set initial position to end |

### File Seek Whence Values

| Constant | Meaning |
|---|---|
| `MPI_SEEK_SET` | From beginning |
| `MPI_SEEK_CUR` | From current position |
| `MPI_SEEK_END` | From end |

### Data Representation Strings

| String | Meaning |
|---|---|
| `"native"` | No conversion; platform-native bytes |
| `"external32"` | Portable IEEE big-endian |
| `"internal"` | Implementation-defined |

---

## B.8 RMA Constants

### Lock Types

| Constant | Meaning |
|---|---|
| `MPI_LOCK_EXCLUSIVE` | Only one holder; mutual exclusion |
| `MPI_LOCK_SHARED` | Multiple holders allowed |

### Window Assert Flags

| Constant | Use |
|---|---|
| `MPI_MODE_NOSTORE` | No local stores to window since last fence |
| `MPI_MODE_NOPUT` | No MPI_Put in this epoch |
| `MPI_MODE_NOPRECEDE` | No prior synchronization epoch |
| `MPI_MODE_NOSUCCEED` | No future synchronization epoch |
| `MPI_MODE_NOCHECK` | No conflicting locks exist |

### Window Memory Model

| Constant | Meaning |
|---|---|
| `MPI_WIN_UNIFIED` | Unified memory model (coherent) |
| `MPI_WIN_SEPARATE` | Separate memory model |
| `MPI_WIN_NULL` | Null window handle |

---

## B.9 Topology Constants

| Constant | Meaning |
|---|---|
| `MPI_CART` | Cartesian topology |
| `MPI_GRAPH` | Graph topology |
| `MPI_DIST_GRAPH` | Distributed graph topology |
| `MPI_UNDEFINED` | Undefined (use in `MPI_Comm_split` to exclude) |
| `MPI_COMM_TYPE_SHARED` | Shared memory split type |

---

## B.10 Thread Level Constants

| Constant | Value | Meaning |
|---|---|---|
| `MPI_THREAD_SINGLE` | 0 | Only one thread |
| `MPI_THREAD_FUNNELED` | 1 | Main thread calls MPI |
| `MPI_THREAD_SERIALIZED` | 2 | Serialized MPI calls |
| `MPI_THREAD_MULTIPLE` | 3 | Concurrent MPI calls |

---

## B.11 Datatype Array Order

| Constant | Meaning |
|---|---|
| `MPI_ORDER_C` | Row-major (last index varies fastest) |
| `MPI_ORDER_FORTRAN` | Column-major (first index varies fastest) |

---

## B.12 Distribution Types (for `MPI_Type_create_darray`)

| Constant | Meaning |
|---|---|
| `MPI_DISTRIBUTE_BLOCK` | Contiguous block distribution |
| `MPI_DISTRIBUTE_CYCLIC` | Cyclic (round-robin) distribution |
| `MPI_DISTRIBUTE_NONE` | No distribution along this dimension |
| `MPI_DISTRIBUTE_DFLT_DARG` | Default block size |
| `MPI_UNWEIGHTED` | No edge weights in graph topology |
| `MPI_WEIGHTS_EMPTY` | Empty weight array |

---

## B.13 Null Handles

| Type | Null Value |
|---|---|
| `MPI_Comm` | `MPI_COMM_NULL` |
| `MPI_Group` | `MPI_GROUP_NULL` |
| `MPI_Datatype` | `MPI_DATATYPE_NULL` |
| `MPI_Op` | `MPI_OP_NULL` |
| `MPI_Request` | `MPI_REQUEST_NULL` |
| `MPI_Win` | `MPI_WIN_NULL` |
| `MPI_File` | `MPI_FILE_NULL` |
| `MPI_Session` | `MPI_SESSION_NULL` |
| `MPI_Info` | `MPI_INFO_NULL` |
| `MPI_Errhandler` | `MPI_ERRHANDLER_NULL` |
| `MPI_Message` | `MPI_MESSAGE_NULL` |

---

## B.14 MPI_BOTTOM

`MPI_BOTTOM` is a special address constant that represents the bottom of the
process's address space. It is used when `target_disp` in RMA operations is an
absolute address (for `MPI_Win_create_dynamic` with `disp_unit = 1`).

```c
MPI_Put(origin, count, dtype, target, (MPI_Aint)remote_ptr, count, dtype, win);
/* Using MPI_BOTTOM as the window base implicitly */
```

In practice, use `MPI_Get_address` and relative displacements instead.

---

## B.15 Predefined Communicator Attributes

| Key | Type | Meaning |
|---|---|---|
| `MPI_TAG_UB` | `int *` | Maximum tag value |
| `MPI_HOST` | `int *` | Rank of host process (if any) |
| `MPI_IO` | `int *` | Rank with I/O capabilities |
| `MPI_WTIME_IS_GLOBAL` | `int *` | 1 if `MPI_Wtime` is globally synchronized |
| `MPI_APPNUM` | `int *` | Application number in `MPI_COMM_WORLD` |
| `MPI_UNIVERSE_SIZE` | `int *` | Size of MPI universe |

Query with `MPI_Comm_get_attr(MPI_COMM_WORLD, key, &val, &flag)`.

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
