# Chapter 12: Built-in Datatypes

## 12.1 What MPI Datatypes Describe

An MPI datatype describes two things:
1. **What data to communicate**: the logical layout of a sequence of elements in memory.
2. **How much data**: combined with a `count`, determines total bytes transferred.

The separation of "count" and "datatype" allows MPI to describe non-contiguous memory
layouts without copying data — the MPI implementation handles the scatter/gather
internally.

This chapter covers built-in types. Chapter 13 covers derived types that compose
built-ins into complex layouts.

---

## 12.2 C Predefined Datatypes

| MPI Type | C Type | Typical Size |
|---|---|---|
| `MPI_CHAR` | `signed char` | 1 byte |
| `MPI_UNSIGNED_CHAR` | `unsigned char` | 1 byte |
| `MPI_BYTE` | (raw bytes, no type conversion) | 1 byte |
| `MPI_SHORT` | `signed short int` | 2 bytes |
| `MPI_UNSIGNED_SHORT` | `unsigned short int` | 2 bytes |
| `MPI_INT` | `signed int` | 4 bytes |
| `MPI_UNSIGNED` | `unsigned int` | 4 bytes |
| `MPI_LONG` | `signed long int` | 4 or 8 bytes (platform) |
| `MPI_UNSIGNED_LONG` | `unsigned long int` | 4 or 8 bytes |
| `MPI_LONG_LONG` | `signed long long int` | 8 bytes |
| `MPI_UNSIGNED_LONG_LONG` | `unsigned long long int` | 8 bytes |
| `MPI_FLOAT` | `float` | 4 bytes |
| `MPI_DOUBLE` | `double` | 8 bytes |
| `MPI_LONG_DOUBLE` | `long double` | 8–16 bytes (platform) |
| `MPI_WCHAR` | `wchar_t` | 2 or 4 bytes |
| `MPI_C_BOOL` | `_Bool` | 1 byte |
| `MPI_INT8_T` | `int8_t` | 1 byte |
| `MPI_INT16_T` | `int16_t` | 2 bytes |
| `MPI_INT32_T` | `int32_t` | 4 bytes |
| `MPI_INT64_T` | `int64_t` | 8 bytes |
| `MPI_UINT8_T` | `uint8_t` | 1 byte |
| `MPI_UINT16_T` | `uint16_t` | 2 bytes |
| `MPI_UINT32_T` | `uint32_t` | 4 bytes |
| `MPI_UINT64_T` | `uint64_t` | 8 bytes |
| `MPI_C_FLOAT_COMPLEX` | `float _Complex` | 8 bytes |
| `MPI_C_DOUBLE_COMPLEX` | `double _Complex` | 16 bytes |

### MPI_BYTE vs MPI_CHAR

`MPI_BYTE` transfers raw bytes with no type interpretation or conversion. Use it for:
- Serialized/packed data of mixed types
- When interoperability between different platforms with different char signedness matters
- Network-level byte streams

`MPI_CHAR` corresponds to `signed char` and participates in type checking. Prefer
`MPI_BYTE` when you are treating the buffer as raw memory.

### Fixed-Width Types (Preferred for Portability)

```c
/* Portable across all platforms — always use these for cross-node communication */
int32_t local_ids[100];
MPI_Send(local_ids, 100, MPI_INT32_T, dest, tag, comm);

/* Avoid: MPI_LONG is 4 bytes on Windows, 8 bytes on Linux 64-bit */
long bad_ids[100];
MPI_Send(bad_ids, 100, MPI_LONG, dest, tag, comm);  /* not portable */
```

---

## 12.3 Size and Extent Queries

```c
/* Size: number of bytes in one element (what gets transferred) */
MPI_Count size;
MPI_Type_size_x(MPI_DOUBLE, &size);   /* returns 8; MPI_Type_size_x takes MPI_Count* */

/* Extent: stride between repeated elements (includes padding) */
MPI_Aint lb, extent;
MPI_Type_get_extent(MPI_DOUBLE, &lb, &extent);  /* lb=0, extent=8 for MPI_DOUBLE */
```

For built-in types, size == extent. For derived types they can differ (Chapter 13).

`MPI_Aint` is a signed integer type guaranteed to be large enough to hold any memory
address. Use it for displacements and sizes in datatype construction.

---

## 12.4 Large Count Support — MPI 4.0

Standard MPI uses `int` for element counts. An `int` can hold at most 2,147,483,647
(~2 billion). For large-memory machines communicating multi-billion-element arrays,
this is insufficient.

MPI 4.0 introduced `MPI_Count` — a 64-bit signed integer — and `_c` suffix variants
of all communication functions that accept `MPI_Count` instead of `int`:

```c
/* Standard: limited to ~2 billion elements */
MPI_Send(buf, 2000000000, MPI_DOUBLE, dest, tag, comm);  /* near limit */

/* Large count: no limit beyond memory */
MPI_Count big_count = 5000000000LL;   /* 5 billion */
MPI_Send_c(buf, big_count, MPI_DOUBLE, dest, tag, comm);
```

All blocking and non-blocking point-to-point and collective functions have `_c` variants:

| Standard | Large Count |
|---|---|
| `MPI_Send` | `MPI_Send_c` |
| `MPI_Recv` | `MPI_Recv_c` |
| `MPI_Isend` | `MPI_Isend_c` |
| `MPI_Bcast` | `MPI_Bcast_c` |
| `MPI_Allreduce` | `MPI_Allreduce_c` |
| `MPI_Type_contiguous` | `MPI_Type_contiguous_c` |
| ... | ... |

```c
/* Large count Allreduce */
MPI_Count count = 10000000000LL;   /* 10 billion doubles = 80 GB */
double *buf = mmap(NULL, count * sizeof(double), ...);

MPI_Allreduce_c(MPI_IN_PLACE, buf, count, MPI_DOUBLE, MPI_SUM, comm);
```

### When to Use Large Count

- Single MPI messages exceeding 2^31 elements (extremely large).
- More commonly: datatype construction with large offsets or strides. Use
  `MPI_Type_size_x` instead of `MPI_Type_size` to get `MPI_Count` results.
- If you are not working with > 2 GB messages, you do not need `_c` variants.

---

## 12.5 Special Types

### MPI_Aint, MPI_Offset, MPI_Count as Datatypes

These address/offset types also have MPI datatype equivalents for communicating
pointer-sized or file-offset values:

| C Type | MPI Datatype |
|---|---|
| `MPI_Aint` | `MPI_AINT` |
| `MPI_Offset` | `MPI_OFFSET` |
| `MPI_Count` | `MPI_COUNT` |

```c
/* Share memory addresses across ranks (for shared memory or RDMA setup) */
MPI_Aint base_addr;
MPI_Get_address(shared_buffer, &base_addr);
MPI_Bcast(&base_addr, 1, MPI_AINT, 0, comm);
```

### MPI_PACKED

`MPI_PACKED` is used with `MPI_Pack` and `MPI_Unpack` for manual serialization.
This is a legacy approach; derived datatypes are almost always a better solution.

```c
/* Pack heterogeneous data into a byte buffer */
int pos = 0;
char packbuf[1024];

MPI_Pack(&my_int,    1, MPI_INT,    packbuf, 1024, &pos, comm);
MPI_Pack(&my_double, 1, MPI_DOUBLE, packbuf, 1024, &pos, comm);

MPI_Send(packbuf, pos, MPI_PACKED, dest, tag, comm);

/* On receiver */
int recv_int; double recv_double;
int pos2 = 0;
MPI_Recv(packbuf, 1024, MPI_PACKED, src, tag, comm, MPI_STATUS_IGNORE);
MPI_Unpack(packbuf, 1024, &pos2, &recv_int,    1, MPI_INT,    comm);
MPI_Unpack(packbuf, 1024, &pos2, &recv_double, 1, MPI_DOUBLE, comm);
```

Avoid `MPI_Pack`/`MPI_Unpack` in new code. Use `MPI_Type_create_struct` instead (Chapter 13).

---

## 12.6 Type Querying

```c
/* Get the name of a datatype (useful for debugging) */
char name[MPI_MAX_OBJECT_NAME];
int namelen;
MPI_Type_get_name(MPI_DOUBLE, name, &namelen);
printf("Type name: %s\n", name);   /* "MPI_DOUBLE" */

/* Set a name on a derived type */
MPI_Datatype my_type;
/* ... create type ... */
MPI_Type_set_name(my_type, "my_struct_type");
```

---

## Summary

| Topic | Key Points |
|---|---|
| Fixed-width types | Use `MPI_INT32_T`, `MPI_INT64_T` etc. for cross-platform portability |
| `MPI_BYTE` | Raw bytes; no type conversion; use for serialized data |
| `MPI_Type_size_x` | Returns `MPI_Count`; use instead of `MPI_Type_size` for large types |
| `MPI_Count` / `_c` variants | MPI 4.0: enables >2^31 element messages |
| `MPI_AINT`, `MPI_OFFSET` | Communicate pointer/file offset values |
| Avoid `MPI_Pack` | Use derived datatypes instead for structured data |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
