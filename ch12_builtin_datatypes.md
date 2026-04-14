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
| `MPI_CHAR` | `char` | 1 byte |
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

`MPI_CHAR` corresponds to `char` (printable character) and participates in type
checking. `MPI_SIGNED_CHAR` is the separate type for `signed char` as an integral
value. Prefer `MPI_BYTE` when you are treating the buffer as raw memory.

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

## 12.4 Large Count Support — MPI 3.0/4.0

Standard MPI uses `int` for element counts. An `int` can hold at most 2,147,483,647
(~2 billion). For large-memory machines communicating multi-billion-element arrays,
this is insufficient.

MPI 3.0 introduced `MPI_Count` — a 64-bit signed integer. MPI 4.0 added `_c` suffix
variants of all communication functions that accept `MPI_Count` instead of `int`:

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

### MPI_Aint — Address Integer

`MPI_Aint` is a **signed integer type** guaranteed to be wide enough to hold any
byte address or byte displacement on the current platform. On 64-bit systems it is
8 bytes; on 32-bit systems it is 4 bytes. It is analogous to `ptrdiff_t` or
`intptr_t` from `<stdint.h>`, but defined by MPI for use in all datatype and RMA APIs.



Note: 
- `ptrdiff_t`  - defined in `<stddef.h>` or `<cstddef>` for the result of subtracting two pointers. It is used for pointer arithmetic and array indexing when negative values are possible.

- `intptr_t` - defined in `<stdint.h>` or `<cstdint>` as a signed integer type for holding a `void*` pointer. Used to convert a pointer to an integer (and back) without losing data, allowing you to store pointer addresses in integer variables.


**Where it appears:**

- Displacement arrays in `MPI_Type_create_struct` and `MPI_Type_create_hindexed`
- Lower-bound and extent in `MPI_Type_get_extent` / `MPI_Type_create_resized`
- Displacement argument to `MPI_Get`, `MPI_Put`, `MPI_Accumulate` (RMA)
- Return value of `MPI_Get_address`

**Getting an address:**

```c
/* MPI_Get_address is the correct way to obtain an MPI_Aint address.
   Do NOT cast a pointer to MPI_Aint directly — use this function. */
double matrix[100][100];
MPI_Aint addr_00, addr_10;
MPI_Get_address(&matrix[0][0], &addr_00);
MPI_Get_address(&matrix[1][0], &addr_10);
MPI_Aint row_stride = addr_10 - addr_00;   /* bytes between rows */
```

**Arithmetic helpers (MPI 3.1+):**

Pointer arithmetic on `MPI_Aint` values must be done carefully on 32-bit platforms
where overflow can occur. MPI 3.1 added two safe arithmetic functions:

```c
/* MPI_Aint_add: returns base + disp, safe on 32-bit */
MPI_Aint displaced = MPI_Aint_add(base_addr, byte_offset);

/* MPI_Aint_diff: returns addr1 - addr2, result is a signed byte displacement */
MPI_Aint diff = MPI_Aint_diff(addr1, addr2);
```

Prefer these over plain `base_addr + offset` arithmetic — the standard arithmetic
will overflow on 32-bit if `base_addr` is near the top of the address space.

**Broadcasting addresses (RDMA / shared memory setup):**

```c
/* Share memory addresses across ranks for RMA window setup */
MPI_Aint base_addr;
MPI_Get_address(shared_buffer, &base_addr);
MPI_Bcast(&base_addr, 1, MPI_AINT, 0, comm);
```

**Common mistake — `int` displacement overflow:**

```c
/* WRONG: displacement computed as int, overflows beyond 2 GB */
int displacements[3];
displacements[0] = 0;
displacements[1] = sizeof(double) * 1000000000;  /* overflow! */

/* CORRECT: use MPI_Aint for byte displacements */
MPI_Aint displacements[3];
MPI_Get_address(&buf[0],          &displacements[0]);
MPI_Get_address(&buf[1000000000], &displacements[1]);
displacements[1] = MPI_Aint_diff(displacements[1], displacements[0]);
displacements[0] = 0;
```

---

### MPI_Offset — File Position

`MPI_Offset` is a **signed 64-bit integer** used exclusively for file byte positions
in MPI-IO. It is always 64 bits regardless of platform — unlike POSIX `off_t`, which
is 32 bits on 32-bit Linux unless `_FILE_OFFSET_BITS=64` is set.

`MPI_Offset` can address files up to 2^63 bytes (~9.2 EB), covering any realistic
parallel I/O scenario.

**Where it appears:**

- `MPI_File_seek` / `MPI_File_seek_shared` — position the file pointer
- `MPI_File_read_at` / `MPI_File_write_at` and their variants — explicit offset I/O
- `MPI_File_get_position` / `MPI_File_get_byte_offset` — query current position
- `MPI_File_set_view` — the `disp` (displacement) argument is `MPI_Offset`

```c
/* Write each rank's data at a calculated byte offset in a shared file */
MPI_File fh;
MPI_File_open(comm, "output.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY,
              MPI_INFO_NULL, &fh);

MPI_Offset my_offset = (MPI_Offset)rank * local_N * sizeof(double);
MPI_File_write_at(fh, my_offset, local_data, local_N, MPI_DOUBLE,
                  MPI_STATUS_IGNORE);

MPI_File_close(&fh);
```

**Why not use `long` or `size_t`?**

`long` is 4 bytes on Windows 64-bit. `size_t` is unsigned (can't represent
negative seek offsets from `MPI_SEEK_END`). `MPI_Offset` is always the right type
and the only one accepted by MPI-IO functions.

```c
/* MPI_File_seek with negative offset from end — requires signed type */
MPI_Offset footer_offset = -1024;
MPI_File_seek(fh, footer_offset, MPI_SEEK_END);
```

---

### MPI_Count — Large Element Count

`MPI_Count` is a **signed 64-bit integer** for element counts that exceed the
capacity of `int` (2^31 − 1 ≈ 2.1 billion). It was introduced in MPI 3.0
specifically for status queries, and MPI 4.0 extended it to all communication
functions via the `_c`-suffix API.

**Where it appears:**

- `MPI_Get_count_x` / `MPI_Get_elements_x` — query the count of a received message as `MPI_Count`
- `MPI_Type_size_x` — returns the size of a datatype as `MPI_Count`
- `MPI_Status_set_elements_x` — set element count in a generalized request status
- All `_c` suffix functions: `MPI_Send_c`, `MPI_Recv_c`, `MPI_Bcast_c`, `MPI_Allreduce_c`, etc.
- `MPI_Type_contiguous_c`, `MPI_Type_vector_c`, etc. — large-count datatype constructors

**Querying received count:**

```c
MPI_Status status;
MPI_Recv(buf, INT_MAX, MPI_DOUBLE, src, tag, comm, &status);

/* MPI_Get_count returns int — saturates to MPI_UNDEFINED if count > INT_MAX */
int count_int;
MPI_Get_count(&status, MPI_DOUBLE, &count_int);
if (count_int == MPI_UNDEFINED) { /* received more than INT_MAX elements */ }

/* MPI_Get_count_x always works regardless of size */
MPI_Count count_x;
MPI_Get_count_x(&status, MPI_DOUBLE, &count_x);
```

**Querying type size:**

```c
/* MPI_Type_size returns int — overflows for types larger than ~2 GB */
int sz_int;
MPI_Type_size(my_big_type, &sz_int);   /* may be wrong for very large derived types */

/* MPI_Type_size_x returns MPI_Count — always correct */
MPI_Count sz;
MPI_Type_size_x(my_big_type, &sz);
```

**Sending more than 2^31 elements:**

```c
MPI_Count big_count = 5000000000LL;   /* 5 × 10^9 doubles = 40 GB */
double *buf = malloc(big_count * sizeof(double));

/* _c variants accept MPI_Count */
MPI_Send_c(buf, big_count, MPI_DOUBLE, dest, tag, comm);
MPI_Recv_c(buf, big_count, MPI_DOUBLE, src,  tag, comm, MPI_STATUS_IGNORE);
MPI_Allreduce_c(MPI_IN_PLACE, buf, big_count, MPI_DOUBLE, MPI_SUM, comm);
```

**`MPI_UNDEFINED`**: the sentinel value (always −1) returned by count queries when
a count is not representable or not available. Check for it when calling the
non-`_x` count query functions on large messages.

---

### Summary of the Three Types

| Type | Width | Signed? | Domain | Arithmetic helpers |
|---|---|---|---|---|
| `MPI_Aint` | Platform (4 or 8 bytes) | Yes | Byte addresses and displacements | `MPI_Aint_add`, `MPI_Aint_diff` (MPI 3.1) |
| `MPI_Offset` | Always 64-bit | Yes | MPI-IO file byte positions | Plain `+`/`-` (always 64-bit, no overflow risk) |
| `MPI_Count` | Always 64-bit | Yes | Element counts | Plain `+`/`-` ; use `_c` API and `_x` query functions |

Each has a corresponding MPI predefined datatype for communication:

| C Type | MPI Datatype |
|---|---|
| `MPI_Aint` | `MPI_AINT` |
| `MPI_Offset` | `MPI_OFFSET` |
| `MPI_Count` | `MPI_COUNT` |

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
| `MPI_Aint` | Platform-wide signed address integer; use `MPI_Get_address` and `MPI_Aint_add`/`diff` |
| `MPI_Offset` | Always 64-bit signed; MPI-IO file byte positions; replaces `off_t` |
| `MPI_Count` / `_c` variants | Always 64-bit signed; MPI 3.0/4.0: enables >2^31 element messages; use `_x` queries |
| `MPI_AINT`, `MPI_OFFSET`, `MPI_COUNT` | Predefined datatypes for communicating these values |
| Avoid `MPI_Pack` | Use derived datatypes instead for structured data |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
