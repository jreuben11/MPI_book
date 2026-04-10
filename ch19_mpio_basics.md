# Chapter 19: MPI-IO Basics

## 19.1 Why Parallel I/O?

In HPC, output data is often as large as the computation itself — checkpoint files,
scientific datasets, visualization dumps. Writing this data from a single process
is a bottleneck. MPI-IO (defined in MPI 2.0) provides a portable API for
coordinated parallel I/O, allowing all processes to simultaneously access the same
file.

Benefits:
- **Single shared file**: one file per simulation step, not one per process.
- **Portability**: same API on Lustre, GPFS, PVFS, BeeGFS, and local filesystems.
- **Collective I/O**: allows the implementation to optimize access patterns for the
  underlying parallel filesystem.

---

## 19.2 Opening and Closing Files

```c
int MPI_File_open(MPI_Comm comm, const char *filename, int amode,
                  MPI_Info info, MPI_File *fh);

int MPI_File_close(MPI_File *fh);
```

`amode` is a bitwise OR of access mode flags:

| Flag | Meaning |
|---|---|
| `MPI_MODE_RDONLY` | Read-only |
| `MPI_MODE_WRONLY` | Write-only |
| `MPI_MODE_RDWR` | Read/write |
| `MPI_MODE_CREATE` | Create file if not exists |
| `MPI_MODE_EXCL` | Fail if file already exists |
| `MPI_MODE_DELETE_ON_CLOSE` | Delete when all handles closed |
| `MPI_MODE_UNIQUE_OPEN` | No other MPI processes open this file |
| `MPI_MODE_SEQUENTIAL` | Sequential access (enables OS prefetch hints) |
| `MPI_MODE_APPEND` | Start at end of file |

```c
MPI_File fh;

/* Open for writing, create if not exists */
MPI_File_open(MPI_COMM_WORLD, "output.bin",
              MPI_MODE_WRONLY | MPI_MODE_CREATE,
              MPI_INFO_NULL, &fh);

/* ... write ... */

MPI_File_close(&fh);
```

`MPI_File_open` is collective — all processes in `comm` must call it. Passing
`MPI_COMM_SELF` opens the file from a single process.

---

## 19.3 File Positioning

There are three ways to specify file position in MPI-IO:

1. **Explicit offset** (`_at` functions): pass a byte offset directly.
2. **Individual file pointer**: each process maintains its own pointer.
3. **Shared file pointer**: all processes share one pointer (covered in Chapter 20).

For parallel I/O, **explicit offsets are the most reliable and portable** approach.
They avoid any state shared between processes and are easy to reason about.

---

## 19.4 Individual Read/Write at Offset

```c
int MPI_File_read_at(MPI_File fh, MPI_Offset offset,
                     void *buf, int count, MPI_Datatype datatype,
                     MPI_Status *status);

int MPI_File_write_at(MPI_File fh, MPI_Offset offset,
                      const void *buf, int count, MPI_Datatype datatype,
                      MPI_Status *status);
```

`MPI_Offset` is a 64-bit signed integer for file offsets. Each process computes
its own offset independently.

```c
/* Each rank writes its portion of a global array */
int local_n = N / size;                         /* elements per rank */
MPI_Offset offset = (MPI_Offset)rank * local_n * sizeof(double);

double *local_data = malloc(local_n * sizeof(double));
/* ... fill local_data ... */

MPI_File_write_at(fh, offset, local_data, local_n, MPI_DOUBLE, MPI_STATUS_IGNORE);
```

This is an **individual** operation — each process performs its I/O independently.
Individual operations work but do not allow the filesystem to optimize access patterns.

---

## 19.5 Collective Read/Write

Collective variants (`_all` suffix) allow MPI to aggregate small individual I/O
requests into larger, more efficient filesystem accesses:

```c
int MPI_File_read_at_all(MPI_File fh, MPI_Offset offset,
                          void *buf, int count, MPI_Datatype datatype,
                          MPI_Status *status);

int MPI_File_write_at_all(MPI_File fh, MPI_Offset offset,
                           const void *buf, int count, MPI_Datatype datatype,
                           MPI_Status *status);
```

All processes in the communicator that opened the file must call these collectively.
MPI may use a "two-phase I/O" strategy: a subset of processes (aggregators) collect
data from others, then issue large sequential writes to the filesystem.

```c
/* Collective write — preferred for parallel filesystems */
MPI_File_write_at_all(fh, offset, local_data, local_n, MPI_DOUBLE,
                      MPI_STATUS_IGNORE);
```

**Always prefer collective I/O** when all processes are writing simultaneously.
On parallel filesystems like Lustre, collective I/O can improve throughput by
orders of magnitude compared to individual I/O from thousands of processes.

---

## 19.6 File Views

A **file view** tells MPI how to interpret the file from each process's perspective.
Instead of computing offsets manually, you describe the access pattern using an
MPI datatype.

```c
int MPI_File_set_view(MPI_File fh, MPI_Offset disp,
                      MPI_Datatype etype, MPI_Datatype filetype,
                      const char *datarep, MPI_Info info);
```

| Argument | Meaning |
|---|---|
| `disp` | Byte offset to the start of the view |
| `etype` | Elementary datatype (unit of access, e.g., `MPI_DOUBLE`) |
| `filetype` | Pattern that repeats from `disp` (describes which bytes this process accesses) |
| `datarep` | `"native"` (no conversion), `"external32"` (portable), `"internal"` |

After setting a view, you can use `MPI_File_write_all` (without explicit offset)
and MPI handles the mapping to the correct file positions.

### Example: Distribute Rows of a 2D Array

```c
/* 1000×1000 global array, each rank owns 1000/P rows */
int global_rows = 1000, global_cols = 1000;
int local_rows = global_rows / size;

/* Subarray type: this rank's portion of the global array */
int global_sizes[2] = {global_rows, global_cols};
int local_sizes[2]  = {local_rows,  global_cols};
int starts[2]       = {rank * local_rows, 0};

MPI_Datatype filetype;
MPI_Type_create_subarray(2, global_sizes, local_sizes, starts,
                          MPI_ORDER_C, MPI_DOUBLE, &filetype);
MPI_Type_commit(&filetype);

/* Set view: disp=0 (start of file), etype=double, filetype=our subarray */
MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

/* Write local data sequentially (MPI maps it to the right file positions) */
double *local_data = malloc(local_rows * global_cols * sizeof(double));
/* ... fill ... */

MPI_File_write_all(fh, local_data, local_rows * global_cols, MPI_DOUBLE,
                   MPI_STATUS_IGNORE);

MPI_Type_free(&filetype);
free(local_data);
```

This is the canonical pattern for parallel array I/O: define the global array
layout, each rank's subarray, and let MPI handle the file mapping.

---

## 19.7 File Manipulation

```c
/* Get current file size */
MPI_Offset size;
MPI_File_get_size(fh, &size);

/* Truncate or extend the file */
MPI_File_set_size(fh, new_size);

/* Pre-allocate disk space (avoids fragmentation) */
MPI_File_preallocate(fh, expected_size);

/* Synchronize: ensure all writes are visible to other processes */
MPI_File_sync(fh);

/* Delete a file */
MPI_File_delete("filename.bin", MPI_INFO_NULL);
```

`MPI_File_preallocate` is important on parallel filesystems: call it before writing
to reserve contiguous space on the filesystem. Without it, file data may be
fragmented across storage targets, reducing read bandwidth later.

---

## 19.8 MPI-IO Info Hints

Performance-critical hints for parallel filesystems:

```c
MPI_Info info;
MPI_Info_create(&info);

/* Lustre: stripe across 16 OSTs with 4 MB stripe unit */
MPI_Info_set(info, "striping_factor", "16");
MPI_Info_set(info, "striping_unit", "4194304");

/* Enable collective buffering */
MPI_Info_set(info, "romio_cb_write", "enable");
MPI_Info_set(info, "romio_cb_read",  "enable");

/* Number of aggregator processes */
MPI_Info_set(info, "cb_nodes", "16");

MPI_File_open(MPI_COMM_WORLD, "output.bin",
              MPI_MODE_WRONLY | MPI_MODE_CREATE, info, &fh);

MPI_Info_free(&info);
```

Hints are advisory — the implementation may ignore any of them. Always verify with
`MPI_File_get_info` whether your hints were accepted.

---

## 19.9 Non-Blocking I/O

Non-blocking I/O operations (prefix `I`, suffix `_at`) allow overlap of I/O with
computation:

```c
MPI_Request io_req;

/* Start non-blocking write */
MPI_File_iwrite_at(fh, offset, buf, count, MPI_DOUBLE, &io_req);

/* Overlap computation with I/O */
compute_next_step();

/* Wait for I/O to complete */
MPI_Wait(&io_req, MPI_STATUS_IGNORE);
```

Non-blocking collective variants (MPI-3.1+): `MPI_File_iread_all`, `MPI_File_iwrite_all`
(no `_at` suffix; they use the current file view and individual file pointer, not an
explicit offset). Note: `MPI_File_iread_at_all` and `MPI_File_iwrite_at_all` do not
exist in the MPI standard.

---

## Summary

| Function | Purpose |
|---|---|
| `MPI_File_open` | Open file collectively |
| `MPI_File_close` | Close file collectively |
| `MPI_File_set_view` | Define this rank's view of the file |
| `MPI_File_write_at` | Individual write at explicit offset |
| `MPI_File_write_at_all` | Collective write at explicit offset |
| `MPI_File_write_all` | Collective write using current view |
| `MPI_File_preallocate` | Reserve disk space before writing |
| `MPI_File_sync` | Flush writes to storage |
| `MPI_File_iwrite_at` | Non-blocking write |

**Best practices**:
1. Use collective I/O (`_all` suffix) for parallel data
2. Use `MPI_File_set_view` with subarray types instead of manually computing offsets
3. Call `MPI_File_preallocate` before large writes
4. Set striping hints before `MPI_File_open`

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
