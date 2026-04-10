# Chapter 20: Advanced MPI-IO

## 20.1 Shared File Pointers

In addition to individual file pointers (per-process) and explicit offsets,
MPI-IO provides a **shared file pointer** — a single pointer advanced collectively
by all processes. This models a parallel sequential write where each process appends
its contribution in rank order.

### Ordered Write with Shared Pointer

```c
/* Each rank writes in turn using the shared pointer */
MPI_File_write_ordered(fh, local_data, local_n, MPI_DOUBLE, MPI_STATUS_IGNORE);
```

`MPI_File_write_ordered` is collective and serializes writes in rank order.
After the call, each rank has written its data sequentially after the previous
rank's data. This is equivalent to having rank 0 write first, then rank 1, etc.,
but MPI handles the coordination.

Use case: writing variable-length records where each process writes a different
number of elements and you want them concatenated in rank order.

### Query Shared Pointer Position

```c
MPI_Offset shared_pos;
MPI_File_get_position_shared(fh, &shared_pos);
```

### Non-Collective Shared Pointer Operations

For truly sequential, non-collective use of the shared pointer:

```c
MPI_File_write_shared(fh, buf, count, MPI_DOUBLE, &status);
MPI_File_read_shared(fh, buf, count, MPI_DOUBLE, &status);
```

These advance the shared pointer as a side effect, but they are NOT collectively
ordered — the order in which processes write is non-deterministic. Only use these
when process ordering does not matter.

---

## 20.2 Seeking

MPI-IO provides three independent file pointers: individual pointer, shared pointer,
and the displacement from `MPI_File_set_view`. Seeking manipulates the individual
or shared pointer.

```c
/* Seek the individual file pointer */
int MPI_File_seek(MPI_File fh, MPI_Offset offset, int whence);

/* whence: MPI_SEEK_SET (from start), MPI_SEEK_CUR (from current), MPI_SEEK_END */

/* Seek the shared file pointer — NOTE: this is a COLLECTIVE operation;
   all processes that opened the file must call it with the same arguments */
int MPI_File_seek_shared(MPI_File fh, MPI_Offset offset, int whence);
```

```c
/* After writing, seek back to verify */
MPI_File_seek(fh, 0, MPI_SEEK_SET);
MPI_File_read(fh, verify_buf, total_count, MPI_DOUBLE, MPI_STATUS_IGNORE);
```

The individual file pointer position can be queried with:
```c
MPI_Offset pos;
MPI_File_get_position(fh, &pos);
```

---

## 20.3 Non-Contiguous I/O with Derived Datatypes

The combination of `MPI_File_set_view` and derived datatypes is the most powerful
MPI-IO feature. It allows each process to read/write a non-contiguous pattern of
file positions without any manual offset calculation.

### 2D Ghost-Cell Array I/O

A common pattern: each rank holds a local array including ghost cells, but should
only write the interior cells to the file.

```c
int global_nx = 1024, global_ny = 1024;
int local_nx = 256, local_ny = 256;   /* without ghost cells */
int ghost = 1;                          /* one ghost cell layer */

/* Local array includes ghost cells: (local_nx+2) × (local_ny+2) */
int local_nx_g = local_nx + 2*ghost;
int local_ny_g = local_ny + 2*ghost;
double local_data[local_nx_g][local_ny_g];

/* Subarray for what THIS process contributes to the file */
int global_sizes[2] = {global_nx, global_ny};
int local_sizes[2]  = {local_nx, local_ny};
int starts[2]       = {coords[0]*local_nx, coords[1]*local_ny};

MPI_Datatype filetype;
MPI_Type_create_subarray(2, global_sizes, local_sizes, starts,
                          MPI_ORDER_C, MPI_DOUBLE, &filetype);
MPI_Type_commit(&filetype);

/* Subarray for the interior of the LOCAL array (skip ghost cells) */
int full_local_sizes[2] = {local_nx_g, local_ny_g};
int inner_sizes[2]      = {local_nx, local_ny};
int inner_starts[2]     = {ghost, ghost};

MPI_Datatype memtype;
MPI_Type_create_subarray(2, full_local_sizes, inner_sizes, inner_starts,
                          MPI_ORDER_C, MPI_DOUBLE, &memtype);
MPI_Type_commit(&memtype);

/* Set view to the file layout */
MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

/* Write interior of local array to correct file position */
MPI_File_write_all(fh, &local_data[0][0], 1, memtype, MPI_STATUS_IGNORE);

MPI_Type_free(&filetype);
MPI_Type_free(&memtype);
```

This writes only the interior data, skipping ghost cells, directly from the local
array layout without any temporary buffer copies.

---

## 20.4 Data Representations

The `datarep` argument to `MPI_File_set_view` controls byte representation:

| `datarep` string | Meaning |
|---|---|
| `"native"` | Store bytes as-is (no conversion); fastest; not portable across architectures |
| `"external32"` | Portable IEEE representation; same for all platforms; some overhead |
| `"internal"` | Implementation-defined; intermediate portability |

For production HPC codes on homogeneous clusters (all x86_64 Linux), use `"native"`.
For files shared between architectures or with external tools, use `"external32"`.

---

## 20.5 Error Handling for MPI-IO

MPI-IO uses the same error handler mechanism as communicators, but attached to file
handles:

```c
MPI_File_set_errhandler(fh, MPI_ERRORS_RETURN);

int rc = MPI_File_write_at_all(fh, offset, buf, count, MPI_DOUBLE, &status);
if (rc != MPI_SUCCESS) {
    char errstr[MPI_MAX_ERROR_STRING];
    int errlen;
    MPI_Error_string(rc, errstr, &errlen);
    fprintf(stderr, "I/O error: %s\n", errstr);
    MPI_Abort(MPI_COMM_WORLD, rc);
}

/* Check actual count written */
int written;
MPI_Get_count(&status, MPI_DOUBLE, &written);
if (written != count) {
    fprintf(stderr, "Short write: %d of %d\n", written, count);
}
```

---

## 20.6 Performance Tuning Patterns

### Pattern 1: Collective Buffering (Two-Phase I/O)

The most impactful optimization. A subset of processes (aggregators) buffer data
from all processes, then issue large sequential writes.

Enable with hints:
```c
MPI_Info_set(info, "romio_cb_write", "enable");
MPI_Info_set(info, "romio_cb_read",  "enable");
MPI_Info_set(info, "cb_nodes", "16");          /* 16 aggregators */
MPI_Info_set(info, "cb_buffer_size", "16777216"); /* 16 MB buffer per aggregator */
```

### Pattern 2: Lustre Striping

On Lustre filesystems, stripe the file across multiple OSTs before any process writes:

```bash
# Set striping before running the application
lfs setstripe -c 32 -S 4m /path/to/output/directory
```

Or set via MPI-IO hints at file creation:
```c
MPI_Info_set(info, "striping_factor", "32");
MPI_Info_set(info, "striping_unit", "4194304");
```

### Pattern 3: Avoid Small Random Writes

Aggregating small writes through `MPI_File_write_at_all` triggers two-phase I/O.
For very small writes from many processes (< 1 KB per process), consider having one
process gather all data and write serially — the coordination overhead of collective
I/O may exceed the benefit.

### Pattern 4: Preallocate Before Writing

```c
MPI_Offset expected_size = (MPI_Offset)global_n * sizeof(double);
MPI_File_preallocate(fh, expected_size);
```

Preallocating reserves contiguous space on the filesystem before any data is written.
On Lustre, this prevents OST-level metadata operations during the write phase and
ensures optimal stripe layout.

---

## 20.7 Checkpoint/Restart Pattern

A complete checkpoint/restart using MPI-IO:

```c
void write_checkpoint(MPI_Comm comm, const char *filename,
                      double *data, int local_n, int global_n, int rank)
{
    MPI_File fh;
    MPI_File_open(comm, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                  MPI_INFO_NULL, &fh);

    MPI_Offset offset = (MPI_Offset)rank * local_n * sizeof(double);
    MPI_File_write_at_all(fh, offset, data, local_n, MPI_DOUBLE,
                          MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

void read_checkpoint(MPI_Comm comm, const char *filename,
                     double *data, int local_n, int rank)
{
    MPI_File fh;
    MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_Offset offset = (MPI_Offset)rank * local_n * sizeof(double);
    MPI_File_read_at_all(fh, offset, data, local_n, MPI_DOUBLE,
                         MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}
```

For production checkpointing, consider also writing a small metadata file (on rank 0
only) containing the number of processes, global array dimensions, and timestep —
read this before the parallel data to validate that the checkpoint is compatible
with the current run configuration.

---

## Summary

| Function | Purpose |
|---|---|
| `MPI_File_write_ordered` | Collective sequential write in rank order |
| `MPI_File_seek` | Set individual file pointer |
| `MPI_File_seek_shared` | Set shared file pointer |
| `MPI_File_get_position` | Query individual pointer |
| Non-contiguous pattern | `set_view` + memory subarray type + `write_all` |
| `"native"` datarep | Fastest; no conversion; same-architecture only |
| `"external32"` datarep | Portable; cross-architecture |

**Key performance levers**:
- Collective I/O (`_all` suffix) — most important
- Lustre striping hints
- `MPI_File_preallocate` before large writes
- Two-phase I/O via `romio_cb_write` hints
