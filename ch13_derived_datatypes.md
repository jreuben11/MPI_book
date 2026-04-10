# Chapter 13: Derived Datatypes

## 13.1 Why Derived Datatypes?

Without derived datatypes, you must manually pack non-contiguous data into a temporary
buffer, send the buffer, then unpack at the receiver — three copies instead of one
in-place transfer.

Derived datatypes let you describe the memory layout to MPI once. MPI then handles
the scatter/gather internally, often using hardware scatter/gather DMA:

```c
/* Without derived datatype: manual packing */
double tmpbuf[N];
for (int i = 0; i < N; i++) tmpbuf[i] = matrix[i][col]; /* column extract */
MPI_Send(tmpbuf, N, MPI_DOUBLE, dest, tag, comm);

/* With derived datatype: MPI does the gather */
MPI_Datatype col_type;
MPI_Type_vector(N, 1, NCOLS, MPI_DOUBLE, &col_type);
MPI_Type_commit(&col_type);
MPI_Send(&matrix[0][col], 1, col_type, dest, tag, comm);
MPI_Type_free(&col_type);
```

All derived datatypes follow the same lifecycle:
1. **Create** with one of the `MPI_Type_*` functions.
2. **Commit** with `MPI_Type_commit` before use in communication.
3. **Use** in send/receive/collective calls.
4. **Free** with `MPI_Type_free` when done.

---

## 13.2 MPI_Type_contiguous

The simplest derived type: a sequence of elements of another type.

```c
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
```

```c
/* A type representing 3 doubles packed together */
MPI_Datatype vec3;
MPI_Type_contiguous(3, MPI_DOUBLE, &vec3);
MPI_Type_commit(&vec3);

struct { double x, y, z; } particles[1000];
MPI_Send(particles, 1000, vec3, dest, tag, comm);

MPI_Type_free(&vec3);
```

Useful for giving a name to a fixed-size group of elements, or for use as a building
block for more complex types.

---

## 13.3 MPI_Type_vector — Regular Strides

```c
int MPI_Type_vector(int count, int blocklength, int stride,
                    MPI_Datatype oldtype, MPI_Datatype *newtype);
```

Describes `count` blocks, each of `blocklength` elements, separated by `stride`
elements (stride measured in units of `oldtype`, including the block itself).

```
Memory layout (stride=4, blocklength=2):
[X X _ _ X X _ _ X X _ _]
 ^^^^^   ^^^^^   ^^^^^
 block0  block1  block2
```

```c
/* Send one column of a 2D C array (row-major storage) */
double matrix[NROWS][NCOLS];
/* Column j occupies positions: [0][j], [1][j], ..., [NROWS-1][j]
   Stride between elements = NCOLS (one full row) */

MPI_Datatype col_type;
MPI_Type_vector(NROWS,  /* count: number of blocks */
                1,       /* blocklength: 1 element per block */
                NCOLS,   /* stride: NCOLS doubles between starts */
                MPI_DOUBLE, &col_type);
MPI_Type_commit(&col_type);

MPI_Send(&matrix[0][j], 1, col_type, dest, tag, comm);
MPI_Type_free(&col_type);
```

`MPI_Type_hvector` is identical but takes `stride` in bytes (MPI_Aint) rather than
element counts — useful when the stride involves structure padding.

---

## 13.4 MPI_Type_indexed — Irregular Strides

```c
int MPI_Type_indexed(int count,
                     const int blocklengths[],
                     const int displacements[],
                     MPI_Datatype oldtype, MPI_Datatype *newtype);
```

Each block can have a different length and displacement (in elements of `oldtype`).

```c
/* Send a sparse vector: only elements at given indices */
int nnz = 5;
int indices[5] = {0, 3, 7, 12, 20};
int lengths[5] = {1, 1, 1,  1,  1};  /* one element each */

MPI_Datatype sparse_type;
MPI_Type_indexed(nnz, lengths, indices, MPI_DOUBLE, &sparse_type);
MPI_Type_commit(&sparse_type);

double vec[100];
MPI_Send(vec, 1, sparse_type, dest, tag, comm);
MPI_Type_free(&sparse_type);
```

`MPI_Type_hindexed` takes displacements in bytes (`MPI_Aint` array).

---

## 13.5 MPI_Type_struct — Heterogeneous Structures

The most general datatype constructor. Describes an arbitrary combination of types
at arbitrary byte offsets — the direct mapping to a C struct.

```c
int MPI_Type_create_struct(int count,
                            const int blocklengths[],
                            const MPI_Aint displacements[],
                            const MPI_Datatype types[],
                            MPI_Datatype *newtype);
```

**Always use `MPI_Get_address` to compute displacements** — never compute offsets
manually with `sizeof` and arithmetic, because compiler padding may not match your
assumptions across different compilers or platforms.

```c
typedef struct {
    int    id;
    double value;
    char   label[16];
} Record;

Record sample;  /* Use a concrete variable to get addresses */

MPI_Aint base, disp[3];
MPI_Get_address(&sample,        &base);
MPI_Get_address(&sample.id,     &disp[0]);
MPI_Get_address(&sample.value,  &disp[1]);
MPI_Get_address(&sample.label,  &disp[2]);

/* Make displacements relative to the start of the struct */
disp[0] -= base;
disp[1] -= base;
disp[2] -= base;

int blocklens[3] = {1, 1, 16};
MPI_Datatype types[3] = {MPI_INT, MPI_DOUBLE, MPI_CHAR};

MPI_Datatype record_type;
MPI_Type_create_struct(3, blocklens, disp, types, &record_type);
MPI_Type_commit(&record_type);

Record records[100];
MPI_Send(records, 100, record_type, dest, tag, comm);
MPI_Type_free(&record_type);
```

---

## 13.6 Extent and Resizing

When a derived type is used with `count > 1`, MPI places consecutive elements using
the type's **extent** as the stride:

```
Element 0 starts at: buf
Element 1 starts at: buf + 1 * extent
Element 2 starts at: buf + 2 * extent
```

For derived types, the extent may not match your intention. A column-type created with
`MPI_Type_vector` has an extent that extends from the first element to one stride past
the last block — not to the start of the next column.

Use `MPI_Type_create_resized` to explicitly set extent and lower bound:

```c
int MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent,
                             MPI_Datatype *newtype);
```

### Example: Scattering Rows of a Matrix

```c
/* Scatter rows of an NROWS × NCOLS matrix to P processes (one row per process) */
double matrix[NROWS][NCOLS];  /* row-major */

/* Create a type for one row */
MPI_Datatype row_type, row_resized;
MPI_Type_contiguous(NCOLS, MPI_DOUBLE, &row_type);

/* Resize to make the extent exactly one row (NCOLS doubles) */
MPI_Type_create_resized(row_type, 0, NCOLS * sizeof(double), &row_resized);
MPI_Type_commit(&row_resized);
MPI_Type_free(&row_type);

double local_row[NCOLS];
MPI_Scatter(matrix, 1, row_resized,    /* root sends 1 row_resized per rank */
            local_row, NCOLS, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

MPI_Type_free(&row_resized);
```

Without the resize, `MPI_Scatter` would not know where each row begins in the matrix.

---

## 13.7 MPI_Type_create_subarray

Describes a sub-region of a multi-dimensional array. This is the standard way to
distribute a multi-dimensional grid across processes.

```c
int MPI_Type_create_subarray(int ndims,
                              const int array_of_sizes[],
                              const int array_of_subsizes[],
                              const int array_of_starts[],
                              int order,            /* MPI_ORDER_C or MPI_ORDER_FORTRAN */
                              MPI_Datatype oldtype,
                              MPI_Datatype *newtype);
```

```c
/* Global 1000×1000 array; each process owns a 250×250 subarray */
/* This process owns the subarray starting at (row_start, col_start) */

int global_sizes[2] = {1000, 1000};
int local_sizes[2]  = { 250,  250};
int starts[2]       = {row_start, col_start};

MPI_Datatype subarray_type;
MPI_Type_create_subarray(2,            /* ndims */
                          global_sizes, /* full array dimensions */
                          local_sizes,  /* subarray dimensions */
                          starts,       /* starting corner */
                          MPI_ORDER_C,  /* row-major (C convention) */
                          MPI_DOUBLE,
                          &subarray_type);
MPI_Type_commit(&subarray_type);

/* Write this process's data to a shared file at the correct position */
MPI_File_set_view(fh, 0, MPI_DOUBLE, subarray_type, "native", MPI_INFO_NULL);
MPI_File_write_all(fh, local_data, local_n, MPI_DOUBLE, MPI_STATUS_IGNORE);

MPI_Type_free(&subarray_type);
```

`MPI_Type_create_subarray` is the workhorse for MPI-IO with distributed arrays.

---

## 13.8 MPI_Type_create_darray

Describes the portion of a distributed array that belongs to this process according
to a block-cyclic distribution (as in ScaLAPACK). This is a higher-level abstraction
than `MPI_Type_create_subarray`.

```c
int MPI_Type_create_darray(int size, int rank, int ndims,
                            const int array_of_gsizes[],
                            const int array_of_distribs[],
                            const int array_of_dargs[],
                            const int array_of_psizes[],
                            int order, MPI_Datatype oldtype,
                            MPI_Datatype *newtype);
```

Use `MPI_DISTRIBUTE_BLOCK`, `MPI_DISTRIBUTE_CYCLIC`, or `MPI_DISTRIBUTE_NONE` as
distribution types. `MPI_DISTRIBUTE_DFLT_DARG` means "use default block size."

This is mainly used with MPI-IO for reading/writing distributed ScaLAPACK-format
matrices from/to files.

---

## 13.9 Querying Derived Types

```c
MPI_Datatype my_type;
/* ... create and commit ... */

/* Size: bytes actually transferred */
int size;
MPI_Type_size(my_type, &size);

MPI_Aint lb, extent;
MPI_Type_get_extent(my_type, &lb, &extent);

MPI_Aint true_lb, true_extent;
MPI_Type_get_true_extent(my_type, &true_lb, &true_extent);
```

The difference between **extent** and **true extent**:
- **Extent**: includes any padding at the beginning (lower bound) and end that the
  type itself declared. Can be artificially resized with `MPI_Type_create_resized`.
- **True extent**: the actual span of bytes accessed by the type, regardless of
  explicit lower bound and extent settings.

Use `true_extent` when you need to know the actual memory footprint.

---

## 13.10 Worked Examples

### Halo Exchange with Vector Type

```c
/* 2D grid: send left/right columns (vertical halos) */
int local_rows = ..., local_cols = ...;
double grid[local_rows][local_cols + 2]; /* +2 for ghost columns */

/* Type for a single column (vertical strip) */
MPI_Datatype col_type;
MPI_Type_vector(local_rows,       /* count: one block per row */
                1,                 /* blocklength: 1 element */
                local_cols + 2,   /* stride: full row width */
                MPI_DOUBLE, &col_type);
MPI_Type_commit(&col_type);

/* Exchange: send rightmost real column, receive into left ghost column */
MPI_Sendrecv(
    &grid[0][local_cols],    1, col_type, right_rank, 0,   /* send right col */
    &grid[0][0],             1, col_type, left_rank,  0,   /* recv left ghost */
    comm, MPI_STATUS_IGNORE
);

MPI_Type_free(&col_type);
```

### Sending a Struct Array

```c
typedef struct { float pos[3]; float vel[3]; int id; } Particle;

Particle particles[N];

MPI_Aint base, disp[3];
MPI_Get_address(&particles[0],       &base);
MPI_Get_address(&particles[0].pos,   &disp[0]);
MPI_Get_address(&particles[0].vel,   &disp[1]);
MPI_Get_address(&particles[0].id,    &disp[2]);
for (int i = 0; i < 3; i++) disp[i] -= base;

int blocklens[3] = {3, 3, 1};
MPI_Datatype types[3] = {MPI_FLOAT, MPI_FLOAT, MPI_INT};

MPI_Datatype particle_type, particle_resized;
MPI_Type_create_struct(3, blocklens, disp, types, &particle_type);

/* Resize to sizeof(Particle) so arrays work correctly */
MPI_Type_create_resized(particle_type, 0, sizeof(Particle), &particle_resized);
MPI_Type_commit(&particle_resized);
MPI_Type_free(&particle_type);

MPI_Send(particles, N, particle_resized, dest, tag, comm);
MPI_Type_free(&particle_resized);
```

---

## Summary

| Constructor | Layout |
|---|---|
| `MPI_Type_contiguous` | N copies of a type, back-to-back |
| `MPI_Type_vector` | N blocks of M elements, regular stride (in elements) |
| `MPI_Type_hvector` | N blocks of M elements, regular stride (in bytes) |
| `MPI_Type_indexed` | N blocks at irregular positions (in elements) |
| `MPI_Type_hindexed` | N blocks at irregular positions (in bytes) |
| `MPI_Type_create_struct` | Heterogeneous fields at arbitrary byte offsets |
| `MPI_Type_create_subarray` | Rectangular subregion of an n-dim array |
| `MPI_Type_create_darray` | Block-cyclic distributed array slice |
| `MPI_Type_create_resized` | Override lower bound and extent |

**Golden rules**:
- Always call `MPI_Type_commit` before use.
- Always call `MPI_Type_free` when done.
- Always use `MPI_Get_address` for struct displacements — never compute manually.
- Use `MPI_Type_create_resized` when using derived types with count > 1.

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
