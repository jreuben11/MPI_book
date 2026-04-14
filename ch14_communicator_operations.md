# Chapter 14: Communicator Operations

## 14.1 Why Create New Communicators?

`MPI_COMM_WORLD` is a single shared namespace. When multiple independent subsystems
(your code, a library, a solver component) all use `MPI_COMM_WORLD`, their collective
calls can interfere — a library's `MPI_Barrier` might match your application's
`MPI_Barrier` if they happen to execute concurrently.

Creating new communicators provides:
- **Context isolation**: messages on different communicators cannot be received by
  the wrong collective or point-to-point operation.
- **Subset communication**: algorithms that only involve a subset of processes.
- **Topology information**: attaching Cartesian or graph structure (Chapter 15).

---

## 14.2 MPI_Comm_dup

```c
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
```

Creates an exact copy with a new context. The new communicator has the same process
group but is isolated from the original.

```c
/* Library initialization: create a private communicator */
MPI_Comm lib_comm;
MPI_Comm_dup(user_comm, &lib_comm);

/* All library collectives use lib_comm — they cannot collide with user_comm */
MPI_Barrier(lib_comm);

/* When library is done */
MPI_Comm_free(&lib_comm);
```

`MPI_Comm_dup` is a collective — all processes in `comm` must call it simultaneously.
The resulting communicators on all processes refer to the same context.

`MPI_Comm_idup` (MPI 3.0) is the non-blocking version, useful when dup overhead is
significant:

```c
MPI_Request req;
MPI_Comm_idup(comm, &new_comm, &req);
do_initialization_work();
MPI_Wait(&req, MPI_STATUS_IGNORE);
/* new_comm is now ready */
```

---

## 14.3 MPI_Comm_split

```c
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
```

Splits a communicator into subgroups based on `color`. All processes with the same
`color` end up in the same new communicator. Within a new communicator, ranks are
assigned in order of `key` (ties broken by original rank).

```c
/* Split into groups: even-ranked and odd-ranked processes */
int color = rank % 2;
MPI_Comm row_comm;
MPI_Comm_split(MPI_COMM_WORLD, color, rank, &row_comm);

/* Each row_comm contains either {0,2,4,...} or {1,3,5,...} */
int row_rank, row_size;
MPI_Comm_rank(row_comm, &row_rank);
MPI_Comm_size(row_comm, &row_size);

/* Operations within the subgroup */
MPI_Allreduce(MPI_IN_PLACE, &local_val, 1, MPI_DOUBLE, MPI_SUM, row_comm);

MPI_Comm_free(&row_comm);
```

Use `color = MPI_UNDEFINED` to exclude a process from all new communicators
(the process gets `MPI_COMM_NULL`):

```c
/* Only ranks 0..N/2-1 participate */
int color = (rank < size/2) ? 0 : MPI_UNDEFINED;
MPI_Comm half_comm;
MPI_Comm_split(MPI_COMM_WORLD, color, rank, &half_comm);

if (half_comm != MPI_COMM_NULL) {
    /* ... work on first half ... */
    MPI_Comm_free(&half_comm);
}
```

### 2D Decomposition Pattern

Many parallel algorithms — 2D stencil computations, distributed matrix operations,
parallel FFT — decompose work across a 2D logical process grid. Each process owns a
rectangular tile of a 2D domain and communicates with its four neighbors (left, right,
above, below). Two communicators are needed: one per row, one per column.

#### MPI_Dims_create

```c
int MPI_Dims_create(int nnodes, int ndims, int dims[]);
```

Given a total process count `nnodes` and number of dimensions `ndims`, fills in the
`dims` array with a balanced factorisation. Elements set to 0 are computed; non-zero
elements are treated as fixed constraints.

```c
int dims[2] = {0, 0};
MPI_Dims_create(16, 2, dims);  /* → {4, 4} */
MPI_Dims_create(12, 2, dims);  /* → {4, 3} (or {3,4} — impl. dependent) */
MPI_Dims_create(6,  2, dims);  /* → {3, 2} */

/* Fix the number of rows, let MPI choose columns */
int dims2[2] = {2, 0};
MPI_Dims_create(16, 2, dims2); /* → {2, 8} */
```

The factorisation is chosen to minimise the maximum dimension, producing the most
"square" grid possible. This minimises the surface-to-volume ratio of each tile,
which reduces halo exchange volume (see Chapter 6).

#### Rank-to-Grid Mapping

With `dims = {nrows, ncols}`, lay ranks out in **row-major** order — the same order
as C 2D arrays:

```
16 processes, dims = {4, 4}:

          col 0   col 1   col 2   col 3
  row 0: [ r=0 ] [ r=1 ] [ r=2 ] [ r=3 ]
  row 1: [ r=4 ] [ r=5 ] [ r=6 ] [ r=7 ]
  row 2: [ r=8 ] [ r=9 ] [r=10 ] [r=11 ]
  row 3: [r=12 ] [r=13 ] [r=14 ] [r=15 ]
```

Each rank derives its (row, col) coordinates with integer arithmetic:

```c
int row = rank / dims[1];   /* which row am I in? */
int col = rank % dims[1];   /* which column am I in? */
```

For rank 9: `row = 9/4 = 2`, `col = 9%4 = 1` → grid position (2,1). ✓

#### Splitting into Row and Column Communicators

`MPI_Comm_split` maps `color` → "which new communicator do I join?" and `key` →
"what rank do I get within that communicator?"

```c
MPI_Comm row_comm, col_comm;

/* row_comm: color = row index → all processes in the same row join together.
   key = col → within the row communicator, ranks are assigned left-to-right. */
MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);

/* col_comm: color = col index → all processes in the same column join together.
   key = row → within the column communicator, ranks are assigned top-to-bottom. */
MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);
```

Result for the 4×4 grid:

```
row_comm groups (each group = one row):
  row_comm[0]: {rank 0, 1, 2,  3 }  → row_comm ranks 0,1,2,3
  row_comm[1]: {rank 4, 5, 6,  7 }  → row_comm ranks 0,1,2,3
  row_comm[2]: {rank 8, 9, 10, 11}  → row_comm ranks 0,1,2,3
  row_comm[3]: {rank12,13, 14, 15}  → row_comm ranks 0,1,2,3

col_comm groups (each group = one column):
  col_comm[0]: {rank 0, 4,  8, 12}  → col_comm ranks 0,1,2,3
  col_comm[1]: {rank 1, 5,  9, 13}  → col_comm ranks 0,1,2,3
  col_comm[2]: {rank 2, 6, 10, 14}  → col_comm ranks 0,1,2,3
  col_comm[3]: {rank 3, 7, 11, 15}  → col_comm ranks 0,1,2,3
```

Every rank belongs to exactly one `row_comm` and exactly one `col_comm`.

#### Full Example — 2D Halo Neighbour Discovery and Reduction

```c
int dims[2] = {0, 0};
MPI_Dims_create(size, 2, dims);

int nrows = dims[0], ncols = dims[1];
int row = rank / ncols;
int col = rank % ncols;

MPI_Comm row_comm, col_comm;
MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

/* Neighbours in the 2D grid (MPI_PROC_NULL for boundary ranks) */
int left  = (col > 0)        ? rank - 1      : MPI_PROC_NULL;
int right = (col < ncols-1)  ? rank + 1      : MPI_PROC_NULL;
int up    = (row > 0)        ? rank - ncols  : MPI_PROC_NULL;
int down  = (row < nrows-1)  ? rank + ncols  : MPI_PROC_NULL;

/* Reduce across own row — each rank contributes local_row_val */
double local_row_val = compute_row_contribution();
MPI_Allreduce(MPI_IN_PLACE, &local_row_val, 1, MPI_DOUBLE, MPI_SUM, row_comm);

/* Reduce across own column */
double local_col_val = compute_col_contribution();
MPI_Allreduce(MPI_IN_PLACE, &local_col_val, 1, MPI_DOUBLE, MPI_SUM, col_comm);

MPI_Comm_free(&row_comm);
MPI_Comm_free(&col_comm);
```

`MPI_PROC_NULL` is a safe dummy rank: sends to it vanish, receives from it return
immediately with zero bytes — no special-casing needed for boundary processes.

#### When to Use This Pattern

| Use case | What the split provides |
|---|---|
| 2D stencil (PDE, image processing) | Halo exchange with 4 neighbours; row/col reductions for boundary conditions |
| Distributed matrix multiply (SUMMA, Cannon's algorithm) | Broadcast a matrix panel along `row_comm`; reduce result along `col_comm` |
| Parallel FFT | 1D FFTs along rows using `row_comm`; transpose; 1D FFTs along columns using `col_comm` |
| 2D domain I/O | Each row writes its slab collectively using `row_comm` |

Note: Chapter 15 covers `MPI_Cart_create`, which offers the same 2D grid structure
plus built-in shift/neighbour queries (`MPI_Cart_shift`, `MPI_Cart_coords`) without
the manual rank arithmetic shown above. Use Cartesian communicators when topology
information (dimension count, periodicity) needs to be embedded in the communicator
itself.

---

## 14.4 MPI_Comm_split_type

```c
int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key,
                         MPI_Info info, MPI_Comm *newcomm);
```

Splits into groups based on hardware topology rather than user-supplied colors.

The most useful split type is `MPI_COMM_TYPE_SHARED`: processes that can share memory
(i.e., on the same node/socket) are grouped together.

```c
/* Create a communicator of processes sharing the same shared memory domain */
MPI_Comm shared_comm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                    MPI_INFO_NULL, &shared_comm);

int local_rank;
MPI_Comm_rank(shared_comm, &local_rank);
/* local_rank is the within-node rank — use for local_rank==0 I/O pattern */

MPI_Comm_free(&shared_comm);
```

Practical uses:
- Determine which rank is "node master" (`local_rank == 0`).
- Allocate shared memory windows on the node (Chapter 16).
- Build hierarchical communication algorithms.

---

## 14.5 MPI_Comm_create and MPI_Comm_create_group

### MPI_Comm_create

```c
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
```

Creates a new communicator from an explicit group. The group must be a subgroup of
`comm`'s group. All processes in `comm` must call this collectively.

```c
/* Create a communicator for ranks 0,4,8,12 */
MPI_Group world_group, sub_group;
MPI_Comm_group(MPI_COMM_WORLD, &world_group);

int ranks[4] = {0, 4, 8, 12};
MPI_Group_incl(world_group, 4, ranks, &sub_group);

MPI_Comm new_comm;
MPI_Comm_create(MPI_COMM_WORLD, sub_group, &new_comm);
/* Processes not in sub_group get MPI_COMM_NULL */

MPI_Group_free(&world_group);
MPI_Group_free(&sub_group);

if (new_comm != MPI_COMM_NULL) {
    /* ... work with {0,4,8,12} ... */
    MPI_Comm_free(&new_comm);
}
```

### MPI_Comm_create_group

```c
int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag,
                           MPI_Comm *newcomm);
```

Like `MPI_Comm_create` but only the processes in `group` need to call it — processes
not in the group do not participate. This avoids the cost of a global collective when
creating communicators for subsets.

```c
/* Only ranks 0-7 create a communicator; ranks 8+ do not participate */
if (rank < 8) {
    MPI_Group world_group, sub_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int ranks[8] = {0,1,2,3,4,5,6,7};
    MPI_Group_incl(world_group, 8, ranks, &sub_group);

    MPI_Comm sub_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &sub_comm);

    MPI_Group_free(&world_group);
    MPI_Group_free(&sub_group);

    /* Use sub_comm ... */
    MPI_Comm_free(&sub_comm);
}
/* ranks 8+ simply skip this block */
```

---

## 14.6 Inter-Communicators

All communicators discussed so far are **intra-communicators**: all processes are in
the same group and can communicate with each other.

An **inter-communicator** bridges two disjoint groups. Messages sent to "the remote
group" arrive at the corresponding process in that group. Collectives work across
both groups.

```c
/* Create an inter-communicator between two intra-communicators */
MPI_Comm intercomm;
MPI_Intercomm_create(
    local_comm,   /* my local communicator */
    0,            /* local leader rank in local_comm */
    peer_comm,    /* a communicator that spans both groups (e.g., COMM_WORLD) */
    remote_leader, /* peer leader rank in peer_comm */
    tag,
    &intercomm);
```

In an inter-communicator:
- `MPI_Comm_size` returns the size of the **local** group.
- `MPI_Comm_remote_size` returns the size of the **remote** group.
- Point-to-point sends/receives go to/from the remote group.

```c
int remote_size;
MPI_Comm_remote_size(intercomm, &remote_size);
```

### Merging an Inter-Communicator

```c
/* Convert inter-communicator back to intra-communicator */
MPI_Comm merged;
MPI_Intercomm_merge(intercomm, high, &merged);
/* high=0 for "local group first", high=1 for "remote group first" */
```

Inter-communicators are most commonly used with `MPI_Comm_spawn` for dynamic process
management (not covered in this guide) and in coupled multi-physics simulations where
two MPI jobs need to exchange data.

---

## 14.7 Communicator Attributes

You can attach arbitrary key-value pairs to communicators:

```c
/* Create a keyval */
int keyval;
MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN,
                        &keyval, NULL);

/* Attach data to a communicator */
int my_data = 42;
MPI_Comm_set_attr(comm, keyval, &my_data);

/* Retrieve it */
int *retrieved;
int flag;
MPI_Comm_get_attr(comm, keyval, &retrieved, &flag);
if (flag) printf("Data: %d\n", *retrieved);

/* Cleanup */
MPI_Comm_delete_attr(comm, keyval);
MPI_Comm_free_keyval(&keyval);
```

The copy function (first argument to `MPI_Comm_create_keyval`) is called when
`MPI_Comm_dup` duplicates the communicator. Use it to deep-copy your attached data.
`MPI_COMM_NULL_COPY_FN` means "do not copy the attribute on dup."
`MPI_COMM_NULL_DELETE_FN` means "do nothing when the communicator is freed."

Attributes are most useful for library implementation — attaching context to a
user-provided communicator without requiring the user to pass extra state.

---

## 14.8 Comparing and Naming Communicators

```c
/* Compare two communicators */
int result;
MPI_Comm_compare(comm1, comm2, &result);
/* result: MPI_IDENT (same group and context)
           MPI_CONGRUENT (same group, different context)
           MPI_SIMILAR (same members, different order)
           MPI_UNEQUAL (different) */

/* Name a communicator for debugging */
MPI_Comm_set_name(comm, "solver_comm");
char name[MPI_MAX_OBJECT_NAME];
int namelen;
MPI_Comm_get_name(comm, name, &namelen);
```

---

## Summary

| Function | Purpose |
|---|---|
| `MPI_Comm_dup` | Clone with new context; for library isolation |
| `MPI_Comm_idup` | Non-blocking dup |
| `MPI_Comm_split` | Partition by color+key |
| `MPI_Comm_split_type` | Partition by hardware (e.g., `MPI_COMM_TYPE_SHARED`) |
| `MPI_Comm_create` | Create from group (all processes must call) |
| `MPI_Comm_create_group` | Create from group (only members call) |
| `MPI_Intercomm_create` | Bridge two disjoint groups |
| `MPI_Intercomm_merge` | Convert inter- to intra-communicator |
| `MPI_Comm_free` | Release handle; sets to `MPI_COMM_NULL` |
| `MPI_Dims_create` | Compute balanced grid dimensions |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
