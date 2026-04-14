# Chapter 15: Process Topologies

## 15.1 What Topologies Provide

A process topology attaches a virtual graph structure to a communicator. It lets you:

1. **Think in problem coordinates** rather than flat ranks. A 3D stencil code can
   address processes as (x, y, z) instead of rank 0, 1, 2, ...
2. **Let MPI reorder processes** for better physical placement on the network.
3. **Get convenient neighbor queries** (`MPI_Cart_shift`) without manual arithmetic.

Topologies are purely informational — they do not restrict which processes can
communicate. Any process can still send to any other process on any communicator.

---

## 15.2 Cartesian Topology

The most common topology for structured grid codes. Processes are arranged in an
N-dimensional grid.

### Creating a Cartesian Communicator

```c
int MPI_Cart_create(MPI_Comm comm_old, int ndims,
                    const int dims[], const int periods[], int reorder,
                    MPI_Comm *comm_cart);
```

| Argument | Meaning |
|---|---|
| `ndims` | Number of dimensions |
| `dims[]` | Size of the grid in each dimension |
| `periods[]` | Whether each dimension wraps around (periodic) |
| `reorder` | Allow MPI to reorder ranks for better placement |

```c
/* Create a 2D 4×4 periodic Cartesian grid for 16 processes */
int dims[2]    = {4, 4};   /* 4 rows × 4 columns */
int periods[2] = {1, 1};   /* periodic in both dimensions (torus) */
int reorder    = 1;        /* allow MPI to reorder */

MPI_Comm cart_comm;
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
```

**Non-periodic 4×4 grid** (`periods = {0, 0}`):

Ranks are laid out in row-major order. Each cell shows `flat_rank` at coordinates
`(row, col)`. Edge processes have `MPI_PROC_NULL` as their missing neighbor.

```
         col 0     col 1     col 2     col 3
        ┌─────────┬─────────┬─────────┬─────────┐
row 0   │  0(0,0) │  1(0,1) │  2(0,2) │  3(0,3) │  ← no north neighbor
        ├─────────┼─────────┼─────────┼─────────┤
row 1   │  4(1,0) │  5(1,1) │  6(1,2) │  7(1,3) │
        ├─────────┼─────────┼─────────┼─────────┤
row 2   │  8(2,0) │  9(2,1) │ 10(2,2) │ 11(2,3) │
        ├─────────┼─────────┼─────────┼─────────┤
row 3   │ 12(3,0) │ 13(3,1) │ 14(3,2) │ 15(3,3) │  ← no south neighbor
        └─────────┴─────────┴─────────┴─────────┘
             ↑ no west neighbor        no east neighbor ↑
             (col 0)                            (col 3)
```

**Periodic 4×4 torus** (`periods = {1, 1}`): row 0 wraps to row 3 (north/south),
col 0 wraps to col 3 (east/west). Every process has exactly 4 neighbors.

```
   ┌──────────────────────────────────────────┐  (south wraps to north)
   ↓                                          ↑
  [ 0]──[ 1]──[ 2]──[ 3]──┐    row 0
   |     |     |     |     |
  [ 4]──[ 5]──[ 6]──[ 7]──┤    row 1
   |     |     |     |     |
  [ 8]──[ 9]──[10]──[11]──┤    row 2
   |     |     |     |     |
  [12]──[13]──[14]──[15]──┘    row 3
   ↑                        ↑
   └────────────────────────┘  (east wraps to west)
```

### Querying Position

```c
/* Get my coordinates in the Cartesian grid */
/* IMPORTANT: if reorder=1 was passed to MPI_Cart_create, use the cart_comm rank,
   not the MPI_COMM_WORLD rank — the two may differ after reordering */
int cart_rank;
MPI_Comm_rank(cart_comm, &cart_rank);
int coords[2];
MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
printf("Rank %d is at grid position (%d, %d)\n", cart_rank, coords[0], coords[1]);

/* Convert coordinates back to rank */
int target_rank;
int target_coords[2] = {2, 3};
MPI_Cart_rank(cart_comm, target_coords, &target_rank);
```

### MPI_Cart_shift — Neighbor Queries

```c
int MPI_Cart_shift(MPI_Comm comm, int direction, int disp,
                   int *rank_source, int *rank_dest);
```

Returns the rank of the neighbor `disp` steps away in `direction` (0-indexed
dimension). For non-periodic grids, boundary processes get `MPI_PROC_NULL` for
their missing neighbor.

The `direction` argument selects the axis (0 = rows, 1 = cols in a 2D grid).
`disp = +1` steps forward; `disp = -1` steps backward. `rank_source` is who
sends *to* me; `rank_dest` is who I send *to*.

For rank **5** at coordinates **(1,1)** in the non-periodic 4×4 grid:

```
                north = rank 1  (0,1)
                    ↑
                    │
west = rank 4 ←── [5] ──→  east = rank 6
(1,0)          (1,1)               (1,2)
                    │
                    ↓
                south = rank 9  (2,1)

  MPI_Cart_shift(cart_comm, 0, 1, &north, &south)  → north=1,  south=9
  MPI_Cart_shift(cart_comm, 1, 1, &west,  &east )  → west=4,   east=6

Corner rank 0 at (0,0) — non-periodic:
  north = MPI_PROC_NULL  (no row above)
  west  = MPI_PROC_NULL  (no column left)
  south = rank 4,  east = rank 1
```

```c
int left, right, up, down;

/* Shift along dimension 1 (columns) by ±1 */
MPI_Cart_shift(cart_comm, 1, 1, &left,  &right);

/* Shift along dimension 0 (rows) by ±1 */
MPI_Cart_shift(cart_comm, 0, 1, &up, &down);

/* Send to right neighbor, receive from left */
MPI_Sendrecv(right_halo, N, MPI_DOUBLE, right, 0,
             left_halo,  N, MPI_DOUBLE, left,  0,
             cart_comm, MPI_STATUS_IGNORE);
```

`MPI_PROC_NULL` is a special rank that absorbs sends and provides empty receives —
so boundary-rank code does not need special-casing:

```c
/* This works correctly even at boundaries where left = MPI_PROC_NULL */
MPI_Sendrecv(send_left, N, MPI_DOUBLE, left,  0,
             recv_right, N, MPI_DOUBLE, right, 0,
             cart_comm, MPI_STATUS_IGNORE);
/* If left == MPI_PROC_NULL: send is discarded, recv returns 0 elements */
```

---

## 15.3 MPI_Cart_sub — Slicing the Grid

Create sub-communicators corresponding to rows, columns, or other hyperplane slices:

```c
int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm);
```

`remain_dims[i] = 1` means dimension `i` varies in the sub-communicator.
`remain_dims[i] = 0` means dimension `i` is fixed (we cut along it).

**2D grid sliced into row and column communicators:**

```
4×4 Cartesian grid → row_comm (remain_dims={1,0}) → col_comm (remain_dims={0,1})

remain_dims = {1, 0}          remain_dims = {0, 1}
(dimension 0 = rows varies)   (dimension 1 = cols varies)

row communicators:            column communicators:
┌────┬────┬────┬────┐         ┌────┬────┬────┬────┐
│  0 │  1 │  2 │  3 │← row0  │  0 │  1 │  2 │  3 │
├────┼────┼────┼────┤         │  ↕ │  ↕ │  ↕ │  ↕ │
│  4 │  5 │  6 │  7 │← row1  │  4 │  5 │  6 │  7 │
├────┼────┼────┼────┤         │  ↕ │  ↕ │  ↕ │  ↕ │
│  8 │  9 │ 10 │ 11 │← row2  │  8 │  9 │ 10 │ 11 │
├────┼────┼────┼────┤         │  ↕ │  ↕ │  ↕ │  ↕ │
│ 12 │ 13 │ 14 │ 15 │← row3  │ 12 │ 13 │ 14 │ 15 │
└────┴────┴────┴────┘         └────┴────┴────┴────┘
  ↔ all 4 share row_comm        ↕ each col shares col_comm
```

**3D grid sliced into XY planes** — `remain_dims = {1,1,0}` fixes Z, varies X and Y:

```
3D grid dims={2,2,2}:

  Z=0 plane:              Z=1 plane:
  ┌────┬────┐             ┌────┬────┐
  │  0 │  1 │             │  4 │  5 │
  ├────┼────┤             ├────┼────┤
  │  2 │  3 │             │  6 │  7 │
  └────┴────┘             └────┴────┘
  xy_comm[0]              xy_comm[1]
  {0,1,2,3}               {4,5,6,7}
```

```c
/* 3D Cartesian grid; create communicators for each XY plane */
int dims[3]    = {P, Q, R};
int periods[3] = {0, 0, 0};
MPI_Comm cart3d;
MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart3d);

/* Row communicator: dimension 0 varies, dimension 1 fixed */
int remain_row[3] = {1, 0, 0};
MPI_Comm row_comm;
MPI_Cart_sub(cart3d, remain_row, &row_comm);

/* Column communicator: dimension 1 varies, dimension 0 fixed */
int remain_col[3] = {0, 1, 0};
MPI_Comm col_comm;
MPI_Cart_sub(cart3d, remain_col, &col_comm);

/* All-reduce within each row */
MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, row_comm);

MPI_Comm_free(&row_comm);
MPI_Comm_free(&col_comm);
MPI_Comm_free(&cart3d);
```

---

## 15.4 Choosing Grid Dimensions with MPI_Dims_create

```c
int MPI_Dims_create(int nnodes, int ndims, int dims[]);
```

Fills `dims[]` with values that create a roughly balanced grid. Any non-zero entry
in `dims[]` is treated as fixed; zero entries are computed.

```c
int dims[2] = {0, 0};
MPI_Dims_create(size, 2, dims);
/* For size=12: dims might be {4,3} or {3,4} — implementation-defined */

/* Fix one dimension, compute the other */
int dims2[2] = {4, 0};
MPI_Dims_create(size, 2, dims2);
/* dims2[1] = size/4 */
```

---

## 15.5 Distributed Graph Topology

The Cartesian topology works well for structured grids. For irregular communication
patterns (sparse graphs, unstructured meshes, particle codes with dynamic neighbors),
use the distributed graph topology.

**Example: 6-process sparse communication graph**

```
Rank 0 sends to: {1, 3}       Rank 3 sends to: {0, 4}
Rank 1 sends to: {0, 2}       Rank 4 sends to: {3, 5}
Rank 2 sends to: {1, 5}       Rank 5 sends to: {2, 4}

           0 ──── 1 ──── 2
           │             │
           3 ──── 4 ──── 5

Each rank declares only its own in/out edges — no global adjacency matrix needed.
```

```c
int MPI_Dist_graph_create_adjacent(
    MPI_Comm comm_old,
    int indegree,  const int sources[],      const int sourceweights[],
    int outdegree, const int destinations[],  const int destweights[],
    MPI_Info info, int reorder,
    MPI_Comm *comm_dist_graph);
```

Each process describes only its own neighbors. `sources` is who sends to me;
`destinations` is who I send to. Weights are hints to MPI for optimization.

```c
/* Process rank communicates with a sparse set of neighbors */
int n_sources = 3, n_dests = 2;
int sources[3]      = {neighbor_a, neighbor_b, neighbor_c};
int destinations[2] = {neighbor_x, neighbor_y};
/* Use MPI_UNWEIGHTED as the weights pointer — it is a special sentinel, not an array */
MPI_Comm graph_comm;
MPI_Dist_graph_create_adjacent(
    MPI_COMM_WORLD,
    n_sources, sources, MPI_UNWEIGHTED,
    n_dests, destinations, MPI_UNWEIGHTED,
    MPI_INFO_NULL, 1, &graph_comm);
```

### Querying Graph Neighbors

```c
int indegree, outdegree, weighted;
MPI_Dist_graph_neighbors_count(graph_comm, &indegree, &outdegree, &weighted);

int *in_neighbors  = malloc(indegree  * sizeof(int));
int *out_neighbors = malloc(outdegree * sizeof(int));
/* Pass MPI_UNWEIGHTED (not NULL) for weights when graph was created without weights */
MPI_Dist_graph_neighbors(graph_comm, indegree, in_neighbors, MPI_UNWEIGHTED,
                          outdegree, out_neighbors, MPI_UNWEIGHTED);
```

### MPI_Neighbor_alltoall — Topology-Aware Collective

```c
/* Exchange data with all neighbors according to the graph topology */
int MPI_Neighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype,
                           MPI_Comm comm);
```

Each process sends `sendcount` elements to each of its out-neighbors and receives
`recvcount` elements from each in-neighbor. The implementation can use the graph
structure to optimize routing.

```c
/* Send one double to each out-neighbor; receive one from each in-neighbor */
double *send_data = malloc(outdegree * sizeof(double));
double *recv_data = malloc(indegree  * sizeof(double));

/* Fill send_data[i] for out_neighbors[i] */

MPI_Neighbor_alltoall(send_data, 1, MPI_DOUBLE,
                      recv_data, 1, MPI_DOUBLE, graph_comm);
```

Variable-count variant: `MPI_Neighbor_alltoallv`. Non-blocking: `MPI_Ineighbor_alltoall`.

---

## 15.6 Topology-Aware Placement and Reordering

When `reorder = 1`, MPI may reassign ranks to better match the physical network
topology. After `MPI_Cart_create` with `reorder = 1`, the rank of a process in
`cart_comm` may differ from its rank in `comm_old`.

```c
int old_rank = rank;  /* rank in MPI_COMM_WORLD */
MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1 /* reorder */, &cart_comm);

int new_rank;
MPI_Comm_rank(cart_comm, &new_rank);
/* old_rank != new_rank if MPI reordered */
```

In practice, most implementations do not reorder even with `reorder = 1`. The hint
is advisory. Pass `reorder = 0` if you need to preserve the rank ordering (e.g., for
debugging or to avoid confusing existing rank-based code).

---

## 15.7 Worked Example: 2D Stencil with Cartesian Topology

Each process owns a `local_rows × local_cols` interior patch surrounded by a 1-cell
ghost layer. The ghost cells hold copies of the adjacent process's boundary data,
refreshed each time step by the halo exchange.

```
  Memory layout for one process (local_rows=4, local_cols=4, halo width=1):

  grid[0][0..5]  → ghost row (filled from north neighbor via MPI)
  ┌──────┬──────────────────┬──────┐
  │  GW  │  ghost row N     │  GE  │  row 0  (ghost)
  ├──────┼──────────────────┼──────┤
  │  G   │ [1][1]..[1][4]   │  G   │  row 1  ← owned interior
  │  W   │ [2][1]..[2][4]   │  E   │  row 2  ← owned interior
  │      │ [3][1]..[3][4]   │      │  row 3  ← owned interior
  │      │ [4][1]..[4][4]   │      │  row 4  ← owned interior
  ├──────┼──────────────────┼──────┤
  │  GW  │  ghost row S     │  GE  │  row 5  (ghost)
  └──────┴──────────────────┴──────┘
   col 0       col 1-4          col 5
  (ghost W)  (owned cols)    (ghost E)

  Arrows show the 4 halo exchange directions:
        ↑ send row 1 to north / recv into row 0
        ↓ send row 4 to south / recv into row 5
        ← send col 1 to west  / recv into col 0  (non-contiguous → MPI_Type_vector)
        → send col 4 to east  / recv into col 5  (non-contiguous → MPI_Type_vector)
```

The column halo is non-contiguous in memory (elements `grid[r][0]` for `r=1..lr` are
stride `lc+2` apart), so a vector derived datatype is used instead of a raw pointer.

```c
/* 5-point 2D stencil: each process owns a local_rows × local_cols patch */
int dims[2] = {0, 0};
MPI_Dims_create(size, 2, dims);

int periods[2] = {0, 0};  /* non-periodic */
MPI_Comm cart_comm;
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

int cart_rank;
MPI_Comm_rank(cart_comm, &cart_rank);  /* use cart_comm rank, not COMM_WORLD rank */
int coords[2];
MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

int north, south, east, west;
MPI_Cart_shift(cart_comm, 0, 1, &north, &south);  /* dim 0 = rows */
MPI_Cart_shift(cart_comm, 1, 1, &west,  &east);   /* dim 1 = cols */

/* Allocate local grid with halos */
int lr = local_rows, lc = local_cols;
double grid[(lr+2)][(lc+2)];  /* +2 for ghost rows/cols */

/* Halo exchange — 4 directions */
MPI_Request reqs[8];
int nreqs = 0;

/* Vertical: exchange rows */
MPI_Irecv(&grid[0][1],    lc, MPI_DOUBLE, north, 0, cart_comm, &reqs[nreqs++]);
MPI_Irecv(&grid[lr+1][1], lc, MPI_DOUBLE, south, 1, cart_comm, &reqs[nreqs++]);
MPI_Isend(&grid[1][1],    lc, MPI_DOUBLE, north, 1, cart_comm, &reqs[nreqs++]);
MPI_Isend(&grid[lr][1],   lc, MPI_DOUBLE, south, 0, cart_comm, &reqs[nreqs++]);

/* Horizontal: exchange columns (use a vector type for non-contiguous data) */
MPI_Datatype col_type;
MPI_Type_vector(lr, 1, lc+2, MPI_DOUBLE, &col_type);
MPI_Type_commit(&col_type);

MPI_Irecv(&grid[1][0],    1, col_type, west, 2, cart_comm, &reqs[nreqs++]);
MPI_Irecv(&grid[1][lc+1], 1, col_type, east, 3, cart_comm, &reqs[nreqs++]);
MPI_Isend(&grid[1][1],    1, col_type, west, 3, cart_comm, &reqs[nreqs++]);
MPI_Isend(&grid[1][lc],   1, col_type, east, 2, cart_comm, &reqs[nreqs++]);

compute_interior_stencil(grid, lr, lc);

MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

compute_boundary_stencil(grid, lr, lc);
MPI_Type_free(&col_type);
```

---

## Summary

| Function | Purpose |
|---|---|
| `MPI_Cart_create` | Create Cartesian topology communicator |
| `MPI_Cart_coords` | Rank → coordinates |
| `MPI_Cart_rank` | Coordinates → rank |
| `MPI_Cart_shift` | Get neighbor ranks (handles `MPI_PROC_NULL` at boundaries) |
| `MPI_Cart_sub` | Slice grid into row/column/hyperplane communicators |
| `MPI_Dims_create` | Auto-compute balanced grid dimensions |
| `MPI_Dist_graph_create_adjacent` | Irregular/sparse neighbor graph |
| `MPI_Neighbor_alltoall` | Exchange with graph neighbors |
| `MPI_PROC_NULL` | Null rank; absorbs sends, returns empty receives |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
