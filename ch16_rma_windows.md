# Chapter 16: Windows & Memory Models

## 16.1 One-Sided Communication Overview

Point-to-point and collective operations require both sender and receiver to call
MPI functions. **One-sided communication** (Remote Memory Access, RMA) allows one
process to read or write the memory of another process without the remote process
taking any explicit action:

```
Two-sided (P2P):            One-sided (RMA):
 Rank 0: MPI_Send ──────►  Rank 0: MPI_Put ──────► Rank 1 memory
 Rank 1: MPI_Recv           Rank 1: (sleeping)
```

RMA maps closely to RDMA (Remote Direct Memory Access) hardware capabilities
available on InfiniBand, Cray Slingshot, and other HPC interconnects. When the
implementation maps MPI RMA directly to hardware RDMA, the remote process truly
does not participate in the transfer — the NIC handles it.

Use cases for RMA:
- Dynamic or irregular access patterns where the initiator determines what data
  to fetch/store.
- Lock-free data structures shared across processes.
- Accumulation patterns where many processes contribute to a distributed array.
- Emulating shared memory across nodes.

---

## 16.2 Windows

A **window** (`MPI_Win`) is the fundamental RMA object. A window exposes a region
of process memory (or shared memory) to other processes in a communicator.

Three window creation functions cover different use cases:

### MPI_Win_create — Expose existing memory

```c
int MPI_Win_create(void *base, MPI_Aint size, int disp_unit,
                   MPI_Info info, MPI_Comm comm, MPI_Win *win);
```

| Argument | Meaning |
|---|---|
| `base` | Pointer to the memory region to expose |
| `size` | Size in bytes of the exposed region |
| `disp_unit` | Unit for displacements (bytes per unit; typically `sizeof(element)`) |
| `info` | Hints (use `MPI_INFO_NULL` for defaults) |
| `comm` | Communicator; all processes must call this collectively |
| `win` | Output window handle |

```c
double *local_buf = malloc(N * sizeof(double));

MPI_Win win;
MPI_Win_create(local_buf, N * sizeof(double), sizeof(double),
               MPI_INFO_NULL, MPI_COMM_WORLD, &win);

/* ... RMA operations ... */

MPI_Win_free(&win);
free(local_buf);
```

After `MPI_Win_create`, other processes can access `local_buf` using RMA operations,
addressing elements by displacement in units of `disp_unit`.

### MPI_Win_allocate — Let MPI allocate

```c
int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info,
                     MPI_Comm comm, void *baseptr, MPI_Win *win);
```

MPI allocates the memory and exposes it as a window. The memory may be placed in
special RDMA-registered buffers, avoiding the registration overhead of `MPI_Win_create`.

```c
double *local_buf;
MPI_Win win;
MPI_Win_allocate(N * sizeof(double), sizeof(double), MPI_INFO_NULL,
                 MPI_COMM_WORLD, &local_buf, &win);

/* local_buf can be used as regular memory AND as an RMA target */
for (int i = 0; i < N; i++) local_buf[i] = (double)i;

/* ... RMA operations ... */

MPI_Win_free(&win);   /* also frees local_buf */
```

### MPI_Win_create_dynamic — Register memory on the fly

```c
int MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win);
```

Creates a window with no initial memory. Memory regions are attached/detached
later with `MPI_Win_attach` / `MPI_Win_detach`. Useful for dynamic data structures
where the set of accessible memory changes at runtime.

```c
MPI_Win win;
MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win);

double *buf = malloc(N * sizeof(double));
MPI_Win_attach(win, buf, N * sizeof(double));

/* Other processes can now access buf via RMA */
/* Must share the base address with them first */
MPI_Aint my_base;
MPI_Get_address(buf, &my_base);
MPI_Bcast(&my_base, 1, MPI_AINT, 0, MPI_COMM_WORLD);

/* ... RMA using my_base as displacement offset ... */

MPI_Win_detach(win, buf);
free(buf);
MPI_Win_free(&win);
```

With dynamic windows, displacements in RMA calls are absolute memory addresses
(`MPI_Get_address` values), not relative offsets. The effective `disp_unit` is fixed
at 1 by the standard — there is no `disp_unit` parameter on `MPI_Win_create_dynamic`.

---

## 16.3 Shared Memory Windows

`MPI_Win_allocate_shared` creates a window where all processes in the communicator
share the **same physical memory**. This is only useful within a single node
(use `MPI_COMM_TYPE_SHARED` to restrict the communicator):

```c
/* Create a node-local shared memory region */
MPI_Comm shared_comm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                    MPI_INFO_NULL, &shared_comm);

int local_rank;
MPI_Comm_rank(shared_comm, &local_rank);

/* Allocate: each process gets a slice */
MPI_Aint local_size = local_rank == 0 ? TOTAL_SIZE : 0;
MPI_Win shared_win;
double *local_ptr;
MPI_Win_allocate_shared(local_size, sizeof(double), MPI_INFO_NULL,
                        shared_comm, &local_ptr, &shared_win);

/* Any process can get a pointer to another process's slice */
MPI_Aint size_of_rank0; int disp_unit;
double *rank0_ptr;
MPI_Win_shared_query(shared_win, 0, &size_of_rank0, &disp_unit,
                     &rank0_ptr);

/* Now rank0_ptr is a valid pointer to rank 0's memory on this node */
/* Reads/writes to rank0_ptr are real shared memory — no MPI calls needed */
```

This is the MPI replacement for OpenMP shared arrays within a node. Combine with
OpenMP for efficient intra-node parallelism while using standard MPI across nodes.

---

## 16.4 Memory Model: Unified vs. Separate

MPI defines two memory models for RMA operations:

### Unified Memory Model

All processes see a consistent view of window memory at all times. Writes from one
process are immediately visible to others (subject to hardware cache coherency).
This model corresponds to systems with coherent RDMA hardware (most modern HPC
networks).

Query: `MPI_WIN_MODEL` attribute on the window.

```c
int *model;
int flag;
MPI_Win_get_attr(win, MPI_WIN_MODEL, &model, &flag);
if (*model == MPI_WIN_UNIFIED) {
    printf("Unified memory model — loads/stores are immediately visible\n");
}
```

### Separate Memory Model

Each process has its own "public window copy." Writes are not visible until an
explicit synchronization call is made. Requires explicit synchronization to flush
writes and update the public copy.

Most HPC networks implement the unified model. The separate model exists for
completeness and portability to systems with weaker coherency.

**In practice**: write code that works under the separate model (uses explicit
synchronization) — it will work correctly on unified systems too. Code that assumes
unified behavior may fail on some hardware.

---

## 16.5 MPI_Win_free and Cleanup

```c
int MPI_Win_free(MPI_Win *win);
```

Collective: all processes must call it. Frees the window and (for `MPI_Win_allocate`)
the associated memory. After this call, `win = MPI_WIN_NULL`.

Window memory created with `MPI_Win_create` must be freed separately by the user.

---

## Summary

| Function | Window Type | Notes |
|---|---|---|
| `MPI_Win_create` | User-allocated memory | Most general; may need RDMA registration |
| `MPI_Win_allocate` | MPI-allocated memory | Preferred: MPI may optimize placement |
| `MPI_Win_create_dynamic` | Dynamic memory | Attach/detach at runtime; uses absolute addresses |
| `MPI_Win_allocate_shared` | Shared memory | Same physical memory; node-local only |
| `MPI_Win_free` | — | Collective cleanup |

**Key parameters**: `disp_unit` sets the displacement unit; `size` in bytes.
`MPI_Win_create_dynamic` has no `disp_unit` parameter — the standard fixes its
effective unit at 1 (byte addressing) and uses absolute addresses.

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
