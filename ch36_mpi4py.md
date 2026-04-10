# Chapter 36: mpi4py — MPI for Python

## 36.1 What mpi4py Is

mpi4py provides Python bindings for MPI by wrapping the C MPI ABI directly.
Unlike subprocess-based parallelism, mpi4py:

- Maps MPI communicators, datatypes, and requests to Python objects
- Supports the full MPI standard (point-to-point, collectives, RMA, sessions)
- Achieves near-zero-copy for NumPy arrays via Python's buffer protocol
- Shares communicators across C/Fortran/Python code in the same job

```
Python application
    │
    └── mpi4py (Cython bindings → MPI C ABI)
            │
            └── Open MPI / MPICH / Cray MPICH (same .so as C programs)
```

---

## 36.2 Installation

```bash
# Install with pip (uses system MPI)
pip install mpi4py

# Specify which MPI to use (multiple MPIs on same machine)
MPICC=mpicc pip install mpi4py

# Conda
conda install -c conda-forge mpi4py openmpi

# From source (for custom builds)
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py
python setup.py build --mpicc=$(which mpicc)
python setup.py install
```

Verify:
```bash
mpiexec -n 4 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

---

## 36.3 Basic Usage

### Hello World

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

print(f"Rank {rank}/{size} on {name}")

MPI.Finalize()  # optional — called automatically on exit
```

```bash
mpiexec -n 4 python hello.py
```

### Point-to-Point: Python Objects (Pickle)

Lowercase methods use Python's pickle protocol — convenient but slow for large data:

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {"values": [1, 2, 3], "label": "from rank 0"}
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    print(f"Rank 1 received: {data}")
```

### Point-to-Point: NumPy Arrays (Buffer Protocol)

Uppercase methods use the buffer protocol — zero-copy for NumPy, matches C MPI speed:

```python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N = 1_000_000
buf = np.zeros(N, dtype=np.float64)

if rank == 0:
    buf[:] = np.arange(N, dtype=np.float64)
    comm.Send([buf, MPI.DOUBLE], dest=1, tag=0)
elif rank == 1:
    comm.Recv([buf, MPI.DOUBLE], source=0, tag=0)
    print(f"Rank 1 received sum: {buf.sum():.1f}")
```

The `[buf, MPI.DOUBLE]` notation passes the buffer and explicit MPI datatype.
Alternatively, mpi4py can infer the type from the NumPy dtype:

```python
comm.Send(buf, dest=1, tag=0)   # type inferred from buf.dtype
comm.Recv(buf, source=0, tag=0)
```

---

## 36.4 Lowercase vs. Uppercase Methods

| Style | Methods | Data | Serialization | Performance |
|---|---|---|---|---|
| Lowercase | `send`, `recv`, `bcast`, `scatter`, `gather` | Any Python object | Pickle | Slow (serialization overhead) |
| Uppercase | `Send`, `Recv`, `Bcast`, `Scatter`, `Gather` | Buffer protocol objects (NumPy, ctypes, bytearray) | None (zero-copy) | Fast (matches C MPI) |

Always use uppercase methods for numerical data.

---

## 36.5 Collectives

### Broadcast

```python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Pickle broadcast (small data, any type)
if rank == 0:
    config = {"lr": 0.001, "epochs": 100}
else:
    config = None
config = comm.bcast(config, root=0)

# Buffer broadcast (large arrays)
buf = np.zeros(1_000_000, dtype=np.float64)
if rank == 0:
    buf[:] = 1.0
comm.Bcast([buf, MPI.DOUBLE], root=0)
```

### Reduce and Allreduce

```python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Scalar allreduce (pickle path)
local_sum = rank * 1.0
global_sum = comm.allreduce(local_sum, op=MPI.SUM)
if rank == 0:
    print(f"Global sum: {global_sum}")  # n*(n-1)/2

# Array allreduce (buffer path — fast)
local_arr = np.full(1000, fill_value=float(rank), dtype=np.float64)
global_arr = np.empty_like(local_arr)
comm.Allreduce([local_arr, MPI.DOUBLE],
               [global_arr, MPI.DOUBLE],
               op=MPI.SUM)

# In-place
comm.Allreduce(MPI.IN_PLACE, [local_arr, MPI.DOUBLE], op=MPI.SUM)
```

### Scatter and Gather

```python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CHUNK = 100

# Scatter: rank 0 distributes chunks to all ranks
if rank == 0:
    sendbuf = np.arange(size * CHUNK, dtype=np.float64)
else:
    sendbuf = None
recvbuf = np.empty(CHUNK, dtype=np.float64)
comm.Scatter(sendbuf, recvbuf, root=0)

# Gather: collect results back to rank 0
recvbuf *= 2.0  # local computation
if rank == 0:
    result = np.empty(size * CHUNK, dtype=np.float64)
else:
    result = None
comm.Gather(recvbuf, result, root=0)

if rank == 0:
    print(f"Gathered: {result[:5]}")  # [0. 2. 4. 6. 8.]

# Allgather — all ranks receive the full result
all_data = np.empty(size * CHUNK, dtype=np.float64)
comm.Allgather(recvbuf, all_data)
```

### Scatterv / Gatherv (variable counts)

```python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Unequal distribution: rank i gets i+1 elements
counts = [i + 1 for i in range(size)]          # [1, 2, 3, 4] for 4 ranks
displs = [sum(counts[:i]) for i in range(size)] # [0, 1, 3, 6]
total  = sum(counts)

if rank == 0:
    sendbuf = np.arange(total, dtype=np.float64)
else:
    sendbuf = None
recvbuf = np.empty(counts[rank], dtype=np.float64)

comm.Scatterv([sendbuf, counts, displs, MPI.DOUBLE], recvbuf, root=0)
print(f"Rank {rank} received {recvbuf}")
```

---

## 36.6 Non-Blocking Communication

```python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 100_000
send_buf = np.ones(N, dtype=np.float64) * rank
recv_buf = np.empty(N, dtype=np.float64)

left  = (rank - 1) % size
right = (rank + 1) % size

# Non-blocking send/recv — halo exchange pattern
req_send = comm.Isend([send_buf, MPI.DOUBLE], dest=right, tag=0)
req_recv = comm.Irecv([recv_buf, MPI.DOUBLE], source=left, tag=0)

# Overlap with local computation
local_work = np.sum(send_buf ** 2)

# Wait for communication to complete
MPI.Request.Waitall([req_send, req_recv])

print(f"Rank {rank}: received from rank {left}, sum={recv_buf.sum():.0f}")
```

---

## 36.7 Communicator Operations

```python
from mpi4py import MPI

world = MPI.COMM_WORLD
rank  = world.Get_rank()

# Split into even/odd groups
color = rank % 2
sub_comm = world.Split(color, rank)
sub_rank = sub_comm.Get_rank()
sub_size = sub_comm.Get_size()
print(f"World rank {rank}: sub-comm color={color} rank={sub_rank}/{sub_size}")

# Node-local communicator (shared memory)
node_comm = world.Split_type(MPI.COMM_TYPE_SHARED, rank)
local_rank = node_comm.Get_rank()
print(f"World rank {rank}: local rank = {local_rank}")

# Library-safe duplicate
lib_comm = world.Dup()
# Use lib_comm for library operations; world for application

sub_comm.Free()
node_comm.Free()
lib_comm.Free()
```

---

## 36.8 One-Sided Communication (RMA)

```python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Allocate a shared window
N = 1000
win_mem = MPI.Alloc_mem(N * 8, MPI.INFO_NULL)   # 8 bytes = sizeof(double)
buf = np.frombuffer(win_mem, dtype=np.float64)
buf[:] = float(rank)

win = MPI.Win.Create(win_mem, disp_unit=8, info=MPI.INFO_NULL, comm=comm)

win.Fence(0)   # open epoch

if rank == 0:
    target = np.empty(N, dtype=np.float64)
    win.Get([target, N, MPI.DOUBLE], 1)  # get rank 1's window

win.Fence(0)   # close epoch

if rank == 0:
    print(f"Rank 0 fetched from rank 1: {target[:3]}")

win.Free()
MPI.Free_mem(win_mem)
```

---

## 36.9 Interoperability with C MPI Libraries

mpi4py exposes the raw MPI communicator handle, enabling interop with C extensions:

```python
from mpi4py import MPI
import ctypes

comm = MPI.COMM_WORLD

# Get the raw MPI_Comm handle (integer on most platforms)
handle = comm.py2f()  # Fortran-style integer handle (portable)

# Or get the C handle directly (platform-specific)
# comm_ptr = comm.handle  # MPI_Comm as ctypes integer

# Pass to a C library function via ctypes
mylib = ctypes.CDLL("./libmyapp.so")
# mylib.my_mpi_function.restype = None
# mylib.my_mpi_function.argtypes = [ctypes.c_int]
# mylib.my_mpi_function(handle)
```

In the C library:

```c
/* libmyapp.c — called from Python via ctypes */
#include <mpi.h>

void my_mpi_function(MPI_Fint comm_f)
{
    MPI_Comm comm = MPI_Comm_f2c(comm_f);
    int rank;
    MPI_Comm_rank(comm, &rank);
    /* Use comm normally */
}
```

---

## 36.10 mpi4py with NumPy: Zero-Copy Patterns

For scientific computing, always use the uppercase buffer-protocol interface with
contiguous NumPy arrays:

```python
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure arrays are contiguous (C order) before passing to MPI
arr = np.ascontiguousarray(some_array, dtype=np.float64)

# In-place Allreduce (true zero-copy)
comm.Allreduce(MPI.IN_PLACE, arr, op=MPI.SUM)

# Non-contiguous arrays must be packed first
col = arr[:, 0]   # column — not contiguous in memory
col_c = np.ascontiguousarray(col)
comm.Bcast(col_c, root=0)

# Structured dtype for custom MPI_Datatype equivalent
dt = np.dtype([('x', np.float64), ('y', np.float64), ('flag', np.int32)])
pts = np.zeros(1000, dtype=dt)
comm.Bcast(pts, root=0)   # mpi4py handles the structured type
```

---

## 36.11 Running mpi4py Programs

```bash
# Basic launch
mpiexec -n 4 python script.py

# With SLURM
srun --ntasks=64 python script.py

# Hybrid MPI + threading
export OMP_NUM_THREADS=4
srun --ntasks=16 --cpus-per-task=4 python hybrid.py

# With GPU support (Open MPI + CUDA)
export OMPI_MCA_opal_cuda_support=1
mpiexec -n 4 python gpu_script.py
```

### mpi4py with mpi4py.futures (task parallelism)

```python
from mpi4py.futures import MPIPoolExecutor
import numpy as np

def compute(x):
    return np.sum(x ** 2)

with MPIPoolExecutor() as executor:
    data = [np.random.randn(1000) for _ in range(100)]
    results = list(executor.map(compute, data))
    print(f"Total: {sum(results):.2f}")
```

```bash
# Launch: rank 0 is the master; other ranks are workers
mpiexec -n 5 python -m mpi4py.futures script.py
```

---

## 36.12 Performance Considerations

| Concern | Recommendation |
|---|---|
| Large arrays | Always use uppercase (buffer) methods; lowercase pickles everything |
| Contiguous memory | Use `np.ascontiguousarray()` before passing to MPI |
| Communication overhead | Batch small sends; avoid many small messages |
| Python GIL | GIL is held during MPI calls; thread-based overlap limited |
| Async overlap | Use non-blocking calls (`Isend`/`Irecv`) + `Waitall` for overlap |
| Initialization | `MPI.Init()` is called automatically on `from mpi4py import MPI` |

### Benchmark: mpi4py vs C MPI

```python
import numpy as np
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N = 10_000_000   # 80 MB
buf = np.ones(N, dtype=np.float64)

comm.Barrier()
t0 = time.perf_counter()
for _ in range(10):
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
comm.Barrier()
t1 = time.perf_counter()

if rank == 0:
    bw = 10 * N * 8 / (t1 - t0) / 1e9
    print(f"Allreduce bandwidth: {bw:.2f} GB/s")
# Achieves >95% of C MPI performance for large arrays
```

---

## Summary

| Topic | Key Points |
|---|---|
| Installation | `pip install mpi4py`; uses system MPI via `MPICC` |
| lowercase methods | Pickle serialization; works for any Python object; slow |
| Uppercase methods | Buffer protocol; zero-copy for NumPy; fast (matches C MPI) |
| NumPy syntax | `comm.Send([buf, MPI.DOUBLE], dest=...)` or inferred from dtype |
| Collectives | `Bcast`, `Scatter/Gather`, `Allreduce`, `Alltoall` all have buffer variants |
| Non-blocking | `Isend`/`Irecv` return `Request` objects; `Request.Waitall([...])` |
| Communicators | `comm.Split`, `comm.Split_type`, `comm.Dup` — same semantics as C |
| C interop | `comm.py2f()` returns Fortran handle; C side uses `MPI_Comm_f2c()` |
| Futures | `mpi4py.futures.MPIPoolExecutor` for task-parallel workloads |
| Performance | Large arrays hit >95% of C MPI speed; always use uppercase + contiguous arrays |
