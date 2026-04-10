# Chapter 23: Partitioned Communication — MPI 4.0

## 23.1 The Problem Partitioned Communication Solves

In GPU computing and producer/consumer pipelines, data arrives in chunks — a GPU
kernel fills part of a buffer while the MPI runtime could already be sending the
ready portion. Standard MPI sends require the entire buffer to be ready before
initiating the transfer:

```
Standard Send:
  [GPU fills part 0] [GPU fills part 1] [GPU fills part 2] | MPI_Send(whole buffer)
  
Partitioned Send:
  [GPU fills part 0] → MPI_Pready(0) → network sends part 0 immediately
  [GPU fills part 1] → MPI_Pready(1) → network sends part 1
  [GPU fills part 2] → MPI_Pready(2) → network sends part 2
```

**Partitioned communication** (MPI 4.0) allows the sender to independently signal
when each portion (partition) of a message is ready. The receiver can similarly
process partitions as they arrive, before the entire message is complete.

---

## 23.2 Partitioned Send and Receive Initialization

Partitioned communication uses a persistent-request-like lifecycle: initialize once,
start/complete per iteration.

```c
int MPI_Psend_init(const void *buf, int partitions, MPI_Count count,
                   MPI_Datatype datatype, int dest, int tag,
                   MPI_Comm comm, MPI_Info info, MPI_Request *request);

int MPI_Precv_init(void *buf, int partitions, MPI_Count count,
                   MPI_Datatype datatype, int source, int tag,
                   MPI_Comm comm, MPI_Info info, MPI_Request *request);
```

| Argument | Meaning |
|---|---|
| `buf` | Buffer for the entire message |
| `partitions` | Number of partitions |
| `count` | Elements per partition |
| `datatype` | Element type |
| Total message | `partitions × count` elements |

The total buffer size is `partitions * count * sizeof(datatype)`. Each partition
occupies a contiguous slice of the buffer.

---

## 23.3 Signaling Partition Readiness

```c
/* Sender: signal that partition i is ready to send */
int MPI_Pready(int partition, MPI_Request *request);

/* Sender: signal that a range of partitions is ready */
int MPI_Pready_range(int partition_low, int partition_high, MPI_Request *request);

/* Sender: signal an arbitrary set of partitions */
int MPI_Pready_list(int length, int partition_list[], MPI_Request *request);
```

```c
/* Receiver: check if partition i has arrived */
int MPI_Parrived(MPI_Request *request, int partition, int *flag);
```

---

## 23.4 Complete Lifecycle Example

### Sender Side

```c
#define NPARTITIONS 4
#define ELEMS_PER_PART 1024

double buf[NPARTITIONS * ELEMS_PER_PART];
MPI_Request send_req;

/* Initialize once */
MPI_Psend_init(buf, NPARTITIONS, ELEMS_PER_PART, MPI_DOUBLE,
               dest_rank, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &send_req);

for (int step = 0; step < NSTEPS; step++) {
    MPI_Start(&send_req);  /* begin the partitioned send epoch */

    for (int p = 0; p < NPARTITIONS; p++) {
        /* Fill partition p (could be a GPU kernel, async I/O, etc.) */
        double *part_buf = buf + p * ELEMS_PER_PART;
        fill_partition(part_buf, ELEMS_PER_PART, p, step);

        /* Signal: this partition is ready */
        MPI_Pready(p, &send_req);
    }

    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
    /* All partitions sent; send_req is now inactive */
}

MPI_Request_free(&send_req);
```

### Receiver Side

```c
double buf[NPARTITIONS * ELEMS_PER_PART];
MPI_Request recv_req;

/* Initialize once */
MPI_Precv_init(buf, NPARTITIONS, ELEMS_PER_PART, MPI_DOUBLE,
               src_rank, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &recv_req);

for (int step = 0; step < NSTEPS; step++) {
    MPI_Start(&recv_req);  /* begin the partitioned receive epoch */

    /* Process partitions as they arrive, without waiting for all */
    int completed = 0;
    while (completed < NPARTITIONS) {
        for (int p = 0; p < NPARTITIONS; p++) {
            int arrived;
            MPI_Parrived(&recv_req, p, &arrived);
            if (arrived) {
                double *part_buf = buf + p * ELEMS_PER_PART;
                process_partition(part_buf, ELEMS_PER_PART, p);
                completed++;
                /* mark p as processed so we don't check it again */
            }
        }
    }

    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
}

MPI_Request_free(&recv_req);
```

---

## 23.5 GPU Pipeline Pattern

The canonical use case is a GPU-to-CPU or GPU-to-GPU pipeline:

```cpp
/* CUDA example: GPU fills partitions, CPU signals readiness */

cudaStream_t streams[NPARTITIONS];
for (int p = 0; p < NPARTITIONS; p++) cudaStreamCreate(&streams[p]);

MPI_Request send_req;
MPI_Psend_init(d_buf, NPARTITIONS, ELEMS_PER_PART, MPI_DOUBLE,
               dest, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &send_req);

for (int step = 0; step < NSTEPS; step++) {
    MPI_Start(&send_req);

    for (int p = 0; p < NPARTITIONS; p++) {
        /* Launch GPU kernel to fill partition p in its own stream */
        fill_kernel<<<GRID, BLOCK, 0, streams[p]>>>(d_buf + p*ELEMS_PER_PART, p, step);

        /* Register a CUDA callback to call MPI_Pready when kernel completes */
        /* (exact API depends on CUDA version and MPI GPU-aware implementation) */
        register_pready_callback(streams[p], send_req, p);
    }

    MPI_Wait(&send_req, MPI_STATUS_IGNORE);
}
```

In implementations with GPU-aware MPI and hardware RDMA, the network can begin
transmitting partition `p` while the GPU is still computing partition `p+1`.
This hides both GPU compute latency and network latency.

---

## 23.6 Partitioned vs. Persistent vs. Non-Blocking

| Feature | Non-blocking | Persistent | Partitioned |
|---|---|---|---|
| Init cost | Per call | Once | Once |
| Buffer must be ready at Start? | Yes (Isend) | Yes (Start) | **No** |
| Partial readiness | No | No | **Yes** |
| Receiver can process early? | No | No | **Yes** |
| GPU pipeline support | Limited | Limited | **Yes** |

Partitioned communication does not replace non-blocking or persistent requests.
Use it specifically when:
- Buffer is filled incrementally by an async producer (GPU, I/O, other threads)
- Receiver can process data as it arrives (streaming computation)
- The network hardware supports cut-through delivery of partial messages

---

## 23.7 MPI_Info Hints for Partitioned Communication

```c
MPI_Info info;
MPI_Info_create(&info);

/* Hint: partitions will be signaled in order */
MPI_Info_set(info, "partitioned_comm_ordered", "true");

/* Hint: GPU memory buffer (if MPI supports GPU-aware memory) */
MPI_Info_set(info, "buffer_type", "cuda_device");

MPI_Psend_init(buf, NPARTITIONS, ELEMS_PER_PART, MPI_DOUBLE,
               dest, tag, comm, info, &req);
MPI_Info_free(&info);
```

Info keys for partitioned communication are implementation-specific as of MPI 4.0.
Check your MPI vendor's documentation for supported keys.

---

## 23.8 Availability

Partitioned communication is defined in MPI 4.0 (2021). Implementation support:

| Implementation | Status (2025) |
|---|---|
| MPICH 4.x | Full support |
| Open MPI 5.x | Supported |
| Cray MPICH | Supported on HPE Slingshot |
| Intel MPI | Supported in recent versions |

Check with:

```c
#if MPI_VERSION < 4
#error "Partitioned communication requires MPI 4.0"
#endif
```

---

## Summary

| Function | Purpose |
|---|---|
| `MPI_Psend_init` | Initialize partitioned send (inactive) |
| `MPI_Precv_init` | Initialize partitioned receive (inactive) |
| `MPI_Start` | Begin send/receive epoch |
| `MPI_Pready(p, req)` | Signal partition `p` is ready to send |
| `MPI_Pready_range(lo, hi, req)` | Signal range of partitions |
| `MPI_Parrived(req, p, &flag)` | Check if partition `p` has arrived |
| `MPI_Wait` | Complete epoch; request becomes inactive |
| `MPI_Request_free` | Free after all iterations done |

**Key insight**: partitioned communication is to persistent communication as
non-blocking is to blocking — but the "completeness" signal is per-partition, not
per-message. It enables genuine pipeline parallelism between producers, the network,
and consumers.
