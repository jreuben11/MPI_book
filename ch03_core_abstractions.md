# Chapter 3: Core Abstractions

## 3.1 Communicators

A **communicator** is the fundamental scoping mechanism in MPI. Every communication
operation — send, receive, collective, one-sided — happens within a communicator.

A communicator encapsulates:
- A **group**: an ordered set of processes, each assigned a rank (0 to size−1).
- A **context**: an opaque integer that prevents messages in one communicator from
  being received as messages in another communicator, even if they have the same
  tag and endpoint. This is what makes library isolation possible.
- An optional **topology**: Cartesian or graph structure (Chapter 15).
- Optional **attributes**: user-attached key-value pairs (Chapter 14).

### Predefined Communicators

| Communicator | Meaning |
|---|---|
| `MPI_COMM_WORLD` | All processes launched by `mpiexec` |
| `MPI_COMM_SELF` | A communicator containing only the calling process |
| `MPI_COMM_NULL` | Null handle; returned when operations produce no communicator |

```c
MPI_Comm comm = MPI_COMM_WORLD;

int rank, size;
MPI_Comm_rank(comm, &rank);   /* my rank in comm: 0 .. size-1 */
MPI_Comm_size(comm, &size);   /* number of processes in comm  */
```

### Why Context Matters

Suppose a numerical library and your application both call `MPI_Bcast` on
`MPI_COMM_WORLD`. Collectives have no tag — they are matched by communicator and call
order. Without context isolation, their collective calls could cross. In practice,
well-written libraries create their own communicator with `MPI_Comm_dup` so their
operations live in a separate context. Chapter 14 covers communicator creation.

---

## 3.2 Groups

A **group** (`MPI_Group`) is an ordered list of processes. Communicators contain
groups, but groups have no communication capability on their own. You use groups to
construct new communicators.

```c
MPI_Group world_group;
MPI_Comm_group(MPI_COMM_WORLD, &world_group);

/* Get size and rank within the group */
int grank, gsize;
MPI_Group_rank(world_group, &grank);
MPI_Group_size(world_group, &gsize);

/* Create a subgroup from ranks 0,2,4 */
int ranks[3] = {0, 2, 4};
MPI_Group sub_group;
MPI_Group_incl(world_group, 3, ranks, &sub_group);

/* Free groups when done — they are cheap but must be freed */
MPI_Group_free(&world_group);
MPI_Group_free(&sub_group);
```

Groups are most commonly used as stepping stones to new communicators. See Chapter 14.

---

## 3.3 Tags

Every point-to-point message carries a **tag** — a non-negative integer used to
distinguish messages from the same source. The receiver can specify a tag or use
`MPI_ANY_TAG` to accept any tag.

```c
/* Send with tag 42 */
MPI_Send(buf, count, MPI_INT, dest, 42, comm);

/* Receive only messages with tag 42 */
MPI_Recv(buf, count, MPI_INT, source, 42, comm, MPI_STATUS_IGNORE);

/* Receive any tag */
MPI_Recv(buf, count, MPI_INT, source, MPI_ANY_TAG, comm, &status);
int received_tag = status.MPI_TAG;
```

Valid tag range: 0 to `MPI_TAG_UB` (at least 32767; most implementations support
at least 2^30 - 1). Query the actual upper bound:

```c
int *tag_ub;
int flag;
MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &flag);
printf("Max tag: %d\n", *tag_ub);
```

MPI guarantees FIFO ordering within the same (source, tag, communicator) group: if a
sender sends two messages to the same destination with the same tag on the same
communicator, they arrive in the order they were sent. However, tags allow receivers
to selectively match messages — by specifying a particular tag, the receiver can
receive a later-sent message before an earlier one that has a different tag. Use
explicit tags in your protocol design to distinguish message types; do not rely on
ordering across different tag values.

### Tag Discipline

In large programs, define tags as named constants or an enum:

```c
typedef enum {
    TAG_HALO_X  = 100,
    TAG_HALO_Y  = 101,
    TAG_RESULT  = 200,
    TAG_CONTROL = 300,
} AppTag;
```

Avoid magic numbers scattered through the code. Colliding tags within the same
communicator cause messages to be matched incorrectly, producing silent data corruption.

---

## 3.4 MPI Handles

MPI objects are accessed through opaque **handles**. The handle is typically an
integer or pointer typedef — the internal representation is implementation-defined.

| Handle Type | Object | Null Value |
|---|---|---|
| `MPI_Comm` | Communicator | `MPI_COMM_NULL` |
| `MPI_Group` | Group of processes | `MPI_GROUP_NULL` |
| `MPI_Datatype` | Data layout descriptor | `MPI_DATATYPE_NULL` |
| `MPI_Op` | Reduction operation | `MPI_OP_NULL` |
| `MPI_Request` | Non-blocking operation token | `MPI_REQUEST_NULL` |
| `MPI_Win` | RMA window | `MPI_WIN_NULL` |
| `MPI_File` | File handle | `MPI_FILE_NULL` |
| `MPI_Session` | Session (MPI 4.0) | `MPI_SESSION_NULL` |
| `MPI_Info` | Key-value hint store | `MPI_INFO_NULL` |
| `MPI_Errhandler` | Error handler | `MPI_ERRHANDLER_NULL` |

### Handle Lifecycle Rules

1. **Predefined handles** (e.g., `MPI_COMM_WORLD`, `MPI_INT`, `MPI_SUM`) are always
   valid; do not free them.
2. **User-created handles** must be freed with the corresponding `_free` function:
   - `MPI_Comm_free(&comm)`
   - `MPI_Type_free(&type)`
   - `MPI_Op_free(&op)`
   - `MPI_Win_free(&win)`
   - `MPI_File_close(&fh)`
   - `MPI_Request_free(&req)` (use cautiously — see Chapter 6)
3. After freeing, the handle is set to its null value.
4. Handles are **local** — they are not the same across processes. Process 0's
   `MPI_Comm` handle for a communicator has a different numeric value than Process 1's
   handle for the same communicator.

---

## 3.5 MPI_Status

`MPI_Status` is a struct returned by receive operations. It carries metadata about
the received message:

```c
MPI_Status status;
MPI_Recv(buf, count, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
         MPI_COMM_WORLD, &status);

int source = status.MPI_SOURCE;    /* actual sender rank */
int tag    = status.MPI_TAG;       /* actual tag used    */
int error  = status.MPI_ERROR;     /* error code         */

/* Query the actual number of elements received */
int recv_count;
MPI_Get_count(&status, MPI_INT, &recv_count);
```

The three fields `MPI_SOURCE`, `MPI_TAG`, and `MPI_ERROR` are the only fields of
`MPI_Status` directly accessible. Other fields in the struct are implementation-defined
and must not be read.

### MPI_STATUS_IGNORE

When you do not need source, tag, or count information:

```c
MPI_Recv(buf, count, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
```

This avoids allocating a `MPI_Status` on the stack and signals to the implementation
that it can skip filling in the struct. Use it whenever you do not actually inspect the
status.

### MPI_STATUSES_IGNORE

For `MPI_Waitall`, `MPI_Testall`, etc., which take an array of statuses:

```c
MPI_Waitall(count, reqs, MPI_STATUSES_IGNORE);
```

---

## 3.6 Buffer Ownership Rules

MPI imposes strict rules on when you may read or write the buffer passed to a
communication function:

### Blocking Operations

For blocking send/receive, the buffer is safe to reuse immediately after the function
returns:

```c
MPI_Send(buf, count, MPI_INT, dest, tag, comm);
/* buf can be modified here — send is complete */

MPI_Recv(buf, count, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
/* buf contains received data here */
```

### Non-Blocking Operations

For non-blocking operations, the buffer must not be touched until the request is
completed (via `MPI_Wait` or `MPI_Test`):

```c
MPI_Request req;
MPI_Isend(buf, count, MPI_INT, dest, tag, comm, &req);

/* DO NOT modify buf here — the send may still be in progress */

MPI_Wait(&req, MPI_STATUS_IGNORE);
/* buf is safe to reuse now */
```

Violating buffer ownership is undefined behavior. The bugs are typically intermittent
and load-dependent, making them among the hardest to debug in MPI programs.

### RMA (One-Sided)

RMA buffer rules are more complex and depend on synchronization epochs. See Chapter 16.

---

## 3.7 In-Place Operations

Several collectives support `MPI_IN_PLACE` to avoid allocating a separate buffer:

```c
/* Allreduce in place — all ranks use their own buffer as both input and output */
int local_sum = compute_local_sum();
MPI_Allreduce(MPI_IN_PLACE, &local_sum, 1, MPI_INT, MPI_SUM,
              MPI_COMM_WORLD);
/* local_sum now holds the global sum on all ranks */
```

Rules for `MPI_IN_PLACE`:

- Not all functions support it (check the standard for each function).
- For `MPI_Reduce`: only the root process may pass `MPI_IN_PLACE` as sendbuf.
- For `MPI_Scatter` / `MPI_Gather`: the root passes `MPI_IN_PLACE` as recvbuf
  (for scatter) or sendbuf (for gather) — the semantics differ by function.
- `MPI_IN_PLACE` cannot be used in point-to-point operations.

---

## 3.8 MPI_Info

`MPI_Info` is a key-value store for passing hints to MPI operations. Hints are
advisory — the implementation is free to ignore them. They never affect correctness,
only performance.

```c
MPI_Info info;
MPI_Info_create(&info);

/* Hint: we are doing striped I/O across 16 storage targets */
MPI_Info_set(info, "striping_factor", "16");
MPI_Info_set(info, "striping_unit", "1048576");

/* Pass info when opening a file */
MPI_File_open(MPI_COMM_WORLD, "data.bin",
              MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);

MPI_Info_free(&info);
```

You can query what hints the implementation actually honored:

```c
MPI_Info info_used;
MPI_File_get_info(fh, &info_used);

char val[256];
int flag;
/* MPI_Info_get is deprecated in MPI 4.0; use MPI_Info_get_string instead */
MPI_Info_get(info_used, "striping_factor", 256, val, &flag);
if (flag) printf("striping_factor accepted: %s\n", val);
MPI_Info_free(&info_used);
```

Use `MPI_INFO_NULL` when you have no hints:

```c
MPI_File_open(MPI_COMM_WORLD, "data.bin",
              MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
```

MPI 5.0 standardized a broader set of recognized info keys. Previously, keys were
implementation-specific. See Appendix B for the standardized key list.

---

## 3.9 Wildcards: MPI_ANY_SOURCE and MPI_ANY_TAG

In receive operations, two wildcards allow flexible matching:

```c
/* Accept a message from any rank with tag 42 */
MPI_Recv(buf, count, MPI_INT, MPI_ANY_SOURCE, 42, comm, &status);
int actual_sender = status.MPI_SOURCE;

/* Accept a message with any tag from rank 5 */
MPI_Recv(buf, count, MPI_INT, 5, MPI_ANY_TAG, comm, &status);
int actual_tag = status.MPI_TAG;

/* Accept anything */
MPI_Recv(buf, count, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
```

`MPI_ANY_SOURCE` is `−1` in most implementations; `MPI_ANY_TAG` is also typically `−1`.
Do not hard-code these values — always use the named constants.

Wildcards have a performance cost on high-speed networks because the implementation
cannot pre-post the receive to a specific hardware queue. Use specific source/tag when
known.

---

## Summary

| Concept | Key Points |
|---|---|
| Communicator | Scopes communication; context prevents cross-talk between libraries |
| Group | Ordered process list; used to construct communicators |
| Tag | Message discriminator; use named constants; 0 to MPI_TAG_UB |
| Handles | Opaque; local to each process; free user-created handles |
| `MPI_Status` | Source, tag, error; use `MPI_Get_count` for actual receive count |
| `MPI_STATUS_IGNORE` | Use when status fields are not needed |
| Buffer ownership | Blocking: safe after return; non-blocking: safe after Wait/Test |
| `MPI_IN_PLACE` | Eliminates extra buffer in collectives; rules vary by function |
| `MPI_Info` | Advisory hints; never affects correctness |
| Wildcards | `MPI_ANY_SOURCE`, `MPI_ANY_TAG`; use named constants |

---

*© 2025 jreuben1. Licensed under [CC BY 4.0](LICENSE).*
