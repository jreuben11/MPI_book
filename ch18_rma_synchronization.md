# Chapter 18: RMA Synchronization

## 18.1 Why RMA Needs Explicit Synchronization

Unlike point-to-point operations where the `MPI_Wait` completion is well-defined,
RMA operations (`MPI_Put`, `MPI_Get`, `MPI_Accumulate`) return immediately with no
guarantee about when the data arrives at or departs from the remote process.

Synchronization calls define **epochs** — windows of time during which RMA operations
are valid and within which all in-flight operations are guaranteed to complete.

There are two synchronization models:
- **Active target**: both origin and target call synchronization functions.
- **Passive target**: only the origin calls synchronization; target is uninvolved.

---

## 18.2 Active Target: Fence

Fence is the simplest synchronization. All processes in the communicator call
`MPI_Win_fence` collectively, marking the end of one epoch and the beginning of the next.

```c
int MPI_Win_fence(int assert, MPI_Win win);
```

`assert` is a bitwise OR of zero or more of:
- `MPI_MODE_NOSTORE`: no local store to window memory since last fence
- `MPI_MODE_NOPUT`: no MPI_Put in this epoch
- `MPI_MODE_NOPRECEDE`: this fence does not complete a prior epoch
- `MPI_MODE_NOSUCCEED`: this fence does not begin a new epoch

All are hints to the implementation; passing `0` is always correct.

```c
/* Pattern: fill window, fence, get from neighbors, fence */

/* Epoch 0: initialize local window data */
for (int i = 0; i < N; i++) local_buf[i] = (double)rank;
MPI_Win_fence(0, win);   /* all fills complete; RMA epoch begins */

/* Epoch 1: RMA operations */
MPI_Get(recv_buf, N, MPI_DOUBLE, (rank+1) % size, 0, N, MPI_DOUBLE, win);
MPI_Put(send_buf, N, MPI_DOUBLE, (rank-1+size) % size, 0, N, MPI_DOUBLE, win);

MPI_Win_fence(0, win);   /* all Gets/Puts complete; safe to use recv_buf */

process_received_data(recv_buf);
```

### Fence Rules

- All processes must call `MPI_Win_fence` at the same points.
- Between two fences, a process may initiate RMA operations targeting any other
  process in the communicator.
- After the second fence returns, all RMA operations from the epoch are complete
  at both origin and target.

Fence is simple but imposes global synchronization. Use it for structured patterns
where all processes participate in every epoch.

---

## 18.3 Active Target: PSCW (Post/Start/Complete/Wait)

PSCW is a more fine-grained active-target synchronization. The origin and target
each call a different pair of functions, enabling overlapping epochs and avoiding
global barriers.

```
Target side:                    Origin side:
  MPI_Win_post(group_of_origins) ←──┐
                                     │ origin starts after post
  MPI_Win_wait()                 ◄──┼── MPI_Win_start(group_of_targets)
                                     │
                                     │   MPI_Put / MPI_Get ...
                                     │
  (wait returns when all RMA done) ◄─┘── MPI_Win_complete()
```

### Functions

```c
/* Target: expose window to a group of origins */
int MPI_Win_post(MPI_Group group, int assert, MPI_Win win);

/* Target: wait for all origins to complete their RMA */
int MPI_Win_wait(MPI_Win win);

/* Origin: declare intent to access a group of targets */
int MPI_Win_start(MPI_Group group, int assert, MPI_Win win);

/* Origin: signal completion of all RMA operations to targets */
int MPI_Win_complete(MPI_Win win);
```

```c
/* Example: rank 0 puts data into rank 1's window */

MPI_Group world_group, origin_group, target_group;
MPI_Comm_group(MPI_COMM_WORLD, &world_group);

int origin_ranks[1] = {0};
int target_ranks[1] = {1};
MPI_Group_incl(world_group, 1, origin_ranks, &origin_group);
MPI_Group_incl(world_group, 1, target_ranks, &target_group);

if (rank == 1) {
    /* Target: post exposure to rank 0 */
    MPI_Win_post(origin_group, 0, win);
    /* Do other work while rank 0 is writing */
    do_other_work();
    /* Wait for rank 0 to complete its writes */
    MPI_Win_wait(win);
    /* recv_buf now contains rank 0's data */
}

if (rank == 0) {
    /* Origin: start access epoch on rank 1 */
    MPI_Win_start(target_group, 0, win);
    /* Put data into rank 1's window */
    MPI_Put(local_data, N, MPI_DOUBLE, 1, 0, N, MPI_DOUBLE, win);
    /* Signal completion */
    MPI_Win_complete(win);
}

MPI_Group_free(&world_group);
MPI_Group_free(&origin_group);
MPI_Group_free(&target_group);
```

PSCW is more complex than Fence but avoids global synchronization. Targets can post
their windows early and overlap the wait with computation.

---

## 18.4 Passive Target: Lock / Unlock

Passive target synchronization allows an origin to access a target's window without
any action from the target process. This is the truest form of one-sided communication.

```c
/* Lock the target's window */
int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win);

/* Unlock: completes all RMA operations, releases lock */
int MPI_Win_unlock(int rank, MPI_Win win);
```

`lock_type`:
- `MPI_LOCK_EXCLUSIVE`: only one process may hold the lock; mutual exclusion.
- `MPI_LOCK_SHARED`: multiple processes may hold simultaneously; appropriate for
  non-conflicting operations such as reads and `MPI_Accumulate` calls.

```c
/* Rank 0 writes to rank 1's window without rank 1 doing anything */
if (rank == 0) {
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, win);

    MPI_Put(data, N, MPI_DOUBLE, 1, 0, N, MPI_DOUBLE, win);
    /* data is NOT guaranteed to be at rank 1 until Unlock */

    MPI_Win_unlock(1, win);
    /* After Unlock: Put is complete at rank 1 */
}
/* Rank 1 does not call any MPI functions during this period */
```

### MPI_Win_lock_all / MPI_Win_unlock_all

```c
/* Lock all processes simultaneously (shared lock) */
MPI_Win_lock_all(0, win);

/* ... many RMA operations to different targets ... */

MPI_Win_unlock_all(win);
```

More efficient than repeated `lock`/`unlock` when accessing many targets. Uses
shared locking — cannot coexist with any exclusive lock on the same window, but
`MPI_Put`, `MPI_Get`, and `MPI_Accumulate` are all permitted under a shared lock.

### MPI_Win_flush — Partial Progress

Without unlocking, you can force completion of outstanding operations:

```c
MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, win);

MPI_Put(data1, N, MPI_DOUBLE, target, 0, N, MPI_DOUBLE, win);
MPI_Win_flush(target, win);  /* Put is now complete at target */
/* Can now modify data1 safely */

MPI_Put(data2, N, MPI_DOUBLE, target, N, N, MPI_DOUBLE, win);

MPI_Win_unlock(target, win);
```

`MPI_Win_flush` completes all outstanding operations to the given target without
releasing the lock. `MPI_Win_flush_all` flushes to all targets.

`MPI_Win_flush_local` completes the operation locally (buffer safe to reuse) but
does not guarantee remote completion — useful when you need the buffer back but can
tolerate delayed delivery.

---

## 18.5 Synchronization Summary

| Method | Who calls | Scope | Best for |
|---|---|---|---|
| `MPI_Win_fence` | All in comm (collective) | All processes | Simple structured epochs |
| `MPI_Win_post/start/complete/wait` | Origin + target separately | Subset of processes | Pipelined, overlapping epochs |
| `MPI_Win_lock/unlock` | Origin only | One target at a time | Passive-target; lock-free algorithms |
| `MPI_Win_lock_all/unlock_all` | Origin only | All targets | Many-target passive access |
| `MPI_Win_flush` | Origin only | One target | Partial progress during a lock epoch |

---

## 18.6 Common Mistakes

**Mistake 1**: Reading data from `MPI_Get` before the epoch ends.
```c
MPI_Win_fence(0, win);
MPI_Get(buf, N, MPI_DOUBLE, src, 0, N, MPI_DOUBLE, win);
use_data(buf);         /* BUG: buf not valid until next fence */
MPI_Win_fence(0, win);
use_data(buf);         /* Correct */
```

**Mistake 2**: Assuming `MPI_Put` is immediately visible without synchronization.
```c
MPI_Put(data, N, MPI_DOUBLE, target, 0, N, MPI_DOUBLE, win);
/* target reads its window here — data may not be there yet */
```

**Mistake 3**: Overlapping epochs — having a process in two different lock epochs
to the same window simultaneously.

**Mistake 4**: Mixing passive-target locks with fence on the same window epoch
(undefined behavior).

---

## Summary

| Function | Model | Notes |
|---|---|---|
| `MPI_Win_fence` | Active (all) | Simple; global barrier |
| `MPI_Win_post` | Active (target) | Exposes window to group |
| `MPI_Win_start` | Active (origin) | Begins access to group |
| `MPI_Win_complete` | Active (origin) | Ends access epoch |
| `MPI_Win_wait` | Active (target) | Waits for all origins done |
| `MPI_Win_lock` | Passive | Lock one target |
| `MPI_Win_unlock` | Passive | Unlock; completes all ops |
| `MPI_Win_lock_all` | Passive | Lock all targets (shared) |
| `MPI_Win_unlock_all` | Passive | Unlock all targets |
| `MPI_Win_flush` | Passive | Force completion (no unlock) |
| `MPI_Win_flush_local` | Passive | Buffer safety only (no remote complete) |
