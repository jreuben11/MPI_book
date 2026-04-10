# Appendix A: MPI Function Quick Reference

Functions grouped by category. Signatures use C types. All return `int` (error code).
Omit `_c` large-count variants unless noted.

---

## A.1 Initialization & Environment

```c
MPI_Init(int *argc, char ***argv)
MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
MPI_Finalize(void)
MPI_Initialized(int *flag)
MPI_Finalized(int *flag)
MPI_Abort(MPI_Comm comm, int errorcode)

MPI_Comm_rank(MPI_Comm comm, int *rank)
MPI_Comm_size(MPI_Comm comm, int *size)
MPI_Get_processor_name(char *name, int *resultlen)
MPI_Get_version(int *version, int *subversion)
MPI_Get_library_version(char *version, int *resultlen)
MPI_Wtime(void)          → double (wall-clock seconds)
MPI_Wtick(void)          → double (timer resolution)
MPI_Query_thread(int *provided)
MPI_Is_thread_main(int *flag)
```

---

## A.2 Point-to-Point Communication

### Blocking

```c
MPI_Send(const void *buf, int count, MPI_Datatype dtype, int dest, int tag, MPI_Comm comm)
MPI_Recv(void *buf, int count, MPI_Datatype dtype, int source, int tag,
         MPI_Comm comm, MPI_Status *status)
MPI_Bsend(const void *buf, int count, MPI_Datatype dtype, int dest, int tag, MPI_Comm comm)
MPI_Ssend(const void *buf, int count, MPI_Datatype dtype, int dest, int tag, MPI_Comm comm)
MPI_Rsend(const void *buf, int count, MPI_Datatype dtype, int dest, int tag, MPI_Comm comm)
MPI_Sendrecv(sbuf, scount, stype, dest, stag, rbuf, rcount, rtype, src, rtag, comm, *status)
MPI_Sendrecv_replace(buf, count, dtype, dest, stag, src, rtag, comm, *status)
```

### Non-Blocking

```c
MPI_Isend(const void *buf, int count, MPI_Datatype dtype, int dest, int tag,
          MPI_Comm comm, MPI_Request *req)
MPI_Irecv(void *buf, int count, MPI_Datatype dtype, int source, int tag,
          MPI_Comm comm, MPI_Request *req)
MPI_Ibsend(...)    MPI_Issend(...)    MPI_Irsend(...)
```

### Persistent

```c
MPI_Send_init(const void *buf, int count, MPI_Datatype dtype, int dest, int tag,
              MPI_Comm comm, MPI_Request *req)
MPI_Recv_init(void *buf, int count, MPI_Datatype dtype, int source, int tag,
              MPI_Comm comm, MPI_Request *req)
MPI_Bsend_init(...)  MPI_Ssend_init(...)  MPI_Rsend_init(...)
MPI_Start(MPI_Request *req)
MPI_Startall(int count, MPI_Request reqs[])
```

### Probe

```c
MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status)
MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status)
MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *msg, MPI_Status *status)
MPI_Mrecv(void *buf, int count, MPI_Datatype dtype, MPI_Message *msg, MPI_Status *status)
```

### Buffered

```c
MPI_Buffer_attach(void *buffer, int size)
MPI_Buffer_detach(void *buffer_addr, int *size)
```

### Completion

```c
MPI_Wait(MPI_Request *req, MPI_Status *status)
MPI_Test(MPI_Request *req, int *flag, MPI_Status *status)
MPI_Waitall(int count, MPI_Request reqs[], MPI_Status statuses[])
MPI_Waitany(int count, MPI_Request reqs[], int *index, MPI_Status *status)
MPI_Waitsome(int incount, MPI_Request reqs[], int *outcount,
             int indices[], MPI_Status statuses[])
MPI_Testall(...)  MPI_Testany(...)  MPI_Testsome(...)
MPI_Request_free(MPI_Request *req)
MPI_Cancel(MPI_Request *req)
MPI_Test_cancelled(MPI_Status *status, int *flag)
MPI_Get_count(MPI_Status *status, MPI_Datatype dtype, int *count)
MPI_Get_count_c(MPI_Status *status, MPI_Datatype dtype, MPI_Count *count)
```

---

## A.3 Collective Communication

### Blocking Collectives

```c
MPI_Barrier(MPI_Comm comm)
MPI_Bcast(void *buf, int count, MPI_Datatype dtype, int root, MPI_Comm comm)
MPI_Scatter(sbuf, scount, stype, rbuf, rcount, rtype, root, comm)
MPI_Gather(sbuf, scount, stype, rbuf, rcount, rtype, root, comm)
MPI_Allgather(sbuf, scount, stype, rbuf, rcount, rtype, comm)
MPI_Alltoall(sbuf, scount, stype, rbuf, rcount, rtype, comm)
MPI_Reduce(sbuf, rbuf, count, dtype, op, root, comm)
MPI_Allreduce(sbuf, rbuf, count, dtype, op, comm)
MPI_Scan(sbuf, rbuf, count, dtype, op, comm)
MPI_Exscan(sbuf, rbuf, count, dtype, op, comm)

MPI_Scatterv(sbuf, scounts[], displs[], stype, rbuf, rcount, rtype, root, comm)
MPI_Gatherv(sbuf, scount, stype, rbuf, rcounts[], displs[], rtype, root, comm)
MPI_Allgatherv(sbuf, scount, stype, rbuf, rcounts[], displs[], rtype, comm)
MPI_Alltoallv(sbuf, scounts[], sdispls[], stype, rbuf, rcounts[], rdispls[], rtype, comm)
MPI_Alltoallw(sbuf, scounts[], sdispls[], stypes[], rbuf, rcounts[], rdispls[], rtypes[], comm)

MPI_Reduce_scatter(sbuf, rbuf, rcounts[], dtype, op, comm)
MPI_Reduce_scatter_block(sbuf, rbuf, recvcount, dtype, op, comm)
```

### Non-Blocking Collectives (MPI_I* variants of all above)

```c
MPI_Ibcast(...)   MPI_Iscatter(...)   MPI_Igather(...)  MPI_Iallgather(...)
MPI_Ialltoall(...)  MPI_Ireduce(...)  MPI_Iallreduce(...)
MPI_Iscan(...)    MPI_Iexscan(...)    MPI_Ibarrier(...)
```

### Persistent Collectives (MPI 4.0 — MPI_*_init variants)

```c
MPI_Bcast_init(buf, count, dtype, root, comm, info, *req)
MPI_Allreduce_init(sbuf, rbuf, count, dtype, op, comm, info, *req)
MPI_Scatter_init(...)   MPI_Gather_init(...)   MPI_Allgather_init(...)
MPI_Reduce_init(...)    MPI_Alltoall_init(...)  MPI_Barrier_init(...)
```

### Reduction Operations

```c
MPI_Op_create(MPI_User_function *fn, int commute, MPI_Op *op)
MPI_Op_free(MPI_Op *op)
```

---

## A.4 Datatypes

```c
MPI_Type_contiguous(int count, MPI_Datatype old, MPI_Datatype *new)
MPI_Type_vector(int count, int block, int stride, MPI_Datatype old, MPI_Datatype *new)
MPI_Type_hvector(int count, int block, MPI_Aint stride, MPI_Datatype old, MPI_Datatype *new)
MPI_Type_indexed(int count, int blocks[], int displs[], MPI_Datatype old, MPI_Datatype *new)
MPI_Type_hindexed(int count, int blocks[], MPI_Aint displs[], MPI_Datatype old, MPI_Datatype *new)
MPI_Type_create_struct(int count, int blocks[], MPI_Aint displs[],
                        MPI_Datatype types[], MPI_Datatype *new)
MPI_Type_create_subarray(int ndims, int sizes[], int subsizes[], int starts[],
                          int order, MPI_Datatype old, MPI_Datatype *new)
MPI_Type_create_darray(int size, int rank, int ndims, int gsizes[], int distribs[],
                        int dargs[], int psizes[], int order, MPI_Datatype old, MPI_Datatype *new)
MPI_Type_create_resized(MPI_Datatype old, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *new)

MPI_Type_commit(MPI_Datatype *dtype)
MPI_Type_free(MPI_Datatype *dtype)
MPI_Type_size(MPI_Datatype dtype, int *size)
MPI_Type_size_x(MPI_Datatype dtype, MPI_Count *size)
MPI_Type_get_extent(MPI_Datatype dtype, MPI_Aint *lb, MPI_Aint *extent)
MPI_Type_get_extent_x(MPI_Datatype dtype, MPI_Aint *lb, MPI_Count *extent)
MPI_Type_get_true_extent(MPI_Datatype dtype, MPI_Aint *lb, MPI_Aint *extent)
MPI_Get_address(const void *location, MPI_Aint *address)
MPI_Type_set_name(MPI_Datatype dtype, const char *name)
MPI_Type_get_name(MPI_Datatype dtype, char *name, int *resultlen)
```

---

## A.5 Communicators & Groups

```c
MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)
MPI_Comm_idup(MPI_Comm comm, MPI_Comm *newcomm, MPI_Request *req)
MPI_Comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm)
MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm)
MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm)
MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm)
MPI_Comm_create_from_group(MPI_Group group, const char *tag, MPI_Info info,
                            MPI_Errhandler errhndlr, MPI_Comm *newcomm)
MPI_Comm_free(MPI_Comm *comm)
MPI_Comm_compare(MPI_Comm c1, MPI_Comm c2, int *result)
MPI_Comm_remote_size(MPI_Comm comm, int *size)
MPI_Intercomm_create(MPI_Comm lcomm, int lleader, MPI_Comm pcomm, int rleader,
                      int tag, MPI_Comm *newcomm)
MPI_Intercomm_merge(MPI_Comm comm, int high, MPI_Comm *newcomm)

MPI_Comm_group(MPI_Comm comm, MPI_Group *group)
MPI_Group_incl(MPI_Group group, int n, int ranks[], MPI_Group *newgroup)
MPI_Group_excl(MPI_Group group, int n, int ranks[], MPI_Group *newgroup)
MPI_Group_union(MPI_Group g1, MPI_Group g2, MPI_Group *newgroup)
MPI_Group_intersection(MPI_Group g1, MPI_Group g2, MPI_Group *newgroup)
MPI_Group_difference(MPI_Group g1, MPI_Group g2, MPI_Group *newgroup)
MPI_Group_rank(MPI_Group group, int *rank)
MPI_Group_size(MPI_Group group, int *size)
MPI_Group_free(MPI_Group *group)
MPI_Group_from_session_pset(MPI_Session session, const char *pset_name, MPI_Group *newgroup)
```

---

## A.6 Topologies

```c
MPI_Cart_create(MPI_Comm comm, int ndims, int dims[], int periods[], int reorder,
                MPI_Comm *newcomm)
MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[])
MPI_Cart_rank(MPI_Comm comm, int coords[], int *rank)
MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_src, int *rank_dst)
MPI_Cart_sub(MPI_Comm comm, int remain_dims[], MPI_Comm *newcomm)
MPI_Dims_create(int nnodes, int ndims, int dims[])
MPI_Dist_graph_create_adjacent(comm, indegree, sources[], sweights[],
                                outdegree, dests[], dweights[],
                                info, reorder, *newcomm)
MPI_Dist_graph_neighbors_count(comm, *indegree, *outdegree, *weighted)
MPI_Dist_graph_neighbors(comm, maxindegree, sources[], sweights[],
                          maxoutdegree, dests[], dweights[])
MPI_Neighbor_alltoall(sbuf, scount, stype, rbuf, rcount, rtype, comm)
MPI_Neighbor_alltoallv(sbuf, scounts[], sdispls[], stype,
                        rbuf, rcounts[], rdispls[], rtype, comm)
```

---

## A.7 One-Sided Communication (RMA)

```c
MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info,
               MPI_Comm comm, MPI_Win *win)
MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info,
                 MPI_Comm comm, void *baseptr, MPI_Win *win)
MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win)
MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info,
                         MPI_Comm comm, void *baseptr, MPI_Win *win)
MPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr)
MPI_Win_attach(MPI_Win win, void *base, MPI_Aint size)
MPI_Win_detach(MPI_Win win, const void *base)
MPI_Win_free(MPI_Win *win)

MPI_Put(origin_addr, origin_count, origin_dtype, target_rank, target_disp,
        target_count, target_dtype, win)
MPI_Get(origin_addr, origin_count, origin_dtype, target_rank, target_disp,
        target_count, target_dtype, win)
MPI_Accumulate(origin_addr, origin_count, origin_dtype, target_rank, target_disp,
               target_count, target_dtype, op, win)
MPI_Get_accumulate(origin_addr, origin_count, origin_dtype,
                   result_addr, result_count, result_dtype,
                   target_rank, target_disp, target_count, target_dtype, op, win)
MPI_Fetch_and_op(origin_addr, result_addr, dtype, target_rank, target_disp, op, win)
MPI_Compare_and_swap(origin_addr, compare_addr, result_addr, dtype,
                     target_rank, target_disp, win)
MPI_Rput(...)  MPI_Rget(...)  MPI_Raccumulate(...)  MPI_Rget_accumulate(...)

MPI_Win_fence(int assert, MPI_Win win)
MPI_Win_post(MPI_Group group, int assert, MPI_Win win)
MPI_Win_start(MPI_Group group, int assert, MPI_Win win)
MPI_Win_complete(MPI_Win win)
MPI_Win_wait(MPI_Win win)
MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win)
MPI_Win_unlock(int rank, MPI_Win win)
MPI_Win_lock_all(int assert, MPI_Win win)
MPI_Win_unlock_all(MPI_Win win)
MPI_Win_flush(int rank, MPI_Win win)
MPI_Win_flush_all(MPI_Win win)
MPI_Win_flush_local(int rank, MPI_Win win)
MPI_Win_flush_local_all(MPI_Win win)
MPI_Win_sync(MPI_Win win)
```

---

## A.8 MPI-IO

```c
MPI_File_open(MPI_Comm comm, const char *filename, int amode,
              MPI_Info info, MPI_File *fh)
MPI_File_close(MPI_File *fh)
MPI_File_delete(const char *filename, MPI_Info info)
MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype,
                  MPI_Datatype filetype, const char *datarep, MPI_Info info)
MPI_File_get_size(MPI_File fh, MPI_Offset *size)
MPI_File_set_size(MPI_File fh, MPI_Offset size)
MPI_File_preallocate(MPI_File fh, MPI_Offset size)
MPI_File_sync(MPI_File fh)

MPI_File_read_at(fh, offset, buf, count, dtype, *status)
MPI_File_write_at(fh, offset, buf, count, dtype, *status)
MPI_File_read_at_all(fh, offset, buf, count, dtype, *status)
MPI_File_write_at_all(fh, offset, buf, count, dtype, *status)
MPI_File_iread_at(fh, offset, buf, count, dtype, *req)
MPI_File_iwrite_at(fh, offset, buf, count, dtype, *req)
MPI_File_iread_at_all(fh, offset, buf, count, dtype, *req)
MPI_File_iwrite_at_all(fh, offset, buf, count, dtype, *req)

MPI_File_read(fh, buf, count, dtype, *status)        /* individual pointer */
MPI_File_write(fh, buf, count, dtype, *status)
MPI_File_read_all(fh, buf, count, dtype, *status)    /* collective */
MPI_File_write_all(fh, buf, count, dtype, *status)

MPI_File_read_shared(fh, buf, count, dtype, *status) /* shared pointer */
MPI_File_write_shared(fh, buf, count, dtype, *status)
MPI_File_read_ordered(fh, buf, count, dtype, *status)
MPI_File_write_ordered(fh, buf, count, dtype, *status)

MPI_File_seek(fh, offset, whence)
MPI_File_seek_shared(fh, offset, whence)
MPI_File_get_position(fh, *offset)
MPI_File_get_position_shared(fh, *offset)
MPI_File_set_errhandler(fh, errhandler)
MPI_File_get_info(fh, *info)
```

---

## A.9 Error Handling

```c
MPI_Error_string(int errorcode, char *string, int *resultlen)
MPI_Error_class(int errorcode, int *errorclass)
MPI_Comm_create_errhandler(MPI_Comm_errhandler_function *fn, MPI_Errhandler *errhandler)
MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler)
MPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *errhandler)
MPI_Errhandler_free(MPI_Errhandler *errhandler)
MPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler)
MPI_File_set_errhandler(MPI_File fh, MPI_Errhandler errhandler)
/* ULFM (MPI 5.0) */
MPI_Comm_revoke(MPI_Comm comm)
MPI_Comm_shrink(MPI_Comm comm, MPI_Comm *newcomm)
MPI_Comm_agree(MPI_Comm comm, int *flag)
```

---

## A.10 Sessions (MPI 4.0)

```c
MPI_Session_init(MPI_Info info, MPI_Errhandler errhandler, MPI_Session *session)
MPI_Session_finalize(MPI_Session *session)
MPI_Session_get_num_psets(MPI_Session session, MPI_Info info, int *npsets)
MPI_Session_get_nth_pset(MPI_Session session, MPI_Info info, int n, int *len, char *name)
MPI_Session_get_pset_info(MPI_Session session, const char *pset_name,
                           MPI_Info *info)
MPI_Group_from_session_pset(MPI_Session session, const char *pset_name, MPI_Group *newgroup)
MPI_Comm_create_from_group(MPI_Group group, const char *tag, MPI_Info info,
                            MPI_Errhandler errhandler, MPI_Comm *newcomm)
```

---

## A.11 Partitioned Communication (MPI 4.0)

```c
MPI_Psend_init(const void *buf, int partitions, MPI_Count count, MPI_Datatype dtype,
               int dest, int tag, MPI_Comm comm, MPI_Info info, MPI_Request *req)
MPI_Precv_init(void *buf, int partitions, MPI_Count count, MPI_Datatype dtype,
               int source, int tag, MPI_Comm comm, MPI_Info info, MPI_Request *req)
MPI_Pready(int partition, MPI_Request *req)
MPI_Pready_range(int partition_low, int partition_high, MPI_Request *req)
MPI_Pready_list(int length, int partition_list[], MPI_Request *req)
MPI_Parrived(MPI_Request *req, int partition, int *flag)
```

---

## A.12 MPI_Info

```c
MPI_Info_create(MPI_Info *info)
MPI_Info_free(MPI_Info *info)
MPI_Info_set(MPI_Info info, const char *key, const char *value)
MPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag)
MPI_Info_delete(MPI_Info info, const char *key)
MPI_Info_get_nkeys(MPI_Info info, int *nkeys)
MPI_Info_get_nthkey(MPI_Info info, int n, char *key)
MPI_Info_dup(MPI_Info info, MPI_Info *newinfo)
```
