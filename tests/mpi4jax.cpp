#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(void)
{
  int i, flag;

  int nodesize, noderank;
  int size, rank, irank;
  int tablesize, localtablesize;
  int *table, *localtable;
  int *model;

  MPI_Comm allcomm, nodecomm;

  char verstring[MPI_MAX_LIBRARY_VERSION_STRING];
  char nodename[MPI_MAX_PROCESSOR_NAME];

  MPI_Aint winsize;
  int windisp;
  int *winptr;

  int version, subversion, verstringlen, nodestringlen;

  allcomm = MPI_COMM_WORLD;

  MPI_Win wintable;

  tablesize = 5;

  MPI_Init(NULL, NULL);

  MPI_Comm_size(allcomm, &size);
  MPI_Comm_rank(allcomm, &rank);

  MPI_Get_processor_name(nodename, &nodestringlen);

  MPI_Get_version(&version, &subversion);
  MPI_Get_library_version(verstring, &verstringlen);

  if (rank == 0)
    {
      printf("Version %d, subversion %d\n", version, subversion);
      printf("Library <%s>\n", verstring);
    }

  // Create node-local communicator

  MPI_Comm_split_type(allcomm, MPI_COMM_TYPE_SHARED, rank,
              MPI_INFO_NULL, &nodecomm);

  MPI_Comm_size(nodecomm, &nodesize);
  MPI_Comm_rank(nodecomm, &noderank);

  // Only rank 0 on a node actually allocates memory

  localtablesize = 0;

  if (noderank == 0) localtablesize = tablesize;

  // debug info

  printf("Rank %d of %d, rank %d of %d in node <%s>, localtablesize %d\n",
     rank, size, noderank, nodesize, nodename, localtablesize);


  MPI_Win_allocate_shared(localtablesize*sizeof(int), sizeof(int),
              MPI_INFO_NULL, nodecomm, &localtable, &wintable);

  MPI_Win_get_attr(wintable, MPI_WIN_MODEL, &model, &flag);

  if (1 != flag)
    {
      printf("Attribute MPI_WIN_MODEL not defined\n");
    }
  else
    {
      if (MPI_WIN_UNIFIED == *model)
    {
      if (rank == 0) printf("Memory model is MPI_WIN_UNIFIED\n");
    }
      else
    {
      if (rank == 0) printf("Memory model is *not* MPI_WIN_UNIFIED\n");

      MPI_Finalize();
      return 1;
    }
    }

  // need to get local pointer valid for table on rank 0

  table = localtable;

  if (noderank != 0)
    {
      MPI_Win_shared_query(wintable, 0, &winsize, &windisp, &table);
    }

  // All table pointers should now point to copy on noderank 0

  // Initialise table on rank 0 with appropriate synchronisation

  MPI_Win_fence(0, wintable);

  if (noderank == 0)
    {
      for (i=0; i < tablesize; i++)
    {
      table[i] = rank*tablesize + i;
    }
    }

  MPI_Win_fence(0, wintable);

  // Check we did it right

  for (i=0; i < tablesize; i++)
    {
      printf("rank %d, noderank %d, table[%d] = %d\n",
         rank, noderank, i, table[i]);
    }

  MPI_Finalize();
}