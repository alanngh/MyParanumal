#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include <mpi.h>

#include "mesh3D.h"

extern "C"
{
  void *gsParallelGatherScatterSetup(int Ngather, int *gatherIds);
}

// assume nodes locally sorted by rank then global index
// assume gather and scatter are the same sets
ogs_t *meshParallelGatherScatterSetup3D(mesh3D *mesh,    // provides DEVICE
					iint Nlocal,     // number of local nodes
					iint Nbytes,     // number of bytes per node
					iint *gatherLocalIds,  // local index of nodes
					iint *gatherBaseIds,   // global index of their base nodes
					iint *gatherBaseRanks, // rank of their base nodes
					iint *gatherMaxRanks){  // max rank connected to base node

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  ogs_t *ogs = (ogs_t*) calloc(1, sizeof(ogs_t));
  
  // 1. count number of unique base nodes on this process
  ogs->Ngather = 0; // assumes at least one base node
  for(iint n=0;n<Nlocal;++n){
    iint test = (n==0) ? 1: (gatherBaseIds[n] != gatherBaseIds[n-1]);
    ogs->Ngather += test;
  }
  
  iint *gatherOffsets = (iint*) calloc(ogs->Ngather+1, sizeof(iint)); // offset into sorted list of nodes

  // only finds bases
  ogs->Ngather = 0; // reset counter
  for(iint n=0;n<Nlocal;++n){
    iint test = (n==0) ? 1: (gatherBaseIds[n] != gatherBaseIds[n-1]);
    if(test){
      gatherOffsets[ogs->Ngather++] = n;  // increment unique base counter and record index into shuffled list of ndoes
    }
  }
  gatherOffsets[ogs->Ngather] = Nlocal;

  // allocate buffers on DEVICE
  ogs->o_gatherTmp      = mesh->device.malloc(ogs->Ngather*Nbytes);
  ogs->o_gatherOffsets  = mesh->device.malloc((ogs->Ngather+1)*sizeof(iint), gatherOffsets);
  ogs->o_gatherLocalIds = mesh->device.malloc(Nlocal*sizeof(iint), gatherLocalIds);
  
  ogs->gatherLocalIds  = gatherLocalIds;
  ogs->gatherBaseIds   = gatherBaseIds;
  ogs->gatherBaseRanks = gatherBaseRanks;
  ogs->gatherMaxRanks  = gatherMaxRanks;

  // list of nodes to extract from DEVICE gathered array
  ogs->Nhalo = 0;
  for(iint n=0;n<ogs->Ngather;++n){
    int id = gatherOffsets[n];
    if(gatherBaseRanks[id]!=gatherMaxRanks[id] || gatherBaseRanks[id]!=rank){ // is this a shared node ?
      ++ogs->Nhalo;
    }
  }
  
  printf("ogs->Nhalo = %d, ogs->Ngather = %d\n", ogs->Nhalo, ogs->Ngather);

  // set up gather-scatter of halo nodes
  ogs->gatherGsh = NULL;
  if(ogs->Nhalo){
    
    iint *haloLocalIds = (iint*) calloc(ogs->Nhalo, sizeof(iint));
    iint *haloGlobalIds = (iint*) calloc(ogs->Nhalo, sizeof(iint));
    ogs->Nhalo = 0;
    for(iint n=0;n<ogs->Ngather;++n){
      int id = gatherOffsets[n];
      if(gatherBaseRanks[id]!=gatherMaxRanks[id] || gatherBaseRanks[id]!=rank){
	haloLocalIds[ogs->Nhalo] = n;
	haloGlobalIds[ogs->Nhalo] = gatherBaseIds[id];
	++ogs->Nhalo;
      }
    }     
    
    // list of halo node indices
    ogs->o_haloLocalIds = mesh->device.malloc(ogs->Nhalo*sizeof(iint), haloLocalIds);
    
    // allocate buffer for gathering on halo nodes (danger on size of buffer)
    ogs->haloTmp = (dfloat*) malloc(ogs->Nhalo*Nbytes);
    ogs->o_haloTmp  = mesh->device.malloc(ogs->Nhalo*Nbytes, ogs->haloTmp);
    
    // initiate gslib gather-scatter comm pattern
    ogs->gatherGsh = gsParallelGatherScatterSetup(ogs->Nhalo, haloGlobalIds);
    
    free(haloGlobalIds);
    free(haloLocalIds);
  }

  
  ogs->Nscatter = ogs->Ngather;
  ogs->o_scatterOffsets = ogs->o_gatherOffsets;
  ogs->o_scatterLocalIds = ogs->o_gatherLocalIds;
  
  free(gatherOffsets);

  return ogs;
}