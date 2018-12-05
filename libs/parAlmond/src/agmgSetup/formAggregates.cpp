/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus, Rajesh Gandham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "parAlmond.hpp"

namespace parAlmond {


typedef struct{
	dlong  index;
	dlong  Nnbs;
	dlong LNN;
} nbs_t;

  
int compareNBSmax(const void *a, const void *b){
	nbs_t *pa = (nbs_t *)a;	
	nbs_t *pb = (nbs_t *)b;
	
	if (pa->Nnbs + pa->LNN < pb->Nnbs + pb->LNN)	return +1;
	if (pa->Nnbs + pa->LNN > pb->Nnbs + pb->LNN)	return -1;
	
	if (pa->index < pa->index )	return +1;
	if (pa->index > pa->index )	return -1;
	
	return 0;
}

 
int compareNBSmin(const void *a, const void *b){
	nbs_t *pa = (nbs_t *)a;	
	nbs_t *pb = (nbs_t *)b;
	
	if (pa->Nnbs + pa->LNN < pb->Nnbs + pb->LNN)	return -1;
	if (pa->Nnbs + pa->LNN > pb->Nnbs + pb->LNN)	return +1;
	
	if (pa->index < pa->index )	return +1;
	if (pa->index > pa->index )	return -1;
	
	return 0;
}
  
  ///////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////// 
  ///////////////////////////////////////////////////////////////////////////////////
  

void formAggregates(parCSR *A, parCSR *C,
                     hlong* FineToCoarse,
                     hlong* globalAggStarts){

  int rank, size;
  MPI_Comm_rank(A->comm, &rank);
  MPI_Comm_size(A->comm, &size);

  const dlong N   = C->Nrows;
  const dlong M   = C->Ncols;
  const dlong diagNNZ = C->diag->nnz;
  
  //  printf("\n-----------------------------------------\n");
  //printf("rank = %d (LQHSSN) =>  N = %d \t M = %d    ",rank,N,M);
  //printf("\n-----------------------------------------\n");

  dlong   *states = (dlong *)    calloc(M, sizeof(dlong));    //  M > N

  for(dlong i=0; i<N; i++)   // initialize states to -1
    states[i] = -1;
  
  hlong *globalRowStarts = A->globalRowStarts;  

  // construct the local neigbors
  nbs_t *V = (nbs_t *) calloc(N,sizeof(nbs_t));
  
  for(dlong i=0; i<N; i++){  
	V[i].index    = i;
	V[i].Nnbs     = C->diag->rowStarts[i+1] - C->diag->rowStarts[i];
	dlong dist     = C->diag->cols[C->diag->rowStarts[i+1]-1] - C->diag->cols[C->diag->rowStarts[i]];
	if (dist  > 0 )
	  V[i].LNN   =  V[i].Nnbs/dist;
	else
	  V[i].LNN = 0;	  
  }
  
    
  int MySort = 1;
  //options.getArgs("SORT",MySort);
  
  // sort the fake nbs
  if (MySort>0)		qsort(V,N,sizeof(nbs_t),compareNBSmax); 
  
  dlong R_num = 0; 	// number of non-isolated nodes to be used
     
  for (dlong i=0;i<N;i++)
	if(V[i].Nnbs>1)		R_num++;


  dlong R_nodes[N];   // N is number of local elements so at most N aggregates
  dlong R_pos[N];     // 
  
  dlong k = 0;


  // get the sorted indices
  for (dlong i=0;i<N;i++){    
	if(V[i].Nnbs>1){
		R_nodes[k] = V[i].index;	// R_nodes list of nodes to be used
		k++;
	}
  }

  dlong Agg_num = 0;  
  
  // First aggregates
  //#pragma omp parallel for
  for(dlong i=0; i<R_num; i++){
	if (states[R_nodes[i]] == -1){
		int ok = 0;
		// verify that all NBS are free
		for(dlong j=C->diag->rowStarts[R_nodes[i]]; j<C->diag->rowStarts[R_nodes[i]+1];j++){ 
			if (states[C->diag->cols[j]]>-1){
				ok=1;
				break;
			}
		}
		// construct the aggregate
		if (ok == 0){
		        for(dlong j=C->diag->rowStarts[R_nodes[i]]; j<C->diag->rowStarts[R_nodes[i]+1];j++)
		                states[C->diag->cols[j]] = Agg_num;				
							
			Agg_num++;
		}
	 }	 
  }
  
  R_num=0;   // reset the number of nodes to be used

  
  for (dlong i=0;i<N;i++){  // update list of  non-agreggate nodes
	if (states[V[i].index]==-1){      // cambie k por R_num
                R_nodes[R_num] = V[i].index;  // update the list of nodes
		R_num++;					  
	}
  } 
      
  dlong *psudoAgg = (dlong *) calloc(M,sizeof(dlong));   // what is different bwtween dlong and hlong? 

  // count the number of nodes at each aggregate
  for (dlong i=0;i<N;i++)
	if (states[V[i].index]>-1)		psudoAgg[states[V[i].index]]++;

  // #pragma omp parallel for
  for(dlong i=0; i<R_num; i++){
	if (states[R_nodes[i]] == -1){ 	 // sanity check			
		if (C->diag->rowStarts[R_nodes[i]+1] - C->diag->rowStarts[R_nodes[i]] > 1){  	 // at most one neigbor
			dlong Agg_max;
			dlong posIndx = 0;         
			dlong MoreAgg[Agg_num];
			for (dlong j=0;j<Agg_num;j++)
				MoreAgg[j]=0;		
			// count the number of strong conections with each aggregate		
			for(dlong j=C->diag->rowStarts[R_nodes[i]] ; j<C->diag->rowStarts[R_nodes[i]+1] ; j++){  // index 0 is itself
				if (states[C->diag->cols[j]] > -1){
					MoreAgg[states[ C->diag->cols[j]  ]]++;
				}
			}
			// look for the agregates with more strong connections & less nodes
			Agg_max = -1;
			for (dlong j=0;j<Agg_num;j++){
				if (Agg_max <= MoreAgg[j]){
					if (j == 0){
						Agg_max = MoreAgg[j];
						posIndx = j;
					}
					else if (Agg_max < MoreAgg[j]){							
						Agg_max = MoreAgg[j];
						posIndx = j;
					}
					else if (psudoAgg[posIndx] > psudoAgg[j]){
						Agg_max = MoreAgg[j];
						posIndx = j;
					}
				}
			}
			states[R_nodes[i]] = posIndx;				
			psudoAgg[posIndx]++;
		}
		else{  // no neighbors (isolated node)

		  // if (V[R_nodes[i]].offNnbs==0){
		    states[R_nodes[i]] = Agg_num;  // becomes a new aggregate
		    psudoAgg[Agg_num]++;
		    Agg_num++;   		  
		    //
		    //}  
		   // MULTIGRID becomes more slow if its added to an aggregate
		   /* dlong Min = psudoAgg[0];
		    for (dlong k = 1 ; k < Agg_num ; k++){
		      if (psudoAgg[k] < Min )     Min = psudoAgg[k];
		    }

		    states[R_nodes[i]] = Min;
		   */
		    
		}
	}	 
  }
  
  dlong *gNumAggs = (dlong *) calloc(size,sizeof(dlong));
  //globalAggStarts = (hlong *) calloc(size+1,sizeof(hlong));
  
  // count the coarse nodes/aggregates in each rannk
  MPI_Allgather(&Agg_num,1,MPI_DLONG,gNumAggs,1,MPI_DLONG,A->comm);

  globalAggStarts[0] = 0;
  for (int r=0;r<size;r++)
    globalAggStarts[r+1] = globalAggStarts[r] + gNumAggs[r];

  // enumerate the coarse nodes/aggregates
  for(dlong i=0; i<N; i++){
    //    if (states[i] >= 0) 
      FineToCoarse[i] = globalAggStarts[rank] + states[i];
      //else
      // FineToCoarse[i] = -1;
  }
    
 
  // share results
  for (dlong n=N;n<M;n++) FineToCoarse[n] = 0;
	ogsGatherScatter(FineToCoarse, ogsHlong,  ogsAdd, A->ogs);
  

/*

// print info of aggregates  
  
  dlong *AggNum = (dlong *) calloc(Agg_num,sizeof(dlong));
  
  for (dlong i = 0; i<M;i++)
	AggNum[FineToCoarse[i]]++;
    
  printf("\n rank %d elementos por Agregates ==========\n",rank);
  for (dlong i=0;i<Agg_num;i++)
  	printf("Agg %d  total = %d\n",i,AggNum[i]);
  printf("\n=======================================\n");
	*/

  free(states);
  free(V);
  free(psudoAgg);
  
  delete C;
}






} //namespace parAlmond
