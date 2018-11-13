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



/*
void formAggregates(parCSR *A, parCSR *C,
                     hlong* FineToCoarse,
                     hlong* globalAggStarts){

  int rank, size;
  MPI_Comm_rank(A->comm, &rank);
  MPI_Comm_size(A->comm, &size);

  const dlong N   = C->Nrows;
  const dlong M   = C->Ncols;
  const dlong diagNNZ = C->diag->nnz;
  const dlong offdNNZ = C->offd->nnz;

	printf("----------------------\n");
	printf(" rank = %d   N =  %d M = %d \n",rank,N,M);
	printf("----------------------\n");


  dfloat *rands = (dfloat *) calloc(M, sizeof(dfloat));
  int   *states = (int *)    calloc(M, sizeof(int));

  dfloat *Tr = (dfloat *) calloc(M, sizeof(dfloat));
  int    *Ts = (int *)    calloc(M, sizeof(int));
  hlong  *Ti = (hlong *)  calloc(M, sizeof(hlong));
  hlong  *Tc = (hlong *)  calloc(M, sizeof(hlong));

  hlong *globalRowStarts = A->globalRowStarts;

  for(dlong i=0; i<N; i++)
    rands[i] = (dfloat) drand48();

  // add the number of non-zeros in each column
  int *colCnt = (int *) calloc(M,sizeof(int));
  for(dlong i=0; i<diagNNZ; i++)
    colCnt[C->diag->cols[i]]++;

  for(dlong i=0; i<offdNNZ; i++)
    colCnt[C->offd->cols[i]]++;

  //gs for total column counts
  ogsGatherScatter(colCnt, ogsInt, ogsAdd, A->ogs);

  //add random pertubation
  for(int i=0;i<N;++i)
    rands[i] += colCnt[i];

  //gs to fill halo region
  ogsGatherScatter(rands, ogsDfloat, ogsAdd, A->ogs);

  hlong done = 0;
  while(!done){
    // first neighbours
    // #pragma omp parallel for
    for(dlong i=0; i<N; i++){

      int smax = states[i];
      dfloat rmax = rands[i];
      hlong imax = i + globalRowStarts[rank];

      if(smax != 1){
        //local entries
        for(dlong jj=C->diag->rowStarts[i];jj<C->diag->rowStarts[i+1];jj++){
          const dlong col = C->diag->cols[jj];
          if (col==i) continue;
          if(customLess(smax, rmax, imax, states[col], rands[col], col + globalRowStarts[rank])){
            smax = states[col];
            rmax = rands[col];
            imax = col + globalRowStarts[rank];
          }
        }
        //nonlocal entries
        for(dlong jj=C->offd->rowStarts[i];jj<C->offd->rowStarts[i+1];jj++){
          const dlong col = C->offd->cols[jj];
          if(customLess(smax, rmax, imax, states[col], rands[col], A->colMap[col])) {
            smax = states[col];
            rmax = rands[col];
            imax = A->colMap[col];
          }
        }
      }
      Ts[i] = smax;
      Tr[i] = rmax;
      Ti[i] = imax;
    }

    //share results
    for (dlong n=N;n<M;n++) {
      Tr[n] = 0.;
      Ts[n] = 0;
      Ti[n] = 0;
    }
    ogsGatherScatter(Tr, ogsDfloat, ogsAdd, A->ogs);
    ogsGatherScatter(Ts, ogsInt,    ogsAdd, A->ogs);
    ogsGatherScatter(Ti, ogsHlong,  ogsAdd, A->ogs);

    // second neighbours
    // #pragma omp parallel for
    for(dlong i=0; i<N; i++){
      int    smax = Ts[i];
      dfloat rmax = Tr[i];
      hlong  imax = Ti[i];

      //local entries
      for(dlong jj=C->diag->rowStarts[i];jj<C->diag->rowStarts[i+1];jj++){
        const dlong col = C->diag->cols[jj];
        if (col==i) continue;
        if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
          smax = Ts[col];
          rmax = Tr[col];
          imax = Ti[col];
        }
      }
      //nonlocal entries
      for(dlong jj=C->offd->rowStarts[i];jj<C->offd->rowStarts[i+1];jj++){
        const dlong col = C->offd->cols[jj];
        if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
          smax = Ts[col];
          rmax = Tr[col];
          imax = Ti[col];
        }
      }

      // if I am the strongest among all the 1 and 2 ring neighbours
      // I am an MIS node
      if((states[i] == 0) && (imax == (i + globalRowStarts[rank])))
        states[i] = 1;

      // if there is an MIS node within distance 2, I am removed
      if((states[i] == 0) && (smax == 1))
        states[i] = -1;
    }

    //share results
    for (dlong n=N;n<M;n++) states[n] = 0;
    ogsGatherScatter(states, ogsInt, ogsAdd, A->ogs);

    // if number of undecided nodes = 0, algorithm terminates
    hlong cnt = std::count(states, states+N, 0);
    MPI_Allreduce(&cnt,&done,1,MPI_HLONG, MPI_SUM,A->comm);
    done = (done == 0) ? 1 : 0;
  }
  
  
	printf("\n ------ states (RANK  = %d ) antes ------\n",rank);
	for (dlong i=0;i<M;i++)
		printf("%d ",states[i]);
	printf("\n ----------------------\n");


  dlong numAggs = 0;
  dlong *gNumAggs = (dlong *) calloc(size,sizeof(dlong));

  // count the coarse nodes/aggregates
  for(dlong i=0; i<N; i++)
    if(states[i] == 1) numAggs++;

	printf("\n ---------------------\n");
	printf("rank = %d  size =%d  forming  munAggs = %d ",rank,size,numAggs);
	printf("\n ----------------------\n");


  MPI_Allgather(&numAggs,1,MPI_DLONG,gNumAggs,1,MPI_DLONG,A->comm);

	printf("\n ----------rank = %d  size =%d-----------\n",rank,size);
	 for (int r=0;r<size;r++)
    	printf(" %d ",gNumAggs[r]);
	printf("\n ----------------------\n");

  globalAggStarts[0] = 0;
  for (int r=0;r<size;r++)
    globalAggStarts[r+1] = globalAggStarts[r] + gNumAggs[r];

	printf("\n ------ globalAggStarts (RANK  = %d ) ------\n",rank);
	for (dlong i=0;i<size+1;i++)
		printf("%d ",globalAggStarts[i]);
	printf("\n ----------------------\n");

  numAggs = 0;
  // enumerate the coarse nodes/aggregates
  for(dlong i=0; i<N; i++) {
    if(states[i] == 1) {
      FineToCoarse[i] = globalAggStarts[rank] + numAggs++;
    } else {
      FineToCoarse[i] = -1;
    }
  }
  for(dlong i=N; i<M; i++) FineToCoarse[i] = 0;

	printf("\n ------ FineToCoarse (RANK  = %d ) antes ------\n",rank);
	for (dlong i=0;i<M;i++)
		printf("%d\t%d\t%d\t%d\t%d\n",i,FineToCoarse[i],Ts[i],Ti[i],Tc[i]);
	printf("\n ----------------------\n");




  //share the initial aggregate flags
  ogsGatherScatter(FineToCoarse, ogsHlong, ogsAdd, A->ogs);


	printf("\n ------ FineToCoarse (RANK  = %d ) despues ------\n",rank);
	for (dlong i=0;i<M;i++)
		printf("%d\t%d\t%d\t%d\t%d\n",i,FineToCoarse[i],Ts[i],Ti[i],Tc[i]);
	printf("\n ----------------------\n");

  // form the aggregates
  // #pragma omp parallel for
  for(dlong i=0; i<N; i++){
    int   smax = states[i];
    dfloat rmax = rands[i];
    hlong  imax = i + globalRowStarts[rank];
    hlong  cmax = FineToCoarse[i];

    if(smax != 1){
      //local entries
      for(dlong jj=C->diag->rowStarts[i];jj<C->diag->rowStarts[i+1];jj++){
        const dlong col = C->diag->cols[jj];
        if (col==i) continue;
        if(customLess(smax, rmax, imax, states[col], rands[col], col + globalRowStarts[rank])){
          smax = states[col];
          rmax = rands[col];
          imax = col + globalRowStarts[rank];
          cmax = FineToCoarse[col];
        }
      }
      //nonlocal entries
      for(dlong jj=C->offd->rowStarts[i];jj<C->offd->rowStarts[i+1];jj++){
        const dlong col = C->offd->cols[jj];
        if(customLess(smax, rmax, imax, states[col], rands[col], A->colMap[col])){
          smax = states[col];
          rmax = rands[col];
          imax = A->colMap[col];
          cmax = FineToCoarse[col];
        }
      }
    }
    Ts[i] = smax;
    Tr[i] = rmax;
    Ti[i] = imax;
    Tc[i] = cmax;

    if((states[i] == -1) && (smax == 1) && (cmax > -1))
      FineToCoarse[i] = cmax;
  }

  //share results
  for (dlong n=N;n<M;n++) {
    FineToCoarse[n] = 0;
    Tr[n] = 0.;
    Ts[n] = 0;
    Ti[n] = 0;
    Tc[n] = 0;
  }
  
  
  printf("\n ------ FineToCoarse (RANK  = %d ) 1er aggreates antes------\n",rank);
	for (dlong i=0;i<M;i++)
		printf("%d(%d) ",FineToCoarse[i],Ti[i]);
	printf("\n ----------------------\n");
  
  
  ogsGatherScatter(FineToCoarse, ogsHlong,  ogsAdd, A->ogs);
  ogsGatherScatter(Tr,     ogsDfloat, ogsAdd, A->ogs);
  ogsGatherScatter(Ts,     ogsInt,    ogsAdd, A->ogs);
  ogsGatherScatter(Ti,     ogsHlong,  ogsAdd, A->ogs);
  ogsGatherScatter(Tc,     ogsHlong,  ogsAdd, A->ogs);
  
  printf("\n ------ FineToCoarse (RANK  = %d ) 1er aggreates despues------\n",rank);
	for (dlong i=0;i<M;i++)
		printf("%d(%d) ",FineToCoarse[i],Ti[i]);
	printf("\n ----------------------\n");
  

  // second neighbours
  // #pragma omp parallel for
  for(dlong i=0; i<N; i++){
    int    smax = Ts[i];
    dfloat rmax = Tr[i];
    hlong  imax = Ti[i];
    hlong  cmax = Tc[i];

    //local entries
    for(dlong jj=C->diag->rowStarts[i];jj<C->diag->rowStarts[i+1];jj++){
      const dlong col = C->diag->cols[jj];
      if (col==i) continue;
      if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
        smax = Ts[col];
        rmax = Tr[col];
        imax = Ti[col];
        cmax = Tc[col];
      }
    }
    //nonlocal entries
    for(dlong jj=C->offd->rowStarts[i];jj<C->offd->rowStarts[i+1];jj++){
      const dlong col = C->offd->cols[jj];
      if(customLess(smax, rmax, imax, Ts[col], Tr[col], Ti[col])){
        smax = Ts[col];
        rmax = Tr[col];
        imax = Ti[col];
        cmax = Tc[col];
      }
    }

    if((states[i] == -1) && (smax == 1) && (cmax > -1))
      FineToCoarse[i] = cmax;
  }
  
 printf("\n ------ FineToCoarse (RANK  = %d ) 2do aggreates antes------\n",rank);
	for (dlong i=0;i<M;i++)
	if (i%15==0)  printf("\n");
		printf("%d(%d) ",FineToCoarse[i],Ti[i]);
	printf("\n ----------------------\n");




  //share results
  for (dlong n=N;n<M;n++) FineToCoarse[n] = 0;
  ogsGatherScatter(FineToCoarse, ogsHlong,  ogsAdd, A->ogs);

 printf("\n ------ FineToCoarse (RANK  = %d ) 2do aggreates despues------\n",rank);
	for (dlong i=0;i<M;i++)
		if (i%15==0)  printf("\n");
		printf("%d(%d) ",FineToCoarse[i],Ti[i]);
	printf("\n ----------------------\n");



  free(rands);
  free(states);
  free(Tr);
  free(Ts);
  free(Ti);
  free(Tc);

  delete C;
}
*/


typedef struct{
	dlong index;
	dlong Nnbs;
	dlong *nbs;
} nbs_t;


int compareNBS(const void *a, const void *b){
	nbs_t *pa = (nbs_t *)a;	
	nbs_t *pb = (nbs_t *)b;
	
	
	if (pa->Nnbs < pb->Nnbs)	return +1;
	if (pa->Nnbs > pb->Nnbs)	return -1;
	if (pa->index < pa->index )	return +1;
	if (pa->index > pa->index )	return -1;
	
	return 0;
}



void formAggregates(parCSR *A, parCSR *C,
                     hlong* FineToCoarse,
                     hlong* globalAggStarts){

  int rank, size;
  MPI_Comm_rank(A->comm, &rank);
  MPI_Comm_size(A->comm, &size);

  const dlong N   = C->Nrows;
  const dlong M   = C->Ncols;
  const dlong diagNNZ = C->diag->nnz;
  
  printf("\n-----------------------------------------\n");
  printf("rank = %d (LQHSSN) =>  N = %d \t M = %d    ",rank,N,M);
  printf("\n-----------------------------------------\n");

  dlong   *states = (dlong *)    calloc(M, sizeof(dlong));    //  M > N

  for(dlong i=0; i<N; i++)   // initialize states to -1
    states[i] = -1;
  
  hlong *globalRowStarts = A->globalRowStarts;  

  // construct the local neigbors
  nbs_t *V = (nbs_t *) calloc(N,sizeof(nbs_t));
  
  for(dlong i=0; i<N; i++){  
	V[i].index = i;
	V[i].Nnbs  = C->diag->rowStarts[i+1] - C->diag->rowStarts[i];  
	V[i].nbs   = (dlong *) calloc(V[i].Nnbs,sizeof(dlong));
	for (dlong j=0 ; j < V[i].Nnbs ; j++  ){
		V[i].nbs[j] =  C->diag->cols[C->diag->rowStarts[i] + j];
	}
  }
  
  // imprime vecindades
  /*
  printf("\n==================================\n");
  for(dlong i=0; i<N; i++){  
	printf("\n Nbs %d \t\t",i);
	for (dlong j=0 ; j < V[i].Nnbs ; j++  ){
		printf(" %d ",V[i].nbs[j]);
	}
  }
  printf("\n==================================\n");
  */
  
  
  
  int MySort = 1;
  //options.getArgs("SORT",MySort);
  
  // by default the nbs are constructed by local indices
  if (MySort>0)		qsort(V,N,sizeof(nbs_t),compareNBS);  // sort nbs base on strong connections then indices
  
  dlong R_num = 0; 	// number of non-isolated nodes to be used
     
  for (dlong i=0;i<N;i++)
	if(V[i].Nnbs>1)		R_num++;


  //  printf("-----------------------\n")	;
  //printf("R_num = %d\n",R_num);
  //printf("-----------------------\n")	;

  dlong R_nodes[N];   // esta declaracion puede ser problema
  dlong R_pos[N];     // N is number of local elements so at most N aggregates
  
  dlong k = 0;
  
  
  for (dlong i=0;i<N;i++){    
	if(V[i].Nnbs>1){
		R_nodes[k] = V[i].index;	// R_nodes list of nodes to be used
		R_pos[k] = i;				// R_pos indices of the nodes in the nbs
		k++;
	}
  }
  
  /*
  printf("-----------------------\n")	;
  printf("R_num = %d   k = %d  \n",R_num,k);
     printf("-----------------------\n")	;
*/

  dlong Agg_num = 0;  
  
  // First aggregates
  //#pragma omp parallel for
  for(dlong i=0; i<R_num; i++){
	if (states[R_nodes[i]] == -1){
		int ok = 0;
		for(dlong j=0; j<V[R_pos[i]].Nnbs;j++){ 
			if (states[V[R_pos[i]].nbs[j]]>-1){
				ok=1;
				//j = V[R_nodes[i]].Nnbs +10;
				break;
			}
		}				
		if (ok == 0){
			for(dlong j=0; j<V[R_pos[i]].Nnbs;j++){
				states[V[R_pos[i]].nbs[j]] = Agg_num;				
			}				
			Agg_num++;
		}
	 }	 
  }
  
  R_num=0;   // update the number of nodes 

  //for (dlong i=0;i<M;i++)   // number of non-aggregate nodes
  //if (states[i]==-1)		R_num++;
  
  //k = 0;
  
  for (dlong i=0;i<N;i++){  // update list of  non-agreggate nodes
	if (states[V[i].index]==-1){      // cambie k por R_num
		R_nodes[R_num] = V[i].index;
		R_pos[R_num] = i;
		R_num++;					  // since V includes all the nodes this is the number of non-aggregated nodes	
	}
  } 
      
  hlong *psudoAgg = (hlong *) calloc(M,sizeof(hlong));   // overflow with dlong? 

  // count the number of nodes at each aggregate
  for (dlong i=0;i<N;i++)
	if (states[V[i].index]>-1)		psudoAgg[states[V[i].index]]++;

  // #pragma omp parallel for
  for(dlong i=0; i<R_num; i++){
	if (states[R_nodes[i]] == -1){ 	 // sanity check			
		if (V[R_pos[i]].Nnbs>1){  	 // at most one neigbor
			dlong Agg_max;
			dlong posIndx = 0;         
			dlong MoreAgg[Agg_num];
			for (dlong j=0;j<Agg_num;j++)
				MoreAgg[j]=0;		
			// count the number of strong conections with each aggregate		
			for(dlong j=0; j<V[R_pos[i]].Nnbs;j++){  // index 0 is itself
				if (states[V[R_pos[i]].nbs[j]] > -1){
					MoreAgg[states[V[R_pos[i]].nbs[j]]]++;
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
			states[R_nodes[i]] = Agg_num;  // becomes a new aggregate
			psudoAgg[Agg_num]++;
			Agg_num++;								
		}			
	}	 
  }
  
  // csrHaloExchange(A, sizeof(int), states, intSendBuffer, states+A->NlocalCols); since its local this should not be necesary

  dlong *gNumAggs = (dlong *) calloc(size,sizeof(dlong));
  //globalAggStarts = (hlong *) calloc(size+1,sizeof(hlong));
  
  // count the coarse nodes/aggregates in each rannk
  MPI_Allgather(&Agg_num,1,MPI_DLONG,gNumAggs,1,MPI_DLONG,A->comm);

  globalAggStarts[0] = 0;
  for (int r=0;r<size;r++)
    globalAggStarts[r+1] = globalAggStarts[r] + gNumAggs[r];

  // enumerate the coarse nodes/aggregates
  for(dlong i=0; i<N; i++)
    FineToCoarse[i] = globalAggStarts[rank] + states[i];
    
  for(dlong i=N; i<M; i++)
    FineToCoarse[i] = 0;
    
   //share the initial aggregate flags
   //csrHaloExchange(A, sizeof(hlong), FineToCoarse, hlongSendBuffer, FineToCoarse+A->NlocalCols);
 
/* 
 printf("\n rank %d  Agregates antes ==========\n",rank);
  for (dlong i=0;i<M;i++)
  	printf("%d ",FineToCoarse[i]);
  printf("\n=======================================\n");
 */
 
  // share results
  for (dlong n=N;n<M;n++) FineToCoarse[n] = 0;
	ogsGatherScatter(FineToCoarse, ogsHlong,  ogsAdd, A->ogs);
  
  
  /*

  printf("\n rank %d  Agregates despues ==========\n",rank);
  for (dlong i=0;i<M;i++)
  	printf("%d ",FineToCoarse[i]);
  printf("\n=======================================\n");
  */
  
  
  /*
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
