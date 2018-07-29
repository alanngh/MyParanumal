/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#include "ellipticTet3D.h"

solver_t *ellipticSolveSetupTet3D(mesh_t *mesh, dfloat tau, dfloat lambda, int*BCType,
                      occa::kernelInfo &kernelInfo, const char *options, const char *parAlmondOptions){

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  dlong Ntotal = mesh->Np*mesh->Nelements;
  dlong Nblock = (Ntotal+blockSize-1)/blockSize;
  dlong Nhalo = mesh->Np*mesh->totalHaloPairs;
  dlong Nall   = Ntotal + Nhalo;

  solver_t *solver = (solver_t*) calloc(1, sizeof(solver_t));

  solver->tau = tau;

  solver->mesh = mesh;

  solver->BCType = BCType;

  solver->p   = (dfloat*) calloc(Nall,   sizeof(dfloat));
  solver->z   = (dfloat*) calloc(Nall,   sizeof(dfloat));
  solver->Ax  = (dfloat*) calloc(Nall,   sizeof(dfloat));
  solver->Ap  = (dfloat*) calloc(Nall,   sizeof(dfloat));
  solver->tmp = (dfloat*) calloc(Nblock, sizeof(dfloat));

  solver->grad = (dfloat*) calloc(Nall*4, sizeof(dfloat));

  solver->o_p   = mesh->device.malloc(Nall*sizeof(dfloat), solver->p);
  solver->o_rtmp= mesh->device.malloc(Nall*sizeof(dfloat), solver->p);
  solver->o_z   = mesh->device.malloc(Nall*sizeof(dfloat), solver->z);

  solver->o_res = mesh->device.malloc(Nall*sizeof(dfloat), solver->z);
  solver->o_Sres = mesh->device.malloc(Nall*sizeof(dfloat), solver->z);
  solver->o_Ax  = mesh->device.malloc(Nall*sizeof(dfloat), solver->p);
  solver->o_Ap  = mesh->device.malloc(Nall*sizeof(dfloat), solver->Ap);
  solver->o_tmp = mesh->device.malloc(Nblock*sizeof(dfloat), solver->tmp);

  solver->o_grad  = mesh->device.malloc(Nall*4*sizeof(dfloat), solver->grad);

  //setup async halo stream
  solver->defaultStream = mesh->defaultStream;
  solver->dataStream = mesh->dataStream;

  dlong Nbytes = mesh->totalHaloPairs*mesh->Np*sizeof(dfloat);
  if(Nbytes>0){
    occa::memory o_sendBuffer = mesh->device.mappedAlloc(Nbytes, NULL);
    occa::memory o_recvBuffer = mesh->device.mappedAlloc(Nbytes, NULL);

    solver->sendBuffer = (dfloat*) o_sendBuffer.getMappedPointer();
    solver->recvBuffer = (dfloat*) o_recvBuffer.getMappedPointer();

    occa::memory o_gradSendBuffer = mesh->device.mappedAlloc(2*Nbytes, NULL);
    occa::memory o_gradRecvBuffer = mesh->device.mappedAlloc(2*Nbytes, NULL);

    solver->gradSendBuffer = (dfloat*) o_gradSendBuffer.getMappedPointer();
    solver->gradRecvBuffer = (dfloat*) o_gradRecvBuffer.getMappedPointer();
  }else{
    solver->sendBuffer = NULL;
    solver->recvBuffer = NULL;
  }
  mesh->device.setStream(solver->defaultStream);

  solver->type = strdup(dfloatString);

  solver->Nblock = Nblock;

  //fill geometric factors in halo
  if(mesh->totalHaloPairs){
    dfloat *vgeoSendBuffer = (dfloat*) calloc(mesh->totalHaloPairs*mesh->Nvgeo, sizeof(dfloat));

    // import geometric factors from halo elements
    mesh->vgeo = (dfloat*) realloc(mesh->vgeo, (mesh->Nelements+mesh->totalHaloPairs)*mesh->Nvgeo*sizeof(dfloat));

    meshHaloExchange(mesh,
         mesh->Nvgeo*sizeof(dfloat),
         mesh->vgeo,
         vgeoSendBuffer,
         mesh->vgeo + mesh->Nelements*mesh->Nvgeo);

    mesh->o_vgeo =
      mesh->device.malloc((mesh->Nelements + mesh->totalHaloPairs)*mesh->Nvgeo*sizeof(dfloat), mesh->vgeo);
    free(vgeoSendBuffer);
  }

  //build inverse of mass matrix
  mesh->invMM = (dfloat *) calloc(mesh->Np*mesh->Np,sizeof(dfloat));
  for (int n=0;n<mesh->Np*mesh->Np;n++)
    mesh->invMM[n] = mesh->MM[n];
  matrixInverse(mesh->Np,mesh->invMM);

  //check all the bounaries for a Dirichlet
  bool allNeumann = (lambda==0) ? true :false;
  solver->allNeumannPenalty = 1;
  hlong localElements = (hlong) mesh->Nelements;
  hlong totalElements = 0;
  MPI_Allreduce(&localElements, &totalElements, 1, MPI_HLONG, MPI_SUM, MPI_COMM_WORLD);
  solver->allNeumannScale = 1.0/sqrt(mesh->Np*totalElements);

  solver->EToB = (int *) calloc(mesh->Nelements*mesh->Nfaces,sizeof(int));
  for (dlong e=0;e<mesh->Nelements;e++) {
    for (int f=0;f<mesh->Nfaces;f++) {
      int bc = mesh->EToB[e*mesh->Nfaces+f];
      if (bc>0) {
        int BC = BCType[bc]; //get the type of boundary
        solver->EToB[e*mesh->Nfaces+f] = BC; //record it
        if (BC!=2) allNeumann = false; //check if its a Dirchlet
      }
    }
  }
  MPI_Allreduce(&allNeumann, &(solver->allNeumann), 1, MPI::BOOL, MPI_LAND, MPI_COMM_WORLD);
  if (rank==0&&strstr(options,"VERBOSE")) printf("allNeumann = %d \n", solver->allNeumann);

  //set surface mass matrix for continuous boundary conditions
  mesh->sMT = (dfloat *) calloc(mesh->Np*mesh->Nfaces*mesh->Nfp,sizeof(dfloat));
  for (int n=0;n<mesh->Np;n++) {
    for (int m=0;m<mesh->Nfp*mesh->Nfaces;m++) {
      dfloat MSnm = 0;
      for (int i=0;i<mesh->Np;i++){
        MSnm += mesh->MM[n+i*mesh->Np]*mesh->LIFT[m+i*mesh->Nfp*mesh->Nfaces];
      }
      mesh->sMT[n+m*mesh->Np]  = MSnm;
    }
  }
  mesh->o_sMT = mesh->device.malloc(mesh->Np*mesh->Nfaces*mesh->Nfp*sizeof(dfloat), mesh->sMT);

  //copy boundary flags
  solver->o_EToB = mesh->device.malloc(mesh->Nelements*mesh->Nfaces*sizeof(int), solver->EToB);

  //if (rank!=0) 
    occa::setVerboseCompilation(false);

  //add standard boundary functions
  char *boundaryHeaderFileName;
  boundaryHeaderFileName = strdup(DHOLMES "/examples/ellipticTet3D/ellipticBoundary3D.h");
  kernelInfo.addInclude(boundaryHeaderFileName);

  kernelInfo.addParserFlag("automate-add-barriers", "disabled");
  if(mesh->device.mode()=="CUDA"){ 
    kernelInfo.addCompilerFlag("-Xptxas -dlcm=ca");
  }

  kernelInfo.addDefine("p_blockSize", blockSize);

  // add custom defines
  kernelInfo.addDefine("p_NpP", (mesh->Np+mesh->Nfp*mesh->Nfaces));
  kernelInfo.addDefine("p_Nverts", mesh->Nverts);

  //sizes for the coarsen and prolongation kernels. degree N to degree 1
  kernelInfo.addDefine("p_NpFine", mesh->Np);
  kernelInfo.addDefine("p_NpCoarse", mesh->Nverts);

  kernelInfo.addDefine("p_NpFEM", mesh->NpFEM);

  int Nmax = mymax(mesh->Np, mesh->Nfaces*mesh->Nfp);
  kernelInfo.addDefine("p_Nmax", Nmax);

  int NblockV = mymax(1,256/mesh->Np); // get close to 256 threads
  kernelInfo.addDefine("p_NblockV", NblockV);

  int NblockP = mymax(1,256/(5*mesh->Np)); // get close to 256 threads
  kernelInfo.addDefine("p_NblockP", NblockP);

  int NblockG;
  if(mesh->Np<=32) NblockG = ( 32/mesh->Np );
  else NblockG = mymax(1,256/mesh->Np);
  kernelInfo.addDefine("p_NblockG", NblockG);

  for (int r=0;r<size;r++) {
    if (r==rank) {
      mesh->haloExtractKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/meshHaloExtract3D.okl",
    				       "meshHaloExtract3D",
    				       kernelInfo);

      mesh->gatherKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/gather.okl",
    				       "gather",
    				       kernelInfo);

      mesh->scatterKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/scatter.okl",
    				       "scatter",
    				       kernelInfo);

      mesh->gatherScatterKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/gatherScatter.okl",
                   "gatherScatter",
                   kernelInfo);

      mesh->getKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/get.okl",
    				       "get",
    				       kernelInfo);

      mesh->putKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/put.okl",
    				       "put",
    				       kernelInfo);

      mesh->sumKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/sum.okl",
                   "sum",
                   kernelInfo);

      mesh->addScalarKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/addScalar.okl",
                   "addScalar",
                   kernelInfo);

      mesh->maskKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/mask.okl",
                   "mask",
                   kernelInfo);

      solver->AxKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticAxTet3D.okl",
                   "ellipticAxTet3D",
                   kernelInfo);

      solver->partialAxKernel =
          mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticAxTet3D.okl",
                     "ellipticPartialAxTet3D",
                     kernelInfo);

      solver->weightedInnerProduct1Kernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/weightedInnerProduct1.okl",
    				       "weightedInnerProduct1",
    				       kernelInfo);

      solver->weightedInnerProduct2Kernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/weightedInnerProduct2.okl",
    				       "weightedInnerProduct2",
    				       kernelInfo);

      solver->innerProductKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/innerProduct.okl",
    				       "innerProduct",
    				       kernelInfo);

      solver->scaledAddKernel =
          mesh->device.buildKernelFromSource(DHOLMES "/okl/scaledAdd.okl",
    					 "scaledAdd",
    					 kernelInfo);

      solver->dotMultiplyKernel =
          mesh->device.buildKernelFromSource(DHOLMES "/okl/dotMultiply.okl",
    					 "dotMultiply",
    					 kernelInfo);

      solver->dotDivideKernel =
          mesh->device.buildKernelFromSource(DHOLMES "/okl/dotDivide.okl",
    					 "dotDivide",
    					 kernelInfo);


      solver->gradientKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticGradientTet3D.okl",
    				       "ellipticGradientTet3D",
    					 kernelInfo);

      solver->partialGradientKernel =
          mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticGradientTet3D.okl",
                 "ellipticPartialGradientTet3D",
               kernelInfo);

      solver->ipdgKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticAxIpdgTet3D.okl",
    				       "ellipticAxIpdgTet3D",
    				       kernelInfo);

      solver->partialIpdgKernel =
            mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticAxIpdgTet3D.okl",
                       "ellipticPartialAxIpdgTet3D",
                       kernelInfo);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  //on-host version of gather-scatter
  int verbose = strstr(options,"VERBOSE") ? 1:0;
  mesh->hostGsh = gsParallelGatherScatterSetup(mesh->Nelements*mesh->Np, mesh->globalIds, verbose);

  // set up separate gather scatter infrastructure for halo and non halo nodes
  ellipticParallelGatherScatterSetup(solver, options);

  //make a node-wise bc flag using the gsop (prioritize Dirichlet boundaries over Neumann)
  solver->mapB = (int *) calloc(mesh->Nelements*mesh->Np,sizeof(int));
  for (dlong e=0;e<mesh->Nelements;e++) {
    for (int n=0;n<mesh->Np;n++) solver->mapB[n+e*mesh->Np] = 1E9;
    for (int f=0;f<mesh->Nfaces;f++) {
      int bc = mesh->EToB[f+e*mesh->Nfaces];
      if (bc>0) {
        for (int n=0;n<mesh->Nfp;n++) {
          int BCFlag = BCType[bc];
          int fid = mesh->faceNodes[n+f*mesh->Nfp];
          solver->mapB[fid+e*mesh->Np] = mymin(BCFlag,solver->mapB[fid+e*mesh->Np]);
        }
      }
    }
  }
  gsParallelGatherScatter(mesh->hostGsh, solver->mapB, "int", "min"); 

  //use the bc flags to find masked ids
  solver->Nmasked = 0;
  for (dlong n=0;n<mesh->Nelements*mesh->Np;n++) {
    if (solver->mapB[n] == 1E9) {
      solver->mapB[n] = 0.;
    } else if (solver->mapB[n] == 1) { //Dirichlet boundary
      solver->Nmasked++;
    }
  }
  solver->o_mapB = mesh->device.malloc(mesh->Nelements*mesh->Np*sizeof(int), solver->mapB);
  
  solver->maskIds = (dlong *) calloc(solver->Nmasked, sizeof(dlong));
  solver->Nmasked =0; //reset
  for (dlong n=0;n<mesh->Nelements*mesh->Np;n++) {
    if (solver->mapB[n] == 1) solver->maskIds[solver->Nmasked++] = n;
  }
  if (solver->Nmasked) solver->o_maskIds = mesh->device.malloc(solver->Nmasked*sizeof(dlong), solver->maskIds);


  solver->precon = (precon_t*) calloc(1, sizeof(precon_t));

  for (int r=0;r<size;r++) {
    if (r==rank) {      
      #if 0

      solver->precon->overlappingPatchKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticOasPreconTet3D.okl",
                   "ellipticOasPreconTet3D",
                   kernelInfo);
      #endif

      solver->precon->restrictKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPreconRestrictTet3D.okl",
                   "ellipticFooTet3D",
                   kernelInfo);

      solver->precon->coarsenKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPreconCoarsen.okl",
                   "ellipticPreconCoarsen",
                   kernelInfo);

      solver->precon->prolongateKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPreconProlongate.okl",
                   "ellipticPreconProlongate",
                   kernelInfo);


      solver->precon->blockJacobiKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticBlockJacobiPreconTet3D.okl",
                   "ellipticBlockJacobiPreconTet3D",
                   kernelInfo);

      solver->precon->partialblockJacobiKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticBlockJacobiPreconTet3D.okl",
                   "ellipticPartialBlockJacobiPreconTet3D",
                   kernelInfo);

      solver->precon->approxPatchSolverKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPatchSolver3D.okl",
                   "ellipticApproxPatchSolver3D",
                   kernelInfo);

      solver->precon->exactPatchSolverKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPatchSolver3D.okl",
                   "ellipticExactPatchSolver3D",
                   kernelInfo);

      solver->precon->patchGatherKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPatchGather.okl",
                   "ellipticPatchGather",
                   kernelInfo);

      solver->precon->approxFacePatchSolverKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPatchSolver3D.okl",
                   "ellipticApproxFacePatchSolver3D",
                   kernelInfo);

      solver->precon->exactFacePatchSolverKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPatchSolver3D.okl",
                   "ellipticExactFacePatchSolver3D",
                   kernelInfo);

      solver->precon->facePatchGatherKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPatchGather.okl",
                   "ellipticFacePatchGather",
                   kernelInfo);

      solver->precon->approxBlockJacobiSolverKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPatchSolver3D.okl",
                   "ellipticApproxBlockJacobiSolver3D",
                   kernelInfo);

      solver->precon->exactBlockJacobiSolverKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticPatchSolver3D.okl",
                   "ellipticExactBlockJacobiSolver3D",
                   kernelInfo);

      solver->precon->SEMFEMInterpKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticSEMFEMInterp.okl",
                   "ellipticSEMFEMInterp",
                   kernelInfo);

      solver->precon->SEMFEMAnterpKernel =
        mesh->device.buildKernelFromSource(DHOLMES "/okl/ellipticSEMFEMAnterp.okl",
                   "ellipticSEMFEMAnterp",
                   kernelInfo);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  long long int pre = mesh->device.memoryAllocated();

  occaTimerTic(mesh->device,"PreconditionerSetup");
  ellipticPreconditionerSetupTet3D(solver, solver->ogs, tau, lambda, BCType,  options, parAlmondOptions);
  occaTimerToc(mesh->device,"PreconditionerSetup");

  long long int usedBytes = mesh->device.memoryAllocated()-pre;

  solver->precon->preconBytes = usedBytes;

  return solver;
}