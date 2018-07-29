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

#include "ellipticQuad2D.h"

typedef struct {

  int face;
  int signature[4];
  dlong id;

} refPatch_t;


void matrixInverse(int N, dfloat *A);

dfloat matrixConditionNumber(int N, dfloat *A);

void BuildFaceIpdgPatchAx(mesh2D *mesh, dfloat tau, dfloat lambda, int* BCType,
                        dfloat *B, dfloat *Br, dfloat *Bs, dlong face, dfloat *A);

void BuildReferenceFacePatch(mesh2D *mesh, dfloat tau, dfloat lambda, int* BCType,
                        dfloat *B, dfloat *Br, dfloat *Bs, int face, int *signature, dfloat *A);

dlong getFacePatchIndex(refPatch_t *referencePatchList, int numRefPatches, int face, int *signature);




void ellipticBuildFacePatchesQuad2D(solver_t *solver, dfloat tau, dfloat lambda, int *BCType, dfloat rateTolerance,
                                   dlong *Npatches, dlong **patchesIndex, dfloat **patchesInvA,
                                   const char *options){

  mesh2D *mesh = solver->mesh;

  // build some monolithic basis arrays
  dfloat *B  = (dfloat*) calloc(mesh->Np*mesh->Np, sizeof(dfloat));
  dfloat *Br = (dfloat*) calloc(mesh->Np*mesh->Np, sizeof(dfloat));
  dfloat *Bs = (dfloat*) calloc(mesh->Np*mesh->Np, sizeof(dfloat));

  int mode = 0;
  for(int nj=0;nj<mesh->N+1;++nj){
    for(int ni=0;ni<mesh->N+1;++ni){

      int node = 0;

      for(int j=0;j<mesh->N+1;++j){
        for(int i=0;i<mesh->N+1;++i){

          if(nj==j && ni==i)
            B[mode*mesh->Np+node] = 1;
          if(nj==j)
            Br[mode*mesh->Np+node] = mesh->D[ni+mesh->Nq*i]; 
          if(ni==i)
            Bs[mode*mesh->Np+node] = mesh->D[nj+mesh->Nq*j]; 
          
          ++node;
        }
      }
      ++mode;
    }
  }

  //We need the halo element's EToB flags to make the patch matrices
  if (mesh->totalHaloPairs) {
    mesh->EToB = (int *) realloc(mesh->EToB,(mesh->Nelements+mesh->totalHaloPairs)*sizeof(int));
    int *idSendBuffer = (int *) calloc(mesh->totalHaloPairs,sizeof(int));
    meshHaloExchange(mesh, sizeof(int), mesh->EToB, idSendBuffer, mesh->EToB + mesh->Nelements);
    free(idSendBuffer);
  }

  int NpatchElements = 2;
  int patchNp = mesh->Np*NpatchElements;

  //build a list of all face pairs
  mesh->NfacePairs=0;
  for (dlong eM=0; eM<mesh->Nelements;eM++) {
    for (int f=0;f<mesh->Nfaces;f++) {
      dlong eP = mesh->EToE[eM*mesh->Nfaces+f];

      if (eM<eP) mesh->NfacePairs++;
    }
  }

  mesh->EToFPairs = (dlong *) calloc((mesh->Nelements+mesh->totalHaloPairs)*mesh->Nfaces,sizeof(dlong));
  mesh->FPairsToE = (dlong *) calloc(2*mesh->NfacePairs,sizeof(dlong));
  mesh->FPairsToF = (int *) calloc(2*mesh->NfacePairs,sizeof(int));

  //fill with -1
  for (dlong n=0;n<(mesh->Nelements+mesh->totalHaloPairs)*mesh->Nfaces;n++) {
    mesh->EToFPairs[n] = -1;
  }

  dlong cnt=0;
  for (dlong eM=0; eM<mesh->Nelements;eM++) {
    for (int fM=0;fM<mesh->Nfaces;fM++) {
      dlong eP = mesh->EToE[eM*mesh->Nfaces+fM];

      if (eM<eP) {
        mesh->FPairsToE[2*cnt+0] = eM;
        mesh->FPairsToE[2*cnt+1] = eP;

        int fP = mesh->EToF[eM*mesh->Nfaces+fM];

        mesh->FPairsToF[2*cnt+0] = fM;
        mesh->FPairsToF[2*cnt+1] = fP;

        mesh->EToFPairs[mesh->Nfaces*eM+fM] = 2*cnt+0;
        mesh->EToFPairs[mesh->Nfaces*eP+fP] = 2*cnt+1;

        cnt++;
      }
    }
  }


  (*Npatches) = 0;
  dlong numRefPatches=0;
  dlong refPatches = 0;

  //patch inverse storage
  *patchesInvA = (dfloat*) calloc((*Npatches)*patchNp*patchNp, sizeof(dfloat));
  *patchesIndex = (dlong*) calloc(mesh->NfacePairs, sizeof(dlong));

  refPatch_t *referencePatchList = (refPatch_t *) calloc(numRefPatches,sizeof(refPatch_t));

  //temp patch storage
  dfloat *patchA = (dfloat*) calloc(patchNp*patchNp, sizeof(dfloat));
  dfloat *invRefAA = (dfloat*) calloc(patchNp*patchNp, sizeof(dfloat));

  dfloat *refPatchInvA;

  // loop over all face pairs
  for(dlong face=0;face<mesh->NfacePairs;++face){

    //build the patch A matrix for this element
    BuildFaceIpdgPatchAx(mesh, tau, lambda, BCType, B, Br, Bs, face, patchA);

    dlong eM = mesh->FPairsToE[2*face+0];
    dlong eP = mesh->FPairsToE[2*face+1];
    int fM = mesh->FPairsToF[2*face+0];
    int fP = mesh->FPairsToF[2*face+1];

    dlong eM0 = mesh->EToE[eM*mesh->Nfaces+0];
    dlong eM1 = mesh->EToE[eM*mesh->Nfaces+1];
    dlong eM2 = mesh->EToE[eM*mesh->Nfaces+2];
    dlong eM3 = mesh->EToE[eM*mesh->Nfaces+3];

    dlong eP0 = mesh->EToE[eP*mesh->Nfaces+0];
    dlong eP1 = mesh->EToE[eP*mesh->Nfaces+1];
    dlong eP2 = mesh->EToE[eP*mesh->Nfaces+2];
    dlong eP3 = mesh->EToE[eP*mesh->Nfaces+3];

    if(eM0>=0 && eM1>=0 && eM2>=0 && eM3>=0 &&
       eP0>=0 && eP1>=0 && eP2>=0 && eP3>=0){ //check if this is an interiour patch
      
      //get the vertices
      hlong *vM = mesh->EToV + eM*mesh->Nverts;
      hlong *vP = mesh->EToV + eP*mesh->Nverts;

      // intialize signature to -1
      int signature[4];
      for (int n=0;n<mesh->Nverts;n++) signature[n] = -1;

      for (int n=0;n<mesh->Nverts;n++) {
        for (int m=0;m<mesh->Nverts;m++) {
          if (vP[m] == vM[n]) signature[m] = n; 
        }
      }

      dlong index = getFacePatchIndex(referencePatchList,numRefPatches,fM,signature);
      if (index<0) {      
        //build the reference patch for this signature
        ++(*Npatches);
        numRefPatches++;
        *patchesInvA = (dfloat*) realloc(*patchesInvA, (*Npatches)*patchNp*patchNp*sizeof(dfloat));
        referencePatchList = (refPatch_t *) realloc(referencePatchList,numRefPatches*sizeof(refPatch_t));

        referencePatchList[numRefPatches-1].face = fM; 
        referencePatchList[numRefPatches-1].id = (*Npatches)-1;
        for (int n=0;n<mesh->Nverts;n++) 
          referencePatchList[numRefPatches-1].signature[n] = signature[n];

        refPatchInvA = *patchesInvA + ((*Npatches)-1)*patchNp*patchNp;

        // printf("Building reference patch for the face %d, with signature [%d,%d,%d,%d] \n", fM, signature[0], signature[1], signature[2],signature[3]);

        BuildReferenceFacePatch(mesh, tau, lambda, BCType, B, Br, Bs, fM, signature, refPatchInvA); 
        matrixInverse(patchNp, refPatchInvA);        
        index = (*Npatches)-1;
      }

      refPatchInvA = *patchesInvA + index*patchNp*patchNp;

      //hit the patch with the reference inverse
      for(int n=0;n<patchNp;++n){
        for(int m=0;m<patchNp;++m){
          invRefAA[n*patchNp+m] = 0.;
          for (int k=0;k<patchNp;k++) {
            invRefAA[n*patchNp+m] += refPatchInvA[n*patchNp+k]*patchA[k*patchNp+m];
          }
        }
      }

      dfloat cond = matrixConditionNumber(patchNp,invRefAA);
      dfloat rate = (sqrt(cond)-1.)/(sqrt(cond)+1.);

      //printf("Face pair %d's conditioned patch reports cond = %g and rate = %g \n", eM, cond, rate);

      //if the convergence rate is good, use the reference patch, and skip adding this patch
      if (rate < rateTolerance) {
        (*patchesIndex)[face] = index;
        refPatches++;
        continue;
      }
    }
    //add this patch to the patch list
    ++(*Npatches);
    *patchesInvA = (dfloat*) realloc(*patchesInvA, (*Npatches)*patchNp*patchNp*sizeof(dfloat));

    matrixInverse(patchNp, patchA);

    //copy inverse into patchesInvA
    for(int n=0;n<patchNp;++n){
      for(int m=0;m<patchNp;++m){
        dlong id = ((*Npatches)-1)*patchNp*patchNp + n*patchNp + m;
        (*patchesInvA)[id] = patchA[n*patchNp+m];
      }
    }

    (*patchesIndex)[face] = (*Npatches)-1;
  }

  printf("saving "dlongFormat" full patches\n",*Npatches);
  printf("using "dlongFormat" reference patches\n", refPatches);

  free(patchA); free(invRefAA);
  free(B); free(Br); free(Bs);
}


void BuildFaceIpdgPatchAx(mesh2D *mesh, dfloat tau, dfloat lambda, int* BCType,
                        dfloat *B, dfloat *Br, dfloat *Bs, dlong face, dfloat *A) {

  int NpatchElements = 2;
  int patchNp = NpatchElements*mesh->Np;

  // Extract patches
  // B  a
  // a' C

  //start with diagonals
  for(int N=0;N<NpatchElements;++N){
    //element number
    dlong e = mesh->FPairsToE[2*face+N];

    for(int n=0;n<mesh->Np;++n){
      for(int m=0;m<mesh->Np;++m){
        int id = N*mesh->Np*patchNp + n*patchNp + N*mesh->Np + m;
        A[id] = 0;

        // (grad phi_n, grad phi_m)_{D^e}
        for(int i=0;i<mesh->Np;++i){
          dlong base = e*mesh->Np*mesh->Nvgeo + i;
          dfloat drdx = mesh->vgeo[base+mesh->Np*RXID];
          dfloat drdy = mesh->vgeo[base+mesh->Np*RYID];
          dfloat dsdx = mesh->vgeo[base+mesh->Np*SXID];
          dfloat dsdy = mesh->vgeo[base+mesh->Np*SYID];
          dfloat JW   = mesh->vgeo[base+mesh->Np*JWID];
          
          int idn = n*mesh->Np+i;
          int idm = m*mesh->Np+i;
          dfloat dlndx = drdx*Br[idn] + dsdx*Bs[idn];
          dfloat dlndy = drdy*Br[idn] + dsdy*Bs[idn];
          dfloat dlmdx = drdx*Br[idm] + dsdx*Bs[idm];
          dfloat dlmdy = drdy*Br[idm] + dsdy*Bs[idm];
          A[id] += JW*(dlndx*dlmdx+dlndy*dlmdy);
          A[id] += lambda*JW*B[idn]*B[idm];
        }
    

        for (int fM=0;fM<mesh->Nfaces;fM++) {
          // accumulate flux terms for negative and positive traces
          for(int i=0;i<mesh->Nfp;++i){
            int vidM = mesh->faceNodes[i+fM*mesh->Nfp];

            // grab vol geofacs at surface nodes
            dlong baseM = e*mesh->Np*mesh->Nvgeo + vidM;
            dfloat drdxM = mesh->vgeo[baseM+mesh->Np*RXID];
            dfloat drdyM = mesh->vgeo[baseM+mesh->Np*RYID];
            dfloat dsdxM = mesh->vgeo[baseM+mesh->Np*SXID];
            dfloat dsdyM = mesh->vgeo[baseM+mesh->Np*SYID];

            // grab surface geometric factors
            dlong base = mesh->Nsgeo*(e*mesh->Nfp*mesh->Nfaces + fM*mesh->Nfp + i);
            dfloat nx = mesh->sgeo[base+NXID];
            dfloat ny = mesh->sgeo[base+NYID];
            dfloat wsJ = mesh->sgeo[base+WSJID];
            dfloat hinv = mesh->sgeo[base+IHID];
            
            // form negative trace terms in IPDG
            int idnM = n*mesh->Np+vidM; 
            int idmM = m*mesh->Np+vidM;

            dfloat dlndxM = drdxM*Br[idnM] + dsdxM*Bs[idnM];
            dfloat dlndyM = drdyM*Br[idnM] + dsdyM*Bs[idnM];
            dfloat ndotgradlnM = nx*dlndxM+ny*dlndyM;
            dfloat lnM = B[idnM];

            dfloat dlmdxM = drdxM*Br[idmM] + dsdxM*Bs[idmM];
            dfloat dlmdyM = drdyM*Br[idmM] + dsdyM*Bs[idmM];
            dfloat ndotgradlmM = nx*dlmdxM+ny*dlmdyM;
            dfloat lmM = B[idmM];
            
            dfloat penalty = tau*hinv;     
            int bc = mesh->EToB[fM+mesh->Nfaces*e]; //raw boundary flag

            int bcD = 0, bcN =0;
            int bcType = 0;

            if(bc>0) bcType = BCType[bc];          //find its type (Dirichlet/Neumann)

            // this needs to be double checked (and the code where these are used)
            if(bcType==1){ // Dirichlet
              bcD = 1;
              bcN = 0;
            } else if(bcType==2){ // Neumann
              bcD = 0;
              bcN = 1;
            }   

            A[id] += -0.5*(1+bcD)*(1-bcN)*wsJ*lnM*ndotgradlmM;  // -(ln^-, N.grad lm^-)
            A[id] += -0.5*(1+bcD)*(1-bcN)*wsJ*ndotgradlnM*lmM;  // -(N.grad ln^-, lm^-)
            A[id] += +0.5*(1+bcD)*(1-bcN)*wsJ*penalty*lnM*lmM; // +((tau/h)*ln^-,lm^-)
          }
        }
      }
    }
  }

  //now the off-diagonal
  dlong eM = mesh->FPairsToE[2*face+0];
  dlong eP = mesh->FPairsToE[2*face+1];
  int fM = mesh->FPairsToF[2*face+0];

  // accumulate flux terms for positive traces
  for(int n=0;n<mesh->Np;++n){
    for(int m=0;m<mesh->Np;++m){
      int id = n*patchNp + mesh->Np + m;
      A[id] = 0;
      
      for(int i=0;i<mesh->Nfp;++i){
        int vidM = mesh->faceNodes[i+fM*mesh->Nfp];

        // grab vol geofacs at surface nodes
        dlong baseM = eM*mesh->Np*mesh->Nvgeo + vidM;
        dfloat drdxM = mesh->vgeo[baseM+mesh->Np*RXID];
        dfloat drdyM = mesh->vgeo[baseM+mesh->Np*RYID];
        dfloat dsdxM = mesh->vgeo[baseM+mesh->Np*SXID];
        dfloat dsdyM = mesh->vgeo[baseM+mesh->Np*SYID];

        // double check vol geometric factors are in halo storage of vgeo
        dlong idM   = eM*mesh->Nfp*mesh->Nfaces+fM*mesh->Nfp+i;
        int vidP    = (int) (mesh->vmapP[idM]%mesh->Np); // only use this to identify location of positive trace vgeo
        dlong baseP = eP*mesh->Np*mesh->Nvgeo + vidP; // use local offset for vgeo in halo
        dfloat drdxP = mesh->vgeo[baseP+mesh->Np*RXID];
        dfloat drdyP = mesh->vgeo[baseP+mesh->Np*RYID];
        dfloat dsdxP = mesh->vgeo[baseP+mesh->Np*SXID];
        dfloat dsdyP = mesh->vgeo[baseP+mesh->Np*SYID];
        
        // grab surface geometric factors
        dlong base = mesh->Nsgeo*(eM*mesh->Nfp*mesh->Nfaces + fM*mesh->Nfp + i);
        dfloat nx = mesh->sgeo[base+NXID];
        dfloat ny = mesh->sgeo[base+NYID];
        dfloat wsJ = mesh->sgeo[base+WSJID];
        dfloat hinv = mesh->sgeo[base+IHID];
        
        // form trace terms in IPDG
        int idnM = n*mesh->Np+vidM; 
        int idmP = m*mesh->Np+vidP;

        dfloat dlndxM = drdxM*Br[idnM] + dsdxM*Bs[idnM];
        dfloat dlndyM = drdyM*Br[idnM] + dsdyM*Bs[idnM];
        dfloat ndotgradlnM = nx*dlndxM+ny*dlndyM;
        dfloat lnM = B[idnM];
        
        dfloat dlmdxP = drdxP*Br[idmP] + dsdxP*Bs[idmP];
        dfloat dlmdyP = drdyP*Br[idmP] + dsdyP*Bs[idmP];
        dfloat ndotgradlmP = nx*dlmdxP+ny*dlmdyP;
        dfloat lmP = B[idmP];
        
        dfloat penalty = tau*hinv;     
        
        A[id] += -0.5*wsJ*lnM*ndotgradlmP;  // -(ln^-, N.grad lm^+)
        A[id] += +0.5*wsJ*ndotgradlnM*lmP;  // +(N.grad ln^-, lm^+)
        A[id] += -0.5*wsJ*penalty*lnM*lmP; // -((tau/h)*ln^-,lm^+)
      }
    }
  }

  //write the transpose of the off-diagonal block
  for(int n=0;n<mesh->Np;++n){
    for(int m=0;m<mesh->Np;++m){
      int id  = n*patchNp + mesh->Np + m;
      int idT = mesh->Np*patchNp + m*patchNp + n;

      A[idT] = A[id];
    }
  }
}

void BuildReferenceFacePatch(mesh2D *mesh, dfloat tau, dfloat lambda, int* BCType,
                        dfloat *B, dfloat *Br, dfloat *Bs, int face, int *signature, dfloat *A) {
  //build a mini mesh struct for the reference patch
  mesh2D *refMesh = (mesh2D*) calloc(1,sizeof(mesh2D));
  memcpy(refMesh,mesh,sizeof(mesh2D));

   //vertices of reference patch
  int Nv = 12;
  dfloat VX[12] = {-1, 1, 1,-1,-1, 1, 3, 3, 1,-1,-3,-3};
  dfloat VY[12] = {-1,-1, 1, 1,-3,-3,-1, 1, 3, 3, 1,-1};

  hlong EToV[5*4] = {0,1,2,3,
                    1,0,4,5,
                    2,1,6,7,
                    3,2,8,9,
                    0,3,10,11};

  int NpatchElements = 2;                    
  refMesh->Nelements = NpatchElements;

  refMesh->EX = (dfloat *) calloc(mesh->Nverts*NpatchElements,sizeof(dfloat));
  refMesh->EY = (dfloat *) calloc(mesh->Nverts*NpatchElements,sizeof(dfloat));

  refMesh->EToV = (hlong*) calloc(NpatchElements*mesh->Nverts, sizeof(hlong));

  for(int n=0;n<mesh->Nverts;++n){
    hlong v = EToV[n];
    refMesh->EX[n] = VX[v];
    refMesh->EY[n] = VY[v];
    refMesh->EToV[n] = v;
  } 

  int cnt[4] = {0,0,0,0};
  for (int n=0;n<mesh->Nverts;n++) {
    if (signature[n]==-1) {
      //fill the missing vertex based on the face number
      hlong v = EToV[(face+1)*mesh->Nverts+mesh->Nverts-2 + cnt[face]];  
      refMesh->EX[mesh->Nverts+n] = VX[v];
      refMesh->EY[mesh->Nverts+n] = VY[v];
      refMesh->EToV[mesh->Nverts+n] = mesh->Nverts+cnt[face]; //extra verts      
      cnt[face]++;
    } else {
      int v = signature[n];  
      refMesh->EX[mesh->Nverts+n] = VX[v];
      refMesh->EY[mesh->Nverts+n] = VY[v];
      refMesh->EToV[mesh->Nverts+n] = v;      
    }
  }  

  refMesh->EToB = (int*) calloc(NpatchElements*mesh->Nfaces,sizeof(int));
  for (int n=0;n<NpatchElements*mesh->Nfaces;n++) refMesh->EToB[n] = 0;

  //build a list of all face pairs
  refMesh->NfacePairs=1;

  refMesh->EToFPairs = (dlong *) calloc(2*mesh->Nfaces,sizeof(dlong));
  refMesh->FPairsToE = (dlong *) calloc(2,sizeof(dlong));
  refMesh->FPairsToF = (int *)  calloc(2,sizeof(int));

  //fill with -1
  for (int n=0;n<2*mesh->Nfaces;n++)  refMesh->EToFPairs[n] = -1;

  refMesh->FPairsToE[0] = 0;
  refMesh->FPairsToE[1] = 1;
  refMesh->FPairsToF[0] = face;
  refMesh->FPairsToF[1] = 0;
  refMesh->EToFPairs[0] = 0;
  refMesh->EToFPairs[refMesh->Nfaces] = 1;

  meshConnect(refMesh);
  meshLoadReferenceNodesQuad2D(refMesh, mesh->N);
  meshPhysicalNodesQuad2D(refMesh);
  meshGeometricFactorsQuad2D(refMesh);
  meshConnectFaceNodes2D(refMesh);
  meshSurfaceGeometricFactorsQuad2D(refMesh);

  //build this reference patch
  BuildFaceIpdgPatchAx(refMesh, tau, lambda, BCType, B, Br, Bs, 0, A);

  free(refMesh->EX);
  free(refMesh->EY);
  free(refMesh->EToV);

  free(refMesh);
}

dlong getFacePatchIndex(refPatch_t *referencePatchList, dlong numRefPatches, int face, int *signature) {

  dlong index = -1;
  for (dlong n=0;n<numRefPatches;n++) {
    if (referencePatchList[n].face == face) {
      if ((referencePatchList[n].signature[0] == signature[0]) &&
          (referencePatchList[n].signature[1] == signature[1]) && 
          (referencePatchList[n].signature[2] == signature[2]) &&
          (referencePatchList[n].signature[3] == signature[3])) {
        index = referencePatchList[n].id;
        break;
      }
    }
  }
  return index;
}





