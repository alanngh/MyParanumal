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

#include "insTet3D.h"

// complete a time step using LSERK4
void insAdvectionStepTet3D(ins_t *ins, int tstep, const char* options){

  mesh3D *mesh = ins->mesh;
  dfloat t = tstep*ins->dt; 

  // field offset at this step
  dlong  offset  = mesh->Nelements+mesh->totalHaloPairs;  
  dlong ioffset  = ins->index*offset;
  
  //Exctract Halo On Device
  if(mesh->totalHaloPairs>0){
    ins->totalHaloExtractKernel(mesh->Nelements,
                                mesh->totalHaloPairs,
                                mesh->o_haloElementList,
                                ioffset,
                                ins->o_U,
                                ins->o_V,
                                ins->o_W,
                                ins->o_P,
                                ins->o_tHaloBuffer);

    // copy extracted halo to HOST
    ins->o_tHaloBuffer.copyTo(ins->tSendBuffer);

    // start halo exchange
    meshHaloExchangeStart(mesh,
                          mesh->Np*(ins->NTfields)*sizeof(dfloat),
                          ins->tSendBuffer,
                          ins->tRecvBuffer);
  }

  // Compute Volume Contribution
  if(strstr(options, "CUBATURE")){
    ins->advectionCubatureVolumeKernel(mesh->Nelements,
                                       mesh->o_vgeo,
                                       mesh->o_cubDrWT,
                                       mesh->o_cubDsWT,
                                       mesh->o_cubDtWT,
                                       mesh->o_cubInterpT,
                                       ioffset,
                                       ins->o_U,
                                       ins->o_V,
                                       ins->o_W,
                                       ins->o_NU,
                                       ins->o_NV,
                                       ins->o_NW);
  } else {
    ins->advectionVolumeKernel(mesh->Nelements,
                               mesh->o_vgeo,
                               mesh->o_DrT,
                               mesh->o_DsT,
                               mesh->o_DtT,
                               ioffset,
                               ins->o_U,
                               ins->o_V,
                               ins->o_W,
                               ins->o_NU,
                               ins->o_NV,
                               ins->o_NW);
  }

  // Compute Volume Contribution
  ins->gradientVolumeKernel(mesh->Nelements,
                            mesh->o_vgeo,
                            mesh->o_DrT,
                            mesh->o_DsT,
                            mesh->o_DtT,
                            ioffset,
                            ins->o_P,
                            ins->o_Px,
                            ins->o_Py,
                            ins->o_Pz);

  // COMPLETE HALO EXCHANGE
  if(mesh->totalHaloPairs>0){

    meshHaloExchangeFinish(mesh);

    ins->o_tHaloBuffer.copyFrom(ins->tRecvBuffer);

    ins->totalHaloScatterKernel(mesh->Nelements,
                                mesh->totalHaloPairs,
                                mesh->o_haloElementList,
                                ioffset,
                                ins->o_U,
                                ins->o_V,
                                ins->o_W,
                                ins->o_P,
                                ins->o_tHaloBuffer);
  }


  if(strstr(options, "CUBATURE")){
    ins->advectionCubatureSurfaceKernel(mesh->Nelements,
                                        mesh->o_sgeo,
                                        mesh->o_intInterpT,
                                        mesh->o_intLIFTT,
                                        mesh->o_vmapM,
                                        mesh->o_vmapP,
                                        mesh->o_EToB,
                                        t,
                                        mesh->o_intx,
                                        mesh->o_inty,
                                        mesh->o_intz,
                                        ioffset,
                                        ins->o_U,
                                        ins->o_V,
                                        ins->o_W,
                                        ins->o_NU,
                                        ins->o_NV,
                                        ins->o_NW);
  } else {
    ins->advectionSurfaceKernel(mesh->Nelements,
                                mesh->o_sgeo,
                                mesh->o_LIFTT,
                                mesh->o_vmapM,
                                mesh->o_vmapP,
                                mesh->o_EToB,
                                t,
                                mesh->o_x,
                                mesh->o_y,
                                mesh->o_z,
                                ioffset,
                                ins->o_U,
                                ins->o_V,
                                ins->o_W,
                                ins->o_NU,
                                ins->o_NV,
                                ins->o_NW);
  }

  // Solve pressure gradient for t^(n+1) grad(p^(n+1))
  t += ins->dt;

  if (strstr(ins->pSolverOptions,"IPDG")) {
    const int solverid = 0; // Pressure Solve
    // Compute Surface Conribution
    ins->gradientSurfaceKernel(mesh->Nelements,
                               mesh->o_sgeo,
                               mesh->o_LIFTT,
                               mesh->o_vmapM,
                               mesh->o_vmapP,
                               mesh->o_EToB,
                               mesh->o_x,
                               mesh->o_y,
                               mesh->o_z,
                               t,
                               ins->dt,
                               ins->c0,
                               ins->c1,
                               ins->c2,
                               ins->index,
                               offset,
                               solverid, // pressure BCs
                               ins->o_PI, //not used
                               ins->o_P,
                               ins->o_Px,
                               ins->o_Py,
                               ins->o_Pz);
  }
}
