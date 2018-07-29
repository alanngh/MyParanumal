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

#include "acoustics.h"

dfloat acousticsDopriEstimate(acoustics_t *acoustics){
  
  mesh_t *mesh = acoustics->mesh;
  
  // should insert err = acousticsDopriEstimate2D() here
  //Error estimation 
  //E. HAIRER, S.P. NORSETT AND G. WANNER, SOLVING ORDINARY
  //      DIFFERENTIAL EQUATIONS I. NONSTIFF PROBLEMS. 2ND EDITION.
  dlong Ntotal = mesh->Nelements*mesh->Np*mesh->Nfields;
  acoustics->rkErrorEstimateKernel(Ntotal, 
				   acoustics->ATOL,
				   acoustics->RTOL,
				   acoustics->o_q,
				   acoustics->o_rkq,
				   acoustics->o_rkerr,
				   acoustics->o_errtmp);
  
  acoustics->o_errtmp.copyTo(acoustics->errtmp);
  dfloat localerr = 0;
  dfloat err = 0;
  for(dlong n=0;n<acoustics->Nblock;++n){
    localerr += acoustics->errtmp[n];
  }
  MPI_Allreduce(&localerr, &err, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);

  err = sqrt(err/(mesh->Np*acoustics->totalElements*acoustics->Nfields));
  
  return err;
}
  
