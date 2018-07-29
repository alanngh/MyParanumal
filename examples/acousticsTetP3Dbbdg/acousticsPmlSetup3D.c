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

#include "acoustics3D.h"

void acousticsPmlSetup3D(mesh3D *mesh){

  //constant pml absorption coefficient
  dfloat xsigma = 80, ysigma = 80, zsigma = 80;

  //construct element and halo lists
  mesh->MRABpmlNelements     = (int *) calloc(mesh->MRABNlevels,sizeof(int));
  mesh->MRABpmlNhaloElements = (int *) calloc(mesh->MRABNlevels,sizeof(int));
  
  mesh->MRABpmlElementIds = (int **) calloc(mesh->MRABNlevels,sizeof(int*));
  mesh->MRABpmlIds        = (int **) calloc(mesh->MRABNlevels,sizeof(int*));

  mesh->MRABpmlHaloElementIds = (int **) calloc(mesh->MRABNlevels,sizeof(int*));
  mesh->MRABpmlHaloIds        = (int **) calloc(mesh->MRABNlevels,sizeof(int*));

  mesh->MRABpmlNelP      = (int **) calloc(mesh->MRABNlevels,sizeof(int*));
  mesh->MRABpmlNhaloEleP = (int **) calloc(mesh->MRABNlevels,sizeof(int*));
  
  mesh->MRABpmlElIdsP = (int ***) calloc(mesh->MRABNlevels,sizeof(int**));
  mesh->MRABpmlIdsP   = (int ***) calloc(mesh->MRABNlevels,sizeof(int**));  
  
  mesh->MRABpmlHaloEleIdsP = (int ***) calloc(mesh->MRABNlevels,sizeof(int**));
  mesh->MRABpmlHaloIdsP    = (int ***) calloc(mesh->MRABNlevels,sizeof(int**));
  
  for (int lev=0;lev<mesh->MRABNlevels;lev++) {
    mesh->MRABpmlNelP[lev]      = (int *) calloc(mesh->NMax+1,sizeof(int));
    mesh->MRABpmlNhaloEleP[lev] = (int *) calloc(mesh->NMax+1,sizeof(int));
    mesh->MRABpmlElIdsP[lev] = (int **) calloc(mesh->NMax+1,sizeof(int *));
    mesh->MRABpmlIdsP[lev]   = (int **) calloc(mesh->NMax+1,sizeof(int *));  
    mesh->MRABpmlHaloEleIdsP[lev] = (int **) calloc(mesh->NMax+1,sizeof(int *));
    mesh->MRABpmlHaloIdsP[lev]    = (int **) calloc(mesh->NMax+1,sizeof(int *));
  }  

  //count the pml elements
  mesh->pmlNelements=0;
  for (int lev =0;lev<mesh->MRABNlevels;lev++){
    for (int m=0;m<mesh->MRABNelements[lev];m++) {
      int e = mesh->MRABelementIds[lev][m];
      int type = mesh->elementInfo[e];
      if ((type==100)||(type==200)||(type==300)
          ||(type==400)||(type==500)||(type==600)||(type==700)){
        mesh->pmlNelements++;
        mesh->MRABpmlNelements[lev]++;
        mesh->MRABpmlNelP[lev][mesh->N[e]]++;
      }
    }
    for (int m=0;m<mesh->MRABNhaloElements[lev];m++) {
      int e = mesh->MRABhaloIds[lev][m];
      int type = mesh->elementInfo[e];
      if ((type==100)||(type==200)||(type==300)
          ||(type==400)||(type==500)||(type==600)||(type==700)){
        mesh->MRABpmlNhaloElements[lev]++;
        mesh->MRABpmlNhaloEleP[lev][mesh->N[e]]++;
      }
    }
  }

  //set up the pml
  if (mesh->pmlNelements) {

    //construct a numbering of the pml elements
    int *pmlIds = (int *) calloc(mesh->Nelements,sizeof(int));
    int pmlcnt = 0;
    for (int e=0;e<mesh->Nelements;e++) {
      int type = mesh->elementInfo[e];
      if ((type==100)||(type==200)||(type==300)
          ||(type==400)||(type==500)||(type==600)||(type==700))  //pml element
        pmlIds[e] = pmlcnt++;
    }

    //set up lists of pml elements and remove the pml elements from the nonpml MRAB lists
    for (int lev =0;lev<mesh->MRABNlevels;lev++){
      mesh->MRABpmlElementIds[lev] = (int *) calloc(mesh->MRABpmlNelements[lev],sizeof(int));
      mesh->MRABpmlIds[lev]        = (int *) calloc(mesh->MRABpmlNelements[lev],sizeof(int));
      mesh->MRABpmlHaloElementIds[lev] = (int *) calloc(mesh->MRABpmlNhaloElements[lev],sizeof(int));
      mesh->MRABpmlHaloIds[lev]        = (int *) calloc(mesh->MRABpmlNhaloElements[lev],sizeof(int));

      for (int p=0;p<=mesh->NMax;p++) {
        mesh->MRABpmlElIdsP[lev][p] = (int *) calloc(mesh->MRABpmlNelP[lev][p],sizeof(int));
        mesh->MRABpmlIdsP[lev][p]   = (int *) calloc(mesh->MRABpmlNelP[lev][p],sizeof(int));
        mesh->MRABpmlHaloEleIdsP[lev][p] = (int *) calloc(mesh->MRABpmlNhaloEleP[lev][p],sizeof(int)); 
        mesh->MRABpmlHaloIdsP[lev][p]    = (int *) calloc(mesh->MRABpmlNhaloEleP[lev][p],sizeof(int)); 
      }

      int pmlcnt = 0;
      int nonpmlcnt = 0;
      int pmlPcnt[mesh->NMax+1];
      int nonpmlPcnt[mesh->NMax+1];
      for (int p=0;p<=mesh->NMax;p++) {
        pmlPcnt[p] = 0;
        nonpmlPcnt[p] = 0;
      }
      for (int m=0;m<mesh->MRABNelements[lev];m++){
        int e = mesh->MRABelementIds[lev][m];
        int N = mesh->N[e];
        int type = mesh->elementInfo[e];

        if ((type==100)||(type==200)||(type==300)
          ||(type==400)||(type==500)||(type==600)||(type==700)) { //pml element
          mesh->MRABpmlElementIds[lev][pmlcnt] = e;
          mesh->MRABpmlIds[lev][pmlcnt] = pmlIds[e];
          pmlcnt++;
          mesh->MRABpmlElIdsP[lev][N][pmlPcnt[N]] = e;
          mesh->MRABpmlIdsP[lev][N][pmlPcnt[N]] = pmlIds[e];
          pmlPcnt[N]++;
        } else { //nonpml element
          mesh->MRABelementIds[lev][nonpmlcnt] = e;
          nonpmlcnt++;
          mesh->MRABelIdsP[lev][N][nonpmlPcnt[N]] = e;
          nonpmlPcnt[N]++;
        }
      }

      pmlcnt = 0;
      nonpmlcnt = 0;
      for (int p=0;p<=mesh->NMax;p++) {
        pmlPcnt[p] = 0;
        nonpmlPcnt[p] = 0;
      }
      for (int m=0;m<mesh->MRABNhaloElements[lev];m++){
        int e = mesh->MRABhaloIds[lev][m];
        int N = mesh->N[e];
        int type = mesh->elementInfo[e];

        if ((type==100)||(type==200)||(type==300)
          ||(type==400)||(type==500)||(type==600)||(type==700)) { //pml element
          mesh->MRABpmlHaloElementIds[lev][pmlcnt] = e;
          mesh->MRABpmlHaloIds[lev][pmlcnt] = pmlIds[e];
          pmlcnt++;
          mesh->MRABpmlHaloEleIdsP[lev][N][pmlPcnt[N]] = e;
          mesh->MRABpmlHaloIdsP[lev][N][pmlPcnt[N]] = pmlIds[e];
          pmlPcnt[N]++;
        } else { //nonpml element
          mesh->MRABhaloIds[lev][nonpmlcnt] = e;
          nonpmlcnt++;
          mesh->MRABhaloIdsP[lev][N][nonpmlPcnt[N]] = e;
          nonpmlPcnt[N]++;
        }
      }

      //resize nonpml element lists
      mesh->MRABNelements[lev] -= mesh->MRABpmlNelements[lev];
      mesh->MRABNhaloElements[lev] -= mesh->MRABpmlNhaloElements[lev];
      mesh->MRABelementIds[lev] = (int*) realloc(mesh->MRABelementIds[lev],mesh->MRABNelements[lev]*sizeof(int));
      mesh->MRABhaloIds[lev]    = (int*) realloc(mesh->MRABhaloIds[lev],mesh->MRABNhaloElements[lev]*sizeof(int));

      for (int p=0;p<=mesh->NMax;p++) {
        mesh->MRABNelP[lev][p]      -= mesh->MRABpmlNelP[lev][p];
        mesh->MRABNhaloEleP[lev][p] -= mesh->MRABpmlNhaloEleP[lev][p];
        mesh->MRABelIdsP[lev][p]   = (int*) realloc(mesh->MRABelIdsP[lev][p],mesh->MRABNelP[lev][p]*sizeof(int));
        mesh->MRABhaloIdsP[lev][p] = (int*) realloc(mesh->MRABhaloIdsP[lev][p],mesh->MRABNhaloEleP[lev][p]*sizeof(int));
      }
    }

    //set up damping parameter
    mesh->pmlSigmaX = (dfloat *) calloc(mesh->pmlNelements*mesh->cubNpMax,sizeof(dfloat));
    mesh->pmlSigmaY = (dfloat *) calloc(mesh->pmlNelements*mesh->cubNpMax,sizeof(dfloat));
    mesh->pmlSigmaZ = (dfloat *) calloc(mesh->pmlNelements*mesh->cubNpMax,sizeof(dfloat));

    //find the bounding box of the whole domain and interior domain
    dfloat xmin = 1e9, xmax =-1e9;
    dfloat ymin = 1e9, ymax =-1e9;
    dfloat zmin = 1e9, zmax =-1e9;
    dfloat pmlxmin = 1e9, pmlxmax =-1e9;
    dfloat pmlymin = 1e9, pmlymax =-1e9;
    dfloat pmlzmin = 1e9, pmlzmax =-1e9;
    for (int e=0;e<mesh->Nelements;e++) {
      for (int n=0;n<mesh->Nverts;n++) {
        dfloat x = mesh->EX[e*mesh->Nverts+n];
        dfloat y = mesh->EY[e*mesh->Nverts+n];
        dfloat z = mesh->EZ[e*mesh->Nverts+n];

        pmlxmin = (pmlxmin > x) ? x : pmlxmin;
        pmlymin = (pmlymin > y) ? y : pmlymin;
        pmlzmin = (pmlzmin > z) ? z : pmlzmin;
        pmlxmax = (pmlxmax < x) ? x : pmlxmax;
        pmlymax = (pmlymax < y) ? y : pmlymax;
        pmlzmax = (pmlzmax < z) ? z : pmlzmax;
      }

      //skip pml elements
      int type = mesh->elementInfo[e];
      if ((type==100)||(type==200)||(type==300)
            ||(type==400)||(type==500)||(type==600)||(type==700)) continue;

      for (int n=0;n<mesh->Nverts;n++) {
        dfloat x = mesh->EX[e*mesh->Nverts+n];
        dfloat y = mesh->EY[e*mesh->Nverts+n];
        dfloat z = mesh->EZ[e*mesh->Nverts+n];

        xmin = (xmin > x) ? x : xmin;
        ymin = (ymin > y) ? y : ymin;
        zmin = (zmin > z) ? z : zmin;
        xmax = (xmax < x) ? x : xmax;
        ymax = (ymax < y) ? y : ymax;
        zmax = (zmax < z) ? z : zmax;
      }
    }

    dfloat xmaxScale = pow(pmlxmax-xmax,2);
    dfloat xminScale = pow(pmlxmin-xmin,2);
    dfloat ymaxScale = pow(pmlymax-ymax,2);
    dfloat yminScale = pow(pmlymin-ymin,2);
    dfloat zmaxScale = pow(pmlzmax-zmax,2);
    dfloat zminScale = pow(pmlzmin-zmin,2);

    //set up the damping factor
    for (int lev =0;lev<mesh->MRABNlevels;lev++){
      for (int m=0;m<mesh->MRABpmlNelements[lev];m++) {
        int e = mesh->MRABpmlElementIds[lev][m];
        int N = mesh->N[e];
        int pmlId = mesh->MRABpmlIds[lev][m];
        int type = mesh->elementInfo[e];

        int id = e*mesh->Nverts;

        dfloat xe1 = mesh->EX[id+0]; /* x-coordinates of vertices */
        dfloat xe2 = mesh->EX[id+1];
        dfloat xe3 = mesh->EX[id+2];
        dfloat xe4 = mesh->EX[id+3];

        dfloat ye1 = mesh->EY[id+0]; /* y-coordinates of vertices */
        dfloat ye2 = mesh->EY[id+1];
        dfloat ye3 = mesh->EY[id+2];
        dfloat ye4 = mesh->EY[id+3];

        dfloat ze1 = mesh->EZ[id+0]; /* z-coordinates of vertices */
        dfloat ze2 = mesh->EZ[id+1];
        dfloat ze3 = mesh->EZ[id+2];
        dfloat ze4 = mesh->EZ[id+3];

        for(int n=0;n<mesh->cubNp[N];++n){ /* for each node */
          // cubature node coordinates
          dfloat rn = mesh->cubr[N][n];
          dfloat sn = mesh->cubs[N][n];
          dfloat tn = mesh->cubt[N][n];

          /* physical coordinate of interpolation node */
          dfloat x = -0.5*(rn+sn+tn+1.)*xe1 + 0.5*(1+rn)*xe2 + 0.5*(1+sn)*xe3 + 0.5*(1+tn)*xe4 ;
          dfloat y = -0.5*(rn+sn+tn+1.)*ye1 + 0.5*(1+rn)*ye2 + 0.5*(1+sn)*ye3 + 0.5*(1+tn)*ye4 ;
          dfloat z = -0.5*(rn+sn+tn+1.)*ze1 + 0.5*(1+rn)*ze2 + 0.5*(1+sn)*ze3 + 0.5*(1+tn)*ze4 ;

          if (type==100) { //X Pml
            if(x>xmax)
              mesh->pmlSigmaX[mesh->cubNpMax*pmlId + n] = xsigma*pow(x-xmax,2)/xmaxScale;
            if(x<xmin)
              mesh->pmlSigmaX[mesh->cubNpMax*pmlId + n] = xsigma*pow(x-xmin,2)/xminScale;
          } else if (type==200) { //Y Pml
            if(y>ymax)
              mesh->pmlSigmaY[mesh->cubNpMax*pmlId + n] = ysigma*pow(y-ymax,2)/ymaxScale;
            if(y<ymin)
              mesh->pmlSigmaY[mesh->cubNpMax*pmlId + n] = ysigma*pow(y-ymin,2)/yminScale;
          } else if (type==400) { //Z Pml
            if(z>zmax)
              mesh->pmlSigmaZ[mesh->cubNpMax*pmlId + n] = zsigma*pow(z-zmax,2)/zmaxScale;
            if(z<zmin)
              mesh->pmlSigmaZ[mesh->cubNpMax*pmlId + n] = zsigma*pow(z-zmin,2)/zminScale;
          } else if (type==300) { //XY Pml
            if(x>xmax)
              mesh->pmlSigmaX[mesh->cubNpMax*pmlId + n] = xsigma*pow(x-xmax,2)/xmaxScale;
            if(x<xmin)
              mesh->pmlSigmaX[mesh->cubNpMax*pmlId + n] = xsigma*pow(x-xmin,2)/xminScale;
            if(y>ymax)
              mesh->pmlSigmaY[mesh->cubNpMax*pmlId + n] = ysigma*pow(y-ymax,2)/ymaxScale;
            if(y<ymin)
              mesh->pmlSigmaY[mesh->cubNpMax*pmlId + n] = ysigma*pow(y-ymin,2)/yminScale;
          } else if (type==500) { //XZ Pml
            if(x>xmax)
              mesh->pmlSigmaX[mesh->cubNpMax*pmlId + n] = xsigma*pow(x-xmax,2)/xmaxScale;
            if(x<xmin)
              mesh->pmlSigmaX[mesh->cubNpMax*pmlId + n] = xsigma*pow(x-xmin,2)/xminScale;
            if(z>zmax)
              mesh->pmlSigmaZ[mesh->cubNpMax*pmlId + n] = zsigma*pow(z-zmax,2)/zmaxScale;
            if(z<zmin)
              mesh->pmlSigmaZ[mesh->cubNpMax*pmlId + n] = zsigma*pow(z-zmin,2)/zminScale;
          } else if (type==600) { //YZ Pml
            if(y>ymax)
              mesh->pmlSigmaY[mesh->cubNpMax*pmlId + n] = ysigma*pow(y-ymax,2)/ymaxScale;
            if(y<ymin)
              mesh->pmlSigmaY[mesh->cubNpMax*pmlId + n] = ysigma*pow(y-ymin,2)/yminScale;
            if(z>zmax)
              mesh->pmlSigmaZ[mesh->cubNpMax*pmlId + n] = zsigma*pow(z-zmax,2)/zmaxScale;
            if(z<zmin)
              mesh->pmlSigmaZ[mesh->cubNpMax*pmlId + n] = zsigma*pow(z-zmin,2)/zminScale;
          } else if (type==700) { //XYZ Pml
            if(x>xmax)
              mesh->pmlSigmaX[mesh->cubNpMax*pmlId + n] = xsigma*pow(x-xmax,2)/xmaxScale;
            if(x<xmin)
              mesh->pmlSigmaX[mesh->cubNpMax*pmlId + n] = xsigma*pow(x-xmin,2)/xminScale;
            if(y>ymax)
              mesh->pmlSigmaY[mesh->cubNpMax*pmlId + n] = ysigma*pow(y-ymax,2)/ymaxScale;
            if(y<ymin)
              mesh->pmlSigmaY[mesh->cubNpMax*pmlId + n] = ysigma*pow(y-ymin,2)/yminScale;
            if(z>zmax)
              mesh->pmlSigmaZ[mesh->cubNpMax*pmlId + n] = zsigma*pow(z-zmax,2)/zmaxScale;
            if(z<zmin)
              mesh->pmlSigmaZ[mesh->cubNpMax*pmlId + n] = zsigma*pow(z-zmin,2)/zminScale;
          }
        }
      }
    }

    printf("PML: found %d elements inside absorbing layers and %d elements outside\n",
    mesh->pmlNelements, mesh->Nelements-mesh->pmlNelements);

    // assume quiescent pml
    mesh->pmlNfields = 4;
    mesh->pmlq    = (dfloat*) calloc(mesh->pmlNelements*mesh->NpMax*mesh->pmlNfields, sizeof(dfloat));
    mesh->pmlrhsq = (dfloat*) calloc(3*mesh->pmlNelements*mesh->NpMax*mesh->pmlNfields, sizeof(dfloat));

    // set up PML on DEVICE
    mesh->o_pmlq      = mesh->device.malloc(mesh->pmlNelements*mesh->NpMax*mesh->pmlNfields*sizeof(dfloat), mesh->pmlq);
    mesh->o_pmlrhsq   = mesh->device.malloc(3*mesh->pmlNelements*mesh->NpMax*mesh->pmlNfields*sizeof(dfloat), mesh->pmlrhsq);
    mesh->o_pmlSigmaX = mesh->device.malloc(mesh->pmlNelements*mesh->cubNpMax*sizeof(dfloat),mesh->pmlSigmaX);
    mesh->o_pmlSigmaY = mesh->device.malloc(mesh->pmlNelements*mesh->cubNpMax*sizeof(dfloat),mesh->pmlSigmaY);
    mesh->o_pmlSigmaZ = mesh->device.malloc(mesh->pmlNelements*mesh->cubNpMax*sizeof(dfloat),mesh->pmlSigmaZ);

    mesh->o_MRABpmlElementIds     = (occa::memory *) malloc(mesh->MRABNlevels*sizeof(occa::memory));
    mesh->o_MRABpmlIds            = (occa::memory *) malloc(mesh->MRABNlevels*sizeof(occa::memory));
    mesh->o_MRABpmlHaloElementIds = (occa::memory *) malloc(mesh->MRABNlevels*sizeof(occa::memory));
    mesh->o_MRABpmlHaloIds        = (occa::memory *) malloc(mesh->MRABNlevels*sizeof(occa::memory));

    mesh->o_MRABpmlElIdsP      = (occa::memory **) malloc(mesh->MRABNlevels*sizeof(occa::memory));
    mesh->o_MRABpmlIdsP        = (occa::memory **) malloc(mesh->MRABNlevels*sizeof(occa::memory));
    mesh->o_MRABpmlHaloEleIdsP = (occa::memory **) malloc(mesh->MRABNlevels*sizeof(occa::memory));
    mesh->o_MRABpmlHaloIdsP    = (occa::memory **) malloc(mesh->MRABNlevels*sizeof(occa::memory));

    for (int lev=0;lev<mesh->MRABNlevels;lev++) {
      if (mesh->MRABpmlNelements[lev]) {
        mesh->o_MRABpmlElementIds[lev] = mesh->device.malloc(mesh->MRABpmlNelements[lev]*sizeof(int),
           mesh->MRABpmlElementIds[lev]);
        mesh->o_MRABpmlIds[lev] = mesh->device.malloc(mesh->MRABpmlNelements[lev]*sizeof(int),
           mesh->MRABpmlIds[lev]);
      }
      if (mesh->MRABpmlNhaloElements[lev]) {
        mesh->o_MRABpmlHaloElementIds[lev] = mesh->device.malloc(mesh->MRABpmlNhaloElements[lev]*sizeof(int),
           mesh->MRABpmlHaloElementIds[lev]);
        mesh->o_MRABpmlHaloIds[lev] = mesh->device.malloc(mesh->MRABpmlNhaloElements[lev]*sizeof(int),
           mesh->MRABpmlHaloIds[lev]);
      }

      mesh->o_MRABpmlElIdsP[lev]      = (occa::memory *) malloc((mesh->NMax+1)*sizeof(occa::memory));
      mesh->o_MRABpmlIdsP[lev]        = (occa::memory *) malloc((mesh->NMax+1)*sizeof(occa::memory));
      mesh->o_MRABpmlHaloEleIdsP[lev] = (occa::memory *) malloc((mesh->NMax+1)*sizeof(occa::memory));
      mesh->o_MRABpmlHaloIdsP[lev]    = (occa::memory *) malloc((mesh->NMax+1)*sizeof(occa::memory));        
      for (int p=1;p<=mesh->NMax;p++) {
        if (mesh->MRABpmlNelP[lev][p]) {
          mesh->o_MRABpmlElIdsP[lev][p] = mesh->device.malloc(mesh->MRABpmlNelP[lev][p]*sizeof(int),
             mesh->MRABpmlElIdsP[lev][p]);
          mesh->o_MRABpmlIdsP[lev][p] = mesh->device.malloc(mesh->MRABpmlNelP[lev][p]*sizeof(int),
             mesh->MRABpmlIdsP[lev][p]);
        }
        if (mesh->MRABpmlNhaloEleP[lev][p]) {
          mesh->o_MRABpmlHaloEleIdsP[lev][p] = mesh->device.malloc(mesh->MRABpmlNhaloEleP[lev][p]*sizeof(int),
             mesh->MRABpmlHaloEleIdsP[lev][p]);
          mesh->o_MRABpmlHaloIdsP[lev][p] = mesh->device.malloc(mesh->MRABpmlNhaloEleP[lev][p]*sizeof(int),
             mesh->MRABpmlHaloIdsP[lev][p]);
        }
      }
    }

    free(pmlIds);
  }
}