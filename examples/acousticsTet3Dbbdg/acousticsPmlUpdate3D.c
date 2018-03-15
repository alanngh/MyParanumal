#include "acoustics3D.h"


void acousticsMRABpmlUpdate3D(mesh3D *mesh,
                           dfloat a1,
                           dfloat a2,
                           dfloat a3, iint lev, dfloat dt){

  for(iint et=0;et<mesh->MRABpmlNelements[lev];et++){
    iint e = mesh->MRABpmlElementIds[lev][et];
    iint pmlId = mesh->MRABpmlIds[lev][et];

    for(iint n=0;n<mesh->Np;++n){
      iint id = mesh->Nfields*(e*mesh->Np + n);
      iint pid = mesh->pmlNfields*(pmlId*mesh->Np + n);

      iint rhsId1 = 3*id + ((mesh->MRABshiftIndex[lev]+0)%3)*mesh->Nfields;
      iint rhsId2 = 3*id + ((mesh->MRABshiftIndex[lev]+1)%3)*mesh->Nfields;
      iint rhsId3 = 3*id + ((mesh->MRABshiftIndex[lev]+2)%3)*mesh->Nfields;
      iint pmlrhsId1 = 3*pid + ((mesh->MRABshiftIndex[lev]+0)%3)*mesh->pmlNfields;
      iint pmlrhsId2 = 3*pid + ((mesh->MRABshiftIndex[lev]+1)%3)*mesh->pmlNfields;
      iint pmlrhsId3 = 3*pid + ((mesh->MRABshiftIndex[lev]+2)%3)*mesh->pmlNfields;

      mesh->pmlq[pid+0] += dt*(a1*mesh->pmlrhsq[pmlrhsId1+0] + a2*mesh->pmlrhsq[pmlrhsId2+0] + a3*mesh->pmlrhsq[pmlrhsId3+0]);
      mesh->pmlq[pid+1] += dt*(a1*mesh->pmlrhsq[pmlrhsId1+1] + a2*mesh->pmlrhsq[pmlrhsId2+1] + a3*mesh->pmlrhsq[pmlrhsId3+1]);
      mesh->pmlq[pid+2] += dt*(a1*mesh->pmlrhsq[pmlrhsId1+2] + a2*mesh->pmlrhsq[pmlrhsId2+2] + a3*mesh->pmlrhsq[pmlrhsId3+2]);
      mesh->q[id+0] += dt*(a1*mesh->rhsq[rhsId1+0] + a2*mesh->rhsq[rhsId2+0] + a3*mesh->rhsq[rhsId3+0]);
      mesh->q[id+1] += dt*(a1*mesh->rhsq[rhsId1+1] + a2*mesh->rhsq[rhsId2+1] + a3*mesh->rhsq[rhsId3+1]);
      mesh->q[id+2] += dt*(a1*mesh->rhsq[rhsId1+2] + a2*mesh->rhsq[rhsId2+2] + a3*mesh->rhsq[rhsId3+2]);
      mesh->q[id+3] = mesh->pmlq[pid+0]+mesh->pmlq[pid+1]+mesh->pmlq[pid+2];
    }

    //write new traces to fQ
    for (iint f =0;f<mesh->Nfaces;f++) {
      for (iint n=0;n<mesh->Nfp;n++) {
        iint id  = e*mesh->Nfp*mesh->Nfaces + f*mesh->Nfp + n;
        iint qidM = mesh->Nfields*mesh->vmapM[id];

        iint qid = mesh->Nfields*id;
        // save trace node values of q
        for (iint fld=0; fld<mesh->Nfields;fld++) {
          mesh->fQP[qid+fld] = mesh->q[qidM+fld];
          mesh->fQM[qid+fld] = mesh->q[qidM+fld];
        }
      }
    }
  }
}

void acousticsMRABpmlUpdateTrace3D(mesh3D *mesh,
                           dfloat a1,
                           dfloat a2,
                           dfloat a3, iint lev, dfloat dt){

  for(iint et=0;et<mesh->MRABpmlNhaloElements[lev];et++){
    iint e = mesh->MRABpmlHaloElementIds[lev][et];
    iint pmlId = mesh->MRABpmlHaloIds[lev][et];

    dfloat s_q[mesh->Np*mesh->Nfields];

    for(iint n=0;n<mesh->Np;++n){
      iint id = mesh->Nfields*(e*mesh->Np + n);
      iint pid = mesh->pmlNfields*(pmlId*mesh->Np + n);

      iint rhsId1 = 3*id + ((mesh->MRABshiftIndex[lev]+0)%3)*mesh->Nfields;
      iint rhsId2 = 3*id + ((mesh->MRABshiftIndex[lev]+1)%3)*mesh->Nfields;
      iint rhsId3 = 3*id + ((mesh->MRABshiftIndex[lev]+2)%3)*mesh->Nfields;
      iint pmlrhsId1 = 3*pid + ((mesh->MRABshiftIndex[lev]+0)%3)*mesh->pmlNfields;
      iint pmlrhsId2 = 3*pid + ((mesh->MRABshiftIndex[lev]+1)%3)*mesh->pmlNfields;
      iint pmlrhsId3 = 3*pid + ((mesh->MRABshiftIndex[lev]+2)%3)*mesh->pmlNfields;

      // Increment solutions
      dfloat px = mesh->pmlq[pid+0] + dt*(a1*mesh->pmlrhsq[pmlrhsId1+0] + a2*mesh->pmlrhsq[pmlrhsId2+0] + a3*mesh->pmlrhsq[pmlrhsId3+0]);
      dfloat py = mesh->pmlq[pid+1] + dt*(a1*mesh->pmlrhsq[pmlrhsId1+1] + a2*mesh->pmlrhsq[pmlrhsId2+1] + a3*mesh->pmlrhsq[pmlrhsId3+1]);
      dfloat pz = mesh->pmlq[pid+2] + dt*(a1*mesh->pmlrhsq[pmlrhsId1+2] + a2*mesh->pmlrhsq[pmlrhsId2+2] + a3*mesh->pmlrhsq[pmlrhsId3+2]);
      s_q[n*mesh->Nfields+0] = mesh->q[id+0] + dt*(a1*mesh->rhsq[rhsId1+0] + a2*mesh->rhsq[rhsId2+0] + a3*mesh->rhsq[rhsId3+0]);
      s_q[n*mesh->Nfields+1] = mesh->q[id+1] + dt*(a1*mesh->rhsq[rhsId1+1] + a2*mesh->rhsq[rhsId2+1] + a3*mesh->rhsq[rhsId3+1]);
      s_q[n*mesh->Nfields+2] = mesh->q[id+2] + dt*(a1*mesh->rhsq[rhsId1+2] + a2*mesh->rhsq[rhsId2+2] + a3*mesh->rhsq[rhsId3+2]);
      s_q[n*mesh->Nfields+3] = px+py+pz;
    }

    //write new traces to fQ
    for (iint f =0;f<mesh->Nfaces;f++) {
      for (iint n=0;n<mesh->Nfp;n++) {
        iint id  = e*mesh->Nfp*mesh->Nfaces + f*mesh->Nfp + n;
        iint qidM = mesh->Nfields*(mesh->vmapM[id]-e*mesh->Np);

        iint qid = n*mesh->Nfields + f*mesh->Nfp*mesh->Nfields + e*mesh->Nfaces*mesh->Nfp*mesh->Nfields;
        // save trace node values of q
        for (iint fld=0; fld<mesh->Nfields;fld++) {
          mesh->fQM[qid+fld] = s_q[qidM+fld];
          mesh->fQP[qid+fld] = s_q[qidM+fld];
        }
      }
    }
  }
}


void acousticsMRABpmlUpdate3D_wadg(mesh3D *mesh,
                           dfloat a1,
                           dfloat a2,
                           dfloat a3, iint lev, dfloat dt){

  for(iint et=0;et<mesh->MRABpmlNelements[lev];et++){
    iint e = mesh->MRABpmlElementIds[lev][et];
    iint pmlId = mesh->MRABpmlIds[lev][et];

    dfloat p[mesh->Np];
    dfloat cubp[mesh->cubNp];

    for(iint n=0;n<mesh->Np;++n){
      iint id = mesh->Nfields*(e*mesh->Np + n);
      iint pid = mesh->pmlNfields*(pmlId*mesh->Np + n);

      iint rhsId1 = 3*id + ((mesh->MRABshiftIndex[lev]+0)%3)*mesh->Nfields;
      iint rhsId2 = 3*id + ((mesh->MRABshiftIndex[lev]+1)%3)*mesh->Nfields;
      iint rhsId3 = 3*id + ((mesh->MRABshiftIndex[lev]+2)%3)*mesh->Nfields;
      iint pmlrhsId1 = 3*pid + ((mesh->MRABshiftIndex[lev]+0)%3)*mesh->pmlNfields;
      iint pmlrhsId2 = 3*pid + ((mesh->MRABshiftIndex[lev]+1)%3)*mesh->pmlNfields;
      iint pmlrhsId3 = 3*pid + ((mesh->MRABshiftIndex[lev]+2)%3)*mesh->pmlNfields;

      mesh->pmlq[pid+0] += dt*(a1*mesh->pmlrhsq[pmlrhsId1+0] + a2*mesh->pmlrhsq[pmlrhsId2+0] + a3*mesh->pmlrhsq[pmlrhsId3+0]);
      mesh->pmlq[pid+1] += dt*(a1*mesh->pmlrhsq[pmlrhsId1+1] + a2*mesh->pmlrhsq[pmlrhsId2+1] + a3*mesh->pmlrhsq[pmlrhsId3+1]);
      mesh->pmlq[pid+2] += dt*(a1*mesh->pmlrhsq[pmlrhsId1+2] + a2*mesh->pmlrhsq[pmlrhsId2+2] + a3*mesh->pmlrhsq[pmlrhsId3+2]);
      mesh->q[id+0] += dt*(a1*mesh->rhsq[rhsId1+0] + a2*mesh->rhsq[rhsId2+0] + a3*mesh->rhsq[rhsId3+0]);
      mesh->q[id+1] += dt*(a1*mesh->rhsq[rhsId1+1] + a2*mesh->rhsq[rhsId2+1] + a3*mesh->rhsq[rhsId3+1]);
      mesh->q[id+2] += dt*(a1*mesh->rhsq[rhsId1+2] + a2*mesh->rhsq[rhsId2+2] + a3*mesh->rhsq[rhsId3+2]);
      p[n] = mesh->pmlq[pid+0]+mesh->pmlq[pid+1]+mesh->pmlq[pid+2];
    }

    // Interpolate rhs to cubature nodes
    for(iint n=0;n<mesh->cubNp;++n){
      cubp[n] = 0.f;
      for (iint i=0;i<mesh->Np;++i){
        cubp[n] += mesh->cubInterp[n*mesh->Np + i] * p[i];
      }
      // Multiply result by wavespeed c2 at cubature node
      cubp[n] *= mesh->c2[n + e*mesh->cubNp];
    }

    // Increment solution, project result back down
    for(iint n=0;n<mesh->Np;++n){
      // Project scaled rhs down
      dfloat c2p = 0.f;
      for (iint i=0;i<mesh->cubNp;++i){
        c2p += mesh->cubProject[n*mesh->cubNp + i] * cubp[i];
      }
      iint id = mesh->Nfields*(e*mesh->Np + n);
      mesh->q[id+3] = c2p;
    }

    //write new traces to fQ
    for (iint f =0;f<mesh->Nfaces;f++) {
      for (iint n=0;n<mesh->Nfp;n++) {
        iint id  = e*mesh->Nfp*mesh->Nfaces + f*mesh->Nfp + n;
        iint qidM = mesh->Nfields*mesh->vmapM[id];

        iint qid = mesh->Nfields*id;
        // save trace node values of q
        for (iint fld=0; fld<mesh->Nfields;fld++) {
          mesh->fQP[qid+fld] = mesh->q[qidM+fld];
          mesh->fQM[qid+fld] = mesh->q[qidM+fld];
        }
      }
    }
  }
}

void acousticsMRABpmlUpdateTrace3D_wadg(mesh3D *mesh,
                           dfloat a1,
                           dfloat a2,
                           dfloat a3, iint lev, dfloat dt){

  for(iint et=0;et<mesh->MRABpmlNhaloElements[lev];et++){
    iint e = mesh->MRABpmlHaloElementIds[lev][et];
    iint pmlId = mesh->MRABpmlHaloIds[lev][et];

    dfloat cubp[mesh->cubNp];
    dfloat s_q[mesh->Np*mesh->Nfields];

    for(iint n=0;n<mesh->Np;++n){
      iint id = mesh->Nfields*(e*mesh->Np + n);
      iint pid = mesh->pmlNfields*(pmlId*mesh->Np + n);

      iint rhsId1 = 3*id + ((mesh->MRABshiftIndex[lev]+0)%3)*mesh->Nfields;
      iint rhsId2 = 3*id + ((mesh->MRABshiftIndex[lev]+1)%3)*mesh->Nfields;
      iint rhsId3 = 3*id + ((mesh->MRABshiftIndex[lev]+2)%3)*mesh->Nfields;
      iint pmlrhsId1 = 3*pid + ((mesh->MRABshiftIndex[lev]+0)%3)*mesh->pmlNfields;
      iint pmlrhsId2 = 3*pid + ((mesh->MRABshiftIndex[lev]+1)%3)*mesh->pmlNfields;
      iint pmlrhsId3 = 3*pid + ((mesh->MRABshiftIndex[lev]+2)%3)*mesh->pmlNfields;

      // Increment solutions
      dfloat px = mesh->pmlq[pid+0] + dt*(a1*mesh->pmlrhsq[pmlrhsId1+0] + a2*mesh->pmlrhsq[pmlrhsId2+0] + a3*mesh->pmlrhsq[pmlrhsId3+0]);
      dfloat py = mesh->pmlq[pid+1] + dt*(a1*mesh->pmlrhsq[pmlrhsId1+1] + a2*mesh->pmlrhsq[pmlrhsId2+1] + a3*mesh->pmlrhsq[pmlrhsId3+1]);
      dfloat pz = mesh->pmlq[pid+2] + dt*(a1*mesh->pmlrhsq[pmlrhsId1+2] + a2*mesh->pmlrhsq[pmlrhsId2+2] + a3*mesh->pmlrhsq[pmlrhsId3+2]);
      s_q[n*mesh->Nfields+0] = mesh->q[id+0] + dt*(a1*mesh->rhsq[rhsId1+0] + a2*mesh->rhsq[rhsId2+0] + a3*mesh->rhsq[rhsId3+0]);
      s_q[n*mesh->Nfields+1] = mesh->q[id+1] + dt*(a1*mesh->rhsq[rhsId1+1] + a2*mesh->rhsq[rhsId2+1] + a3*mesh->rhsq[rhsId3+1]);
      s_q[n*mesh->Nfields+2] = mesh->q[id+2] + dt*(a1*mesh->rhsq[rhsId1+2] + a2*mesh->rhsq[rhsId2+2] + a3*mesh->rhsq[rhsId3+2]);
      s_q[n*mesh->Nfields+2] = px+py+pz;
    }

    // Interpolate rhs to cubature nodes
    for(iint n=0;n<mesh->cubNp;++n){
      cubp[n] = 0.f;
      for (iint i=0;i<mesh->Np;++i){
        cubp[n] += mesh->cubInterp[n*mesh->Np + i] * s_q[i*mesh->Nfields+2];
      }
      // Multiply result by wavespeed c2 at cubature node
      cubp[n] *= mesh->c2[n + e*mesh->cubNp];
    }

    // Increment solution, project result back down
    for(iint n=0;n<mesh->Np;++n){
      // Project scaled rhs down
      s_q[n*mesh->Nfields+3] = 0.f;
      for (iint i=0;i<mesh->cubNp;++i){
        s_q[n*mesh->Nfields+3] += mesh->cubProject[n*mesh->cubNp + i] * cubp[i];
      }
    }

    //write new traces to fQ
    for (iint f =0;f<mesh->Nfaces;f++) {
      for (iint n=0;n<mesh->Nfp;n++) {
        iint id  = e*mesh->Nfp*mesh->Nfaces + f*mesh->Nfp + n;
        iint qidM = mesh->Nfields*(mesh->vmapM[id]-e*mesh->Np);

        iint qid = n*mesh->Nfields + f*mesh->Nfp*mesh->Nfields + e*mesh->Nfaces*mesh->Nfp*mesh->Nfields;
        // save trace node values of q
        for (iint fld=0; fld<mesh->Nfields;fld++) {
          mesh->fQP[qid+fld] = s_q[qidM+fld];
          mesh->fQM[qid+fld] = s_q[qidM+fld];
        }
      }
    }
  }
}



