#ifndef __DEBUG_H__
#define __DEBUG_H__

#include "view_nd.h"
#include "config.h"
#include <stdio.h>

/*
void printL2NormOfF(config *conf, view_4d fn);
void printL2Norm(config *conf, view_2d phi);
void printAll(char* name, config *conf, view_2d phi);
void printAllf(char* name, config *conf, view_4d f);
*/

static void printL2NormOfF(Config *conf, view_4d fn)
{
  int s_nxmax, s_nvxmax, s_nymax, s_nvymax;
  int ix, iy, ivx, ivy;
  Domain* dom = &(conf->dom_);

  s_nxmax  = dom->nxmax_[0];
  s_nymax  = dom->nxmax_[1];
  s_nvxmax = dom->nxmax_[2];
  s_nvymax = dom->nxmax_[3];

  #pragma omp master
  {
    double sum = 0.;
    for(ivy=0; ivy<s_nvymax; ivy++)
    {
      for(ivx=0; ivx<s_nvxmax; ivx++)
      {
        for(iy=0; iy<s_nymax; iy++)
        {
          for(ix=0; ix<s_nxmax; ix++)
          {
            sum += fn[ivy][ivx][iy][ix] * fn[ivy][ivx][iy][ix];
          }
        }
      }
    }
    printf("|f4d*f4d| = %20.13le\n", sum);
  }
}

static void printL2Norm(Config *conf, view_2d phi)
{
  int s_nxmax, s_nymax;
  int ix, iy;
  Domain* dom = &(conf->dom_);

  s_nxmax  = dom->nxmax_[0];
  s_nymax  = dom->nxmax_[1];

  #pragma omp master
  {
    double sum = 0.;
    for(iy=0; iy<s_nymax; iy++)
    {
      for(ix=0; ix<s_nxmax; ix++)
      {
        sum += phi[iy][ix] * phi[iy][ix];
      }
    }
    printf("%20.13le\n", sum);
  }
}

static void printAll(char* name, Config *conf, view_2d phi)
{
  int s_nxmax, s_nymax;
  int ix, iy;
  char filename[32];
  FILE* fileid;
  Domain* dom = &(conf->dom_);

  s_nxmax  = dom->nxmax_[0];
  s_nymax  = dom->nxmax_[1];

  sprintf(filename, "%s.out", name);
  fileid = fopen(filename, "w");
  for(iy=0; iy<s_nymax; iy++)
  {
    for(ix=0; ix<s_nxmax; ix++)
    {
      fprintf(fileid, "%20.13le ", phi[iy][ix]);
      //fprintf(fileid, "%4d %4d %20.13le\n", ix, iy, phi->pt2[iy][ix]);
    }
    fprintf(fileid, "\n");
  }
  fclose(fileid);
}

static void printAllf(char* name, Config *conf, view_4d f)
{
  int s_nxmax, s_nvxmax, s_nymax, s_nvymax;
  int ix, iy, ivx, ivy;
  char filename[32];
  FILE* fileid;
  Domain* dom = &(conf->dom_);

  s_nxmax  = dom->nxmax_[0];
  s_nymax  = dom->nxmax_[1];
  s_nvxmax = dom->nxmax_[2];
  s_nvymax = dom->nxmax_[3];

  sprintf(filename, "%s.out", name);
  fileid = fopen(filename, "w");
  for(ivy=0; ivy<s_nvymax; ivy++)
  {
    for(ivx=0; ivx<s_nvxmax; ivx++)
    {
      for(iy=0; iy<s_nymax; iy++)
      {
        for(ix=0; ix<s_nxmax; ix++)
        {
          fprintf(fileid, "%20.13le ", f[ivy][ivx][iy][ix]);
        }
      }
    }
  }
}

#endif
