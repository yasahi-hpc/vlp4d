#include "init.h"
#if ! defined( ENABLE_OPENACC )
  #include <omp.h>
#else
  #include <openacc.h>
#endif

// Prototypes
void import(const char *f, Config *conf);
void print(Config *conf);
void initcase(Config *conf, view_4d fn);
void testcaseCYL02(Config* conf, view_4d fn);
void testcaseCYL05(Config* conf, view_4d fn);
void testcaseSLD10(Config* conf, view_4d fn);
void testcaseTSI20(Config* conf, view_4d fn);

void import(const char *f, Config *conf) {
  char idcase[8], tmp;
  Physics* phys = &(conf->phys_);
  Domain* dom = &(conf->dom_);
  FILE* stream = fopen(f, "r");

  if(stream == (FILE*)NULL) {
    printf("import: error file not found\n");
    abort();
  }

  /*Physical parameters */
  phys->eps0_ = 1.; /* permittivity of free space */
  phys->echarge_ = 1.; /* charge of particle */
  phys->mass_ = 1; /* mass of particle */
  phys->omega02_ = 0; /* coefficient of applied field */
  phys->vbeam_ = 0.; /* beam velocity */
  phys->S_ = 1; /* lattice period */
  phys->psi_ = 0;

  for(size_t i = 0; i < DIMENSION; i++) {
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %lu\n", &(dom->nxmax_[i]));
  }

  for(size_t i = 0; i < DIMENSION; i++) {
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %le\n", &(dom->minPhy_[i]));
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %le\n", &(dom->maxPhy_[i]));
  }

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fgets(idcase, 7, stream);

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %le\n", &(dom->dt_));

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom->nbiter_));
  
  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom->ifreq_));
   
  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom->fxvx_));

  dom->idcase_ = atoi(idcase);
  
  for(size_t i = 0; i < DIMENSION; i++) {
    dom->dx_[i] = (dom->maxPhy_[i] - dom->minPhy_[i]) / dom->nxmax_[i];
  }
  
  fclose(stream);
};

void print(Config *conf) {
  Domain* dom = &(conf->dom_);

  printf("** Definition of mesh\n");
  printf("Number of points in  x with the coarse mesh : %d\n", dom->nxmax_[0]);
  printf("Number of points in  y with the coarse mesh : %d\n", dom->nxmax_[1]);
  printf("Number of points in Vx with the coarse mesh : %d\n", dom->nxmax_[2]);
  printf("Number of points in Vy with the coarse mesh : %d\n", dom->nxmax_[3]);
  
  printf("\n** Defintion of the geometry of the domain\n");
  printf("Minimal value of Ex : %lf\n", dom->minPhy_[0]);
  printf("Maximal value of Ex : %lf\n", dom->maxPhy_[0]);
  printf("Minimal value of Ey : %lf\n", dom->minPhy_[1]);
  printf("Maximal value of Ey : %lf\n", dom->maxPhy_[1]);
  
  printf("\nMinimal value of Vx : %lf\n", dom->minPhy_[2]);
  printf("Maximal value of Vx   : %lf\n", dom->maxPhy_[2]);
  printf("Minimal value of Vy   : %lf\n", dom->minPhy_[3]);
  printf("Maximal value of Vy   : %lf\n", dom->maxPhy_[3]);
  
  printf("\n** Considered test cases");
  printf("\n-10- Landau Damping");
  printf("\n-11- Landau Damping 2");
  printf("\n-20- Two beam instability");
  printf("\nNumber of the chosen test case : %d\n", dom->idcase_);
  
  printf("\n** Iterations in time and diagnostics\n");
  printf("Time step : %lf\n", dom->dt_);
  printf("Number of total iterations : %d\n", dom->nbiter_);
  printf("Frequency of diagnostics : %d\n", dom->ifreq_);
  
  printf("Diagnostics of fxvx : %d\n", dom->fxvx_);
}

void initcase(Config* conf, view_4d fn) {
  Domain* dom = &(conf->dom_);
  
  switch(dom->idcase_)
  {
    case 2:
      testcaseCYL02(conf, fn);
      break; // CYL02 ;
    case 5:
      testcaseCYL05(conf, fn);
      break; // CYL05 ;
    case 10:
      testcaseSLD10(conf, fn);
      break; // SLD10 ;
    case 20:
      testcaseTSI20(conf, fn);
      break; // TSI20 ;
    default:
      printf("Unknown test case !\n");
      abort();
      break;
  }
}

void testcaseCYL02(Config* conf, view_4d fn) {
  Domain* dom = &(conf->dom_);
  const double PI = M_PI;
  const double AMPLI = 4;
  const double PERIOD = 0.5 * PI;
  const double cc = 0.50 * (6. / 16.);
  const double rc = 0.50 * (4. / 16.);

  int s_nxmax = dom->nxmax_[0];
  int s_nymax = dom->nxmax_[1];
  int s_nvxmax = dom->nxmax_[2];
  int s_nvymax = dom->nxmax_[3];

  #if ! defined( ENABLE_OPENACC )
  #pragma omp for schedule(static) collapse(2)
  #endif
  for(int ivy = 0; ivy < s_nvymax; ivy++) {
    for(int ivx = 0; ivx < s_nvxmax; ivx++) {
      double vx = dom->minPhy_[2] + ivx * dom->dx_[2];
      for(int iy = 0; iy < s_nymax; iy++) {
        for(int ix = 0; ix < s_nxmax; ix++) {
          double x = dom->minPhy_[0] + ix * dom->dx_[0];
          double xx = x;
          double vv = vx;
                                                           
          double hv = 0.0;
          double hx = 0.0;
                                                           
          if((vv <= cc + rc) && (vv >= cc - rc)) {
            hv = cos(PERIOD * ((vv - cc) / rc));
          }
          else if((vv <= -cc + rc) && (vv >= -cc - rc)) {
            hv = -cos(PERIOD * ((vv + cc) / rc));
          }

          if((xx <= cc + rc) && (xx >= cc - rc)) {
            hx = cos(PERIOD * ((xx - cc) / rc));
          }
          else if((xx <= -cc + rc) && (xx >= -cc - rc)) {
            hx = -cos(PERIOD * ((xx + cc) / rc));
          }
                                                           
          fn[ivy][ivx][iy][ix] = (AMPLI * hx * hv);
        }
      }
    }
  }
}

void testcaseCYL05(Config* conf, view_4d fn) {
  Domain * dom = &(conf->dom_);
  const double PI = M_PI;
  const double AMPLI = 4;
  const double PERIOD = 0.5 * PI;
  const double cc = 0.50 * (6. / 16.);
  const double rc = 0.50 * (4. / 16.);

  int s_nxmax = dom->nxmax_[0];
  int s_nymax = dom->nxmax_[1];
  int s_nvxmax = dom->nxmax_[2];
  int s_nvymax = dom->nxmax_[3];

  #if ! defined( ENABLE_OPENACC )
  #pragma omp for schedule(static) collapse(2)
  #endif
  for(int ivy = 0; ivy < s_nvymax; ivy++) {
    for(int ivx = 0; ivx < s_nvxmax; ivx++) {
      double vy = dom->minPhy_[3] + ivy * dom->dx_[3];
      for(int iy = 0; iy < s_nymax; iy++) {
        double y = dom->minPhy_[1] + iy * dom->dx_[1];
        for(int ix = 0; ix < s_nxmax; ix++) {
          double xx = y;
          double vv = vy;
                                                           
          double hv = 0.0;
          double hx = 0.0;
                                                           
          if((vv <= cc + rc) && (vv >= cc - rc)) {
            hv = cos(PERIOD * ((vv - cc) / rc));
          }
          else if((vv <= -cc + rc) && (vv >= -cc - rc)) {
            hv = -cos(PERIOD * ((vv + cc) / rc));
          }

          if((xx <= cc + rc) && (xx >= cc - rc)) {
            hx = cos(PERIOD * ((xx - cc) / rc));
          }
          else if((xx <= -cc + rc) && (xx >= -cc - rc)) {
            hx = -cos(PERIOD * ((xx + cc) / rc));
          }
                                                           
          fn[ivy][ivx][iy][ix] = (AMPLI * hx * hv);
        }
      }
    }
  }
}

void testcaseSLD10(Config* conf, view_4d fn) {
  Domain * dom = &(conf->dom_);
  int s_nxmax = dom->nxmax_[0];
  int s_nymax = dom->nxmax_[1];
  int s_nvxmax = dom->nxmax_[2];
  int s_nvymax = dom->nxmax_[3];

  #if ! defined( ENABLE_OPENACC )
  #pragma omp for schedule(static) collapse(2)
  #endif
  for(int ivy = 0; ivy < s_nvymax; ivy++) {
    for(int ivx = 0; ivx < s_nvxmax; ivx++) {
      double vy = dom->minPhy_[3] + ivy * dom->dx_[3];
      double vx = dom->minPhy_[2] + ivx * dom->dx_[2];
      for(int iy = 0; iy < s_nymax; iy++) {
        double y = dom->minPhy_[1] + iy * dom->dx_[1];
        for(int ix = 0; ix < s_nxmax; ix++) {
          double x = dom->minPhy_[0] + ix * dom->dx_[0];

          double sum = (vx * vx + vy * vy);
          fn[ivy][ivx][iy][ix] = (1. / (2 * M_PI)) * exp(-0.5 * (sum)) * (1 + 0.05 * (cos(0.5 * x) * cos(0.5 * y)));
        }
      }
    }
  }
}

void testcaseTSI20(Config* conf, view_4d fn) {
  Domain * dom = &(conf->dom_);
  int s_nxmax = dom->nxmax_[0];
  int s_nymax = dom->nxmax_[1];
  int s_nvxmax = dom->nxmax_[2];
  int s_nvymax = dom->nxmax_[3];

  double xi = 0.90;
  double alpha = 0.05;
  /*  eps=0.15;  */
  /* x et y sont definis sur [0,2*pi] */

  #if ! defined( ENABLE_OPENACC )
  #pragma omp for schedule(static) collapse(2)
  #endif
  for(int ivy = 0; ivy < s_nvymax; ivy++) {
    for(int ivx = 0; ivx < s_nvxmax; ivx++) {
      double vy = dom->minPhy_[3] + ivy * dom->dx_[3];
      double vx = dom->minPhy_[2] + ivx * dom->dx_[2];
      for(int iy = 0; iy < s_nymax; iy++) {
        //double y = dom->minPhy_[1] + iy * dom->dx_[1];
        for(int ix = 0; ix < s_nxmax; ix++) {
          double x = dom->minPhy_[0] + ix * dom->dx_[0];

          double sum = (vx * vx + vy * vy);
          fn[ivy][ivx][iy][ix] = (1 + alpha * (cos(0.5 * x))) * 1 / (2 * M_PI) * ((2 - 2 * xi) / (3 - 2 * xi)) * (1 + .5 * vx * vx / (1 - xi)) * exp(-.5 * sum);
        }
      }
    }
  }
}

void init(const char *file, Config *conf, view_4d &fn, view_4d &fnp1, Efield **ef, Diags **dg) {
  const Domain* dom = &conf->dom_; 

  #if ! defined( ENABLE_OPENACC )
    #pragma omp master
  #endif
    {
      import(file, conf);
      print(conf);

      // Allocate 4D data structures
      allocate(fn, dom->nxmax_);
      allocate(fnp1, dom->nxmax_);

      *ef = new Efield(conf, {dom->nxmax_[0], dom->nxmax_[1]});

      // allocate and initialize diagnostics data structures
      *dg = new Diags(conf);
    }
  #if ! defined( ENABLE_OPENACC )
    #pragma omp barrier
  #endif

  initcase(conf, fn);
  
  #if defined( ENABLE_OPENACC )
    // After initialization on host side, update device
    int device = 0;
    acc_set_device_num(device, acc_device_nvidia);
    float64 *dptr_fn   = fn.raw();
    float64 *dptr_fnp1 = fnp1.raw();
    int n = dom->nxmax_[0] * dom->nxmax_[1] * dom->nxmax_[2] * dom->nxmax_[3];
    #pragma acc enter data create(dptr_fn[0:n],dptr_fnp1[0:n])
    #pragma acc update device(dptr_fn[0:n])
  #endif
}

void finalize(Config *conf, view_4d &fn, view_4d &fnp1, Efield **ef, Diags **dg) {
  #if defined( ENABLE_OPENACC )
    float64 *dptr_fn   = fn.raw();
    float64 *dptr_fnp1 = fnp1.raw();
    const Domain* dom = &conf->dom_; 
    int n = dom->nxmax_[0] * dom->nxmax_[1] * dom->nxmax_[2] * dom->nxmax_[3];
    #pragma acc exit data delete(dptr_fn[0:n],dptr_fnp1[0:n])
  #else
    #pragma omp master
  #endif
    {
      // Store diagnostics
      (*dg)->save(conf);

      // deallocate 4D data structures
      deallocate(fn);
      deallocate(fnp1);
      delete *ef;
      delete *dg;
    }
  #if ! defined( ENABLE_OPENACC )
    #pragma omp barrier
  #endif
}
