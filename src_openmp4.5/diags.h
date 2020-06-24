#ifndef __DIAGS_H__
#define __DIAGS_H__

#include "config.h"
#include "efield.h"
#include "view_nd.h"

struct Diags {
private:
  view_1d nrj_;
  view_1d mass_;

public:
  Diags(Config *conf);
  virtual ~Diags();
      
  void compute(Config* conf, Efield* ef, int iter);
  void save(Config *conf);
};

#endif
