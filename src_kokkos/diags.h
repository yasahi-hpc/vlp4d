#ifndef __DIAGS_H__
#define __DIAGS_H__

#include "types.h"
#include "config.h"
#include "efield.h"

struct Diags
{
private:
  view_1d_host nrj_;
  view_1d_host mass_;

public:
  Diags(Config *conf);
  virtual ~Diags();

  void compute(Config *conf, Efield *ef, int iter);
  void save(Config *conf);
};

#endif
