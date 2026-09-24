// SHOC S3D reference: the 27 unmodified kernel files compiled as plain C (the OpenCL qualifiers
// defined away, get_global_id(0) = the host loop variable s3d_gid), so the scalar reference is the
// same source as the Hwacha kernels; SHOC itself does not verify the S3D results.
#include <math.h>
#include <float.h>
#define SINGLE_PRECISION
#define N_GP 64   /* must match CLDEFS_s3d in the Makefile and N in s3d_main.c */
int s3d_gid;
#define __kernel
#define __global
#define get_global_id(d) s3d_gid
#define exp10(x) powf(10.0f, (x))
#define exp(x) expf(x)
#define log(x) logf(x)
#define fmin(a, b) fminf(a, b)
#define fmax(a, b) fmaxf(a, b)
#include "gr_base.cl"
#include "ratt.cl"
#include "ratt2.cl"
#include "ratt3.cl"
#include "ratt4.cl"
#include "ratt5.cl"
#include "ratt6.cl"
#include "ratt7.cl"
#include "ratt8.cl"
#include "ratt9.cl"
#include "ratt10.cl"
#include "ratx.cl"
#include "ratx2.cl"
#include "ratx4.cl"
#include "ratxb.cl"
#include "qssa.cl"
#include "qssa2.cl"
#include "qssab.cl"
#include "rdsmh.cl"
#include "rdwdot.cl"
#include "rdwdot2.cl"
#include "rdwdot3.cl"
#include "rdwdot6.cl"
#include "rdwdot7.cl"
#include "rdwdot8.cl"
#include "rdwdot9.cl"
#include "rdwdot10.cl"
