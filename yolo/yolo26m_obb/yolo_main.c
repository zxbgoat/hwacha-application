// Host for an Ultralytics YOLO26 model (torch-mlir -> hwacha-mlir -> hwacha-cc).
// Input, output kind and a PyTorch reference come from <case>_check.bin (export_yolo.py). Entry = `net`.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
asm(".section .rodata\n.balign 8\n.globl check_bin\ncheck_bin:\n.incbin \"" CHECK_FILE "\"\n.previous");
extern const unsigned char check_bin[];
static long rdcycle(void){long c;asm volatile("rdcycle %0":"=r"(c));return c;}
float __math_oflowf(unsigned s){return s?-1.0f/0.0f:1.0f/0.0f;}
float __math_uflowf(unsigned s){return s?-0.0f:0.0f;}
int *__errno(void){static int e;return &e;}
uintptr_t handle_trap(uintptr_t c,uintptr_t e,uintptr_t r[32]){uintptr_t t;asm volatile("csrr %0, mtval":"=r"(t));printf("TRAP cause %lx epc %lx tval %lx\n",c,e,t);exit(1);}
// bump allocator over all memory after the linked image (_image_end .. MEM_MB), so the arena
// scales with the simulator's memory (spike -m<MEM_MB>) instead of a fixed static buffer.
#ifndef MEM_MB
#define MEM_MB 2048
#endif
// end of .weights_hi (yolo.ld), past the crt stacks. Taken through an absolute .dword in .data: with >2 GB
// of weights the symbol is out of pc-relative reach of this code.
asm(".section .data\n.balign 8\n.globl image_end_p\nimage_end_p: .dword _image_end\n.previous");
extern char *image_end_p;
static char *abase; static long atop, alim;
void *malloc(size_t n){
  if(!abase){abase=(char*)(((uintptr_t)image_end_p+63)&~63UL); alim=(long)(0x80000000UL+((unsigned long)MEM_MB<<20)-(uintptr_t)abase);}
  long p=(atop+63)&~63L; atop=p+n;
  if(atop>alim){printf("arena overflow: %ld MB used of %ld MB (raise MEM)\n",atop>>20,alim>>20);exit(1);}
  return abase+p;}
void free(void*p){(void)p;}
void *aligned_alloc(size_t a,size_t n){(void)a;return malloc(n);}
static const char *ff(float v){static char b[8][32];static int k;char*o=b[k++&7],*q=o;long m=(long)(fabsf(v)*1e6f+0.5f),ip=m/1000000,fp=m%1000000;if(v<0)*q++='-';char t[24];int n=0;do{t[n++]='0'+ip%10;ip/=10;}while(ip);while(n)*q++=t[--n];*q++='.';for(long d=100000;d;d/=10)*q++='0'+(fp/d)%10;*q=0;return o;}
float *net(float *);
static float in[3*256*256];  // 3x64x64, 3x128x128 for the P6 models
int main(void){
  const unsigned char *p=check_bin; int cls=*(const int*)p; p+=4;   // 1: class scores (argmax compared too)
  int isz=*(const int*)p; p+=4; memcpy(in,p,isz*sizeof(float)); p+=isz*sizeof(float);
  int nout=*(const int*)p; p+=4; const float *ref=(const float*)p;
  long c0=rdcycle(); float *y=net(in); long c1=rdcycle();
  float md=0,mx=0; int am=0; for(int k=0;k<nout;k++){float d=fabsf(y[k]-ref[k]);if(d>md)md=d;if(fabsf(ref[k])>mx)mx=fabsf(ref[k]);if(y[k]>y[am])am=k;}
  int ra=0; for(int k=0;k<nout;k++) if(ref[k]>ref[ra]) ra=k;
  // PyTorch's own forward on random weights: certify Hwacha reproduces it (numeric agreement, tolerance
  // 1e-4 + 1e-2 * max|ref| as in ../torchvision). The classification logits are also checked for argmax;
  // the dense outputs (per-anchor predictions, segmentation logits, depth maps) by the numbers alone.
  int ok = (md <= 1e-4f + 1e-2f * mx) && (!cls || am == ra);
  if(!ok){ // diagnostics: the first mismatches over the tolerance (index, hw, ref)
    float tol=1e-4f+1e-2f*mx; int n=0, nbad=0;
    for(int k=0;k<nout;k++) if(fabsf(y[k]-ref[k])>tol){ nbad++; if(n<24){ printf("  [%d] hw=%s ref=%s\n",k,ff(y[k]),ff(ref[k])); n++; } }
    printf("  %d of %d outputs over tolerance %s\n", nbad, nout, ff(tol));
  }
  printf("%s: %d outputs, %ld cycles; max|diff|=%s max|ref|=%s; argmax hw=%d ref=%d%s; %s\n", MODEL, nout, c1-c0, ff(md), ff(mx), am, ra, am==ra?" (match)":"", ok?"ok":"FAIL");
  printf(ok?"yolo PASS\n":"yolo FAIL\n");
  return !ok;
}
