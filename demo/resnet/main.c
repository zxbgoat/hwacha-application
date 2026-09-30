// Hwacha host for the ResNet-50 image-classification demo: feed the preprocessed image to `net`
// (hwacha-cc assembly of the pretrained network), print the top-5 ImageNet classes and check the
// logits against PyTorch's. Image tensor + reference come from check.bin (export.py).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "labels.h"
asm(".section .rodata\n.balign 8\n.globl check_bin\ncheck_bin:\n.incbin \"" CHECK_FILE "\"\n.previous");
extern const unsigned char check_bin[];
static long rdcycle(void){long c;asm volatile("rdcycle %0":"=r"(c));return c;}
float __math_oflowf(unsigned s){return s?-1.0f/0.0f:1.0f/0.0f;}
float __math_uflowf(unsigned s){return s?-0.0f:0.0f;}
int *__errno(void){static int e;return &e;}
uintptr_t handle_trap(uintptr_t c,uintptr_t e,uintptr_t r[32]){uintptr_t t;asm volatile("csrr %0, mtval":"=r"(t));printf("TRAP cause %lx epc %lx tval %lx\n",c,e,t);exit(1);}
// bump allocator over all memory after the linked image (_image_end .. MEM_MB), see ../../torchvision/tv_main.c
#ifndef MEM_MB
#define MEM_MB 2048
#endif
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
// fixed-point float printing (the bare-metal printf has no %f)
static const char *ff(float v,int dec){static char b[8][32];static int k;char*o=b[k++&7],*q=o;long s=1;for(int i=0;i<dec;i++)s*=10;long m=(long)(fabsf(v)*s+0.5f),ip=m/s,fp=m%s;if(v<0)*q++='-';char t[24];int n=0;do{t[n++]='0'+ip%10;ip/=10;}while(ip);while(n)*q++=t[--n];if(dec){*q++='.';for(long d=s/10;d;d/=10)*q++='0'+(fp/d)%10;}*q=0;return o;}
float *net(float *);
static float in[3*224*224];
int main(void){
  const unsigned char *p=check_bin;
  int nin=*(const int*)p; p+=4; memcpy(in,p,nin*sizeof(float)); p+=nin*sizeof(float);
  int nout=*(const int*)p; p+=4; const float *ref=(const float*)p;
  printf("ResNet-50 (ImageNet-1K V2 weights) on Hwacha, input 1x3x224x224\n");
  long c0=rdcycle(); float *y=net(in); long c1=rdcycle();
  // softmax over the 1000 logits
  float mx=y[0]; for(int k=1;k<nout;k++) if(y[k]>mx) mx=y[k];
  float sum=0; static float prob[1000]; for(int k=0;k<nout;k++){prob[k]=expf(y[k]-mx); sum+=prob[k];}
  for(int k=0;k<nout;k++) prob[k]/=sum;
  // top-5 by selection
  int used[5]; printf("top-5:\n");
  for(int r=0;r<5;r++){int b=-1; for(int k=0;k<nout;k++){int skip=0; for(int j=0;j<r;j++) if(used[j]==k) skip=1; if(!skip&&(b<0||prob[k]>prob[b])) b=k;} used[r]=b;
    printf("  %2d. %s%%  %s (class %d)\n",r+1,ff(prob[b]*100,1),labels[b],b);}
  // agreement with PyTorch's logits on the same image
  float md=0,mr=0; int am=0,ra=0; for(int k=0;k<nout;k++){float d=fabsf(y[k]-ref[k]); if(d>md)md=d; if(fabsf(ref[k])>mr)mr=fabsf(ref[k]); if(y[k]>y[am])am=k; if(ref[k]>ref[ra])ra=k;}
  int ok=(md<=1e-4f+1e-2f*mr)&&am==ra;
  if(!ok){int nn=0,ni=0,f=-1; for(int k=0;k<nout;k++){if(isnan(y[k])){nn++; if(f<0)f=k;} else if(isinf(y[k])){ni++; if(f<0)f=k;}}
    printf("  %d NaN, %d inf logits (first at %d); y[0..4]=%s %s %s %s %s; ref[0..4]=%s %s %s %s %s\n",nn,ni,f,ff(y[0],3),ff(y[1],3),ff(y[2],3),ff(y[3],3),ff(y[4],3),ff(ref[0],3),ff(ref[1],3),ff(ref[2],3),ff(ref[3],3),ff(ref[4],3));}
  printf("%ld cycles; vs PyTorch: max|diff|=%s max|ref|=%s, argmax hw=%d ref=%d (%s); %s\n",c1-c0,ff(md,6),ff(mr,3),am,ra,labels[ra],ok?"ok":"FAIL");
  printf(ok?"demo PASS\n":"demo FAIL\n");
  return !ok;
}
