// SHOC MD5Hash (level1): brute-force search of a keyspace (byteLength bytes, valsPerByte values per
// byte) for the key whose MD5 digest matches a target, with the SHOC FindKeyWithDigest_Kernel (each
// work-item hashes valsPerByte consecutive keys). As MD5Hash.cpp: a random index is chosen, its key
// and digest computed on the host (md5hash_ref.c = the host-side md5_2words / IndexToKey of
// MD5Hash.cpp), the kernel must report the same index, key and digest. SHOC's smallest size is
// 7 bytes x 10 values (10^7 keys); here 4 x 10, and work-groups of 64 instead of 256.
#include "common.h"
#include "md5hash_ref.h"
static int bytes_differ(const void *a, const void *b, int n) { const unsigned char *x = a, *y = b; for (int i = 0; i < n; i++) if (x[i] != y[i]) return 1; return 0; }   // no memcmp in the bare-metal libc
#define BYTELEN 4
#define VALS 10
#define LS 64
void FindKeyWithDigest_Kernel_ct(long n, unsigned d0, unsigned d1, unsigned d2, unsigned d3, int keyspace, int byteLength, int valsPerByte, int *foundIndex, unsigned char *foundKey, unsigned *foundDigest);
int main(void) {
  int keyspace = 1; for (int i = 0; i < BYTELEN; i++) keyspace *= VALS;
  int bad = 0;
  for (int pass = 0; pass < 3; pass++) {
    int randomIndex = (int)(rnd() % keyspace); unsigned char randomKey[8] = {0}; unsigned randomDigest[4];
    IndexToKey(randomIndex, BYTELEN, VALS, randomKey); md5_2words((unsigned *)randomKey, BYTELEN, randomDigest);
    int foundIndex = -1; unsigned char foundKey[8] = {0}; unsigned foundDigest[4] = {0};
    int nblocks = CEILDIV(keyspace / VALS, LS);
    unsigned long c0 = cyc();
    FindKeyWithDigest_Kernel_ct(NDRANGE1(nblocks * LS, LS), randomDigest[0], randomDigest[1], randomDigest[2], randomDigest[3], keyspace, BYTELEN, VALS, &foundIndex, foundKey, foundDigest);
    unsigned long c1 = cyc();
    int ok = foundIndex == randomIndex && bytes_differ(foundKey, randomKey, 8) == 0 && bytes_differ(foundDigest, randomDigest, 16) == 0;
    printf("md5hash pass %d: target index %d, found %d, key %s, digest %s (%lu cycles, %d keys)\n", pass, randomIndex, foundIndex, bytes_differ(foundKey, randomKey, 8) ? "differs" : "matches", bytes_differ(foundDigest, randomDigest, 16) ? "differs" : "matches", c1 - c0, keyspace);
    if (!ok) bad++;
  }
  printf("md5hash %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
