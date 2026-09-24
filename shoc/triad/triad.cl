/* SHOC Triad: the kernel source Triad.cpp embeds (mTriadCLSource), verbatim. */
__kernel void Triad(__global const float *memA, 
__global const float *memB, __global float *memC, 
const float s)
{
    int gid = get_global_id(0);
    memC[gid] = memA[gid] + s*memB[gid];
}
