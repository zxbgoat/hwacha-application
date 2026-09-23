/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). The formulas and the order of floating-point operations follow kernel_*
 * in PolyBenchC-4.2.1 so the result matches the C reference exactly where the notes below say so. */
typedef float DATA_TYPE;

/* deriche: recursive Gaussian (Deriche) edge filter, four IIR passes. Each pass is a recurrence
 * along one axis, so a work-item runs one row (passes 1, 2) or one column (passes 3, 4) and the
 * combination steps are one work-item per pixel. The filter coefficients are computed on the host
 * (expf / powf) and passed in. */
__kernel void deriche_kernel1(__global DATA_TYPE *imgIn, __global DATA_TYPE *y1, DATA_TYPE a1, DATA_TYPE a2, DATA_TYPE b1, DATA_TYPE b2, int w, int h)
{
	int i = get_global_id(0);
	if (i < w)
	{
		DATA_TYPE ym1 = 0, ym2 = 0, xm1 = 0;
		int j;
		for (j = 0; j < h; j++)
		{
			y1[i*h + j] = a1*imgIn[i*h + j] + a2*xm1 + b1*ym1 + b2*ym2;
			xm1 = imgIn[i*h + j];
			ym2 = ym1;
			ym1 = y1[i*h + j];
		}
	}
}
__kernel void deriche_kernel2(__global DATA_TYPE *imgIn, __global DATA_TYPE *y2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE b1, DATA_TYPE b2, int w, int h)
{
	int i = get_global_id(0);
	if (i < w)
	{
		DATA_TYPE yp1 = 0, yp2 = 0, xp1 = 0, xp2 = 0;
		int j;
		for (j = h - 1; j >= 0; j--)
		{
			y2[i*h + j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
			xp2 = xp1;
			xp1 = imgIn[i*h + j];
			yp2 = yp1;
			yp1 = y2[i*h + j];
		}
	}
}
__kernel void deriche_kernel3(__global DATA_TYPE *imgOut, __global DATA_TYPE *y1, __global DATA_TYPE *y2, DATA_TYPE c, int w, int h)
{
	int j = get_global_id(0);
	int i = get_global_id(1);
	if ((i < w) && (j < h))
		imgOut[i*h + j] = c * (y1[i*h + j] + y2[i*h + j]);
}
__kernel void deriche_kernel4(__global DATA_TYPE *imgOut, __global DATA_TYPE *y1, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE b1, DATA_TYPE b2, int w, int h)
{
	int j = get_global_id(0);
	if (j < h)
	{
		DATA_TYPE tm1 = 0, ym1 = 0, ym2 = 0;
		int i;
		for (i = 0; i < w; i++)
		{
			y1[i*h + j] = a5*imgOut[i*h + j] + a6*tm1 + b1*ym1 + b2*ym2;
			tm1 = imgOut[i*h + j];
			ym2 = ym1;
			ym1 = y1[i*h + j];
		}
	}
}
__kernel void deriche_kernel5(__global DATA_TYPE *imgOut, __global DATA_TYPE *y2, DATA_TYPE a7, DATA_TYPE a8, DATA_TYPE b1, DATA_TYPE b2, int w, int h)
{
	int j = get_global_id(0);
	if (j < h)
	{
		DATA_TYPE tp1 = 0, tp2 = 0, yp1 = 0, yp2 = 0;
		int i;
		for (i = w - 1; i >= 0; i--)
		{
			y2[i*h + j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
			tp2 = tp1;
			tp1 = imgOut[i*h + j];
			yp2 = yp1;
			yp1 = y2[i*h + j];
		}
	}
}
