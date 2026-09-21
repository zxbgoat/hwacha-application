// stand-in for the Rodinia srad host header the kernel includes: only these two definitions are used
#define fp float
#define NUMBER_THREADS 64   // Rodinia uses 256; Hwacha can give this kernel set at most ~184 lanes per group
