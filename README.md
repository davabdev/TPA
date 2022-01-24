# TPA
Truly Parallel Algorithms is a cross-platform SIMD acceleration library featuring a parallelized and vectorized implementation of std::algorithms.

Pre-Alpha Release.

|Features                                                                                    | Status                                        |
|--------------------------------------------------------------------------------------------|-----------------------------------------------|
|Runtime ISA extension detection                                                             |complete                                       |
|Runtime Detection of Hybrid Architecture                                                    |complete                                       |
|Cross-platform thread affinty and thread dispatch                                           |not yet implemented                            |
|An Optimized Thread Pool                                                                    |complete, subject to change                    |
|Functions to utilize intrinics directly on std::array-like and std::vector-like structures  |incomplete, subject to change                  |
|Paralellized & vectorized basic math                                                        |complete, subject to change                    |
|Paralellized & vectorized comparisions                                                      |complete, subject to change                    |
|Paralellized & vectorized cmath functions                                                   |complete, subject to change                    |
|Paralellized & vectorized type converstion (static_cast)                                    |incomplete                                     |
|Paralellized & vectorized implementation of std::algorithims                                |very, very incomplete!, subject to change      |
|Paralellized & vectorized implementation of std::numeric                                    |very, very incomplete!, subject to change      |
|Paralellized & vectorized string functions                                                  |not yet implemented                            |
|Paralellized & vectorized random number generators                                          |complete, subject to change                    |
|Paralellized & vectorized bitwise operations                                                |complete, subject to change                    |
|Bit Twiddling                                                                               |complete                                       |
|Paralellized & vectorized advanced bit manipulation                                         |incomplete, subject to change                  |
|Paralellized & vectorized statistical functions                                             |incomplete, subject to change                  |
|Paralellized & vectorized finance functions                                                 |not yet implemented                            |  

Requires C++20 or newer.

Detailed documentation of how to use this library will be provided at a later date, as all the information would be subject to major changes at this time. This is a pre-alpha release and this library is not intended to be widely usable at this time. See the "Testing" Solution to get an idea of how to use this library and the performance benefits available. 

ISA Support (including runtime detection and dispatch) as of 2021-11-15 (this list is incomplete)
|ISA         |Run-time Detection|Dispatch & Execution  |
|------------|------------------|----------------------|
|MMX         |Yes               |No*                   |
|3DNow!      |Yes               |No**                  |
|3DNow! EXT  |Yes               |No**                  |
|SSE         |Yes               |Yes, where applicable |
|SSE2        |Yes               |Yes, where applicable |
|SSE3        |Yes               |Yes, where applicable |
|SSSE3       |Yes               |Yes, where applicable |
|SSE4a       |Yes               |No                    |
|XOP         |Yes               |No                    |
|SSE4.1      |Yes               |Yes, where applicable |
|SSE4.2      |Yes               |Yes, where applicable |
|AVX         |Yes               |Yes, where applicable |
|AVX-2       |Yes               |Yes, where applicable |
|FMA         |Yes               |Yes, where applicable |
|AVX-VNNI    |Yes               |Coming later          |
|AVX-512F    |Yes               |Yes, where applicable |
|AVX-512BW   |Yes               |Yes, where applicable |
|AVX-512DQ   |Yes               |Yes, where applicable |
|AVX-512PF   |Yes               |No                    |
|AVX-512ER   |Yes               |No                    |
|AVX-512CD   |Yes               |No                    |
|AVX-512VL   |Yes               |No                    |
|AVX-512VBMI |Yes               |Coming later          |
|AVX-512VBMI2|Yes               |Coming later          |
|AVX-512VNNI |Yes               |Coming later          |
|AVX-512BF   |Yes               |No                    |
|AVX-512F16  |Yes               |No                    |
|AVX512-AMX  |Yes               |No                    |
|BMI1        |Yes               |Yes                   |
|BMI2        |Yes               |Yes                   | 
|NEON        |No***             |Yes                   |

*MMX has been deprecated on Intel CPUs for years and is not utlized in this library.

**3DNow! has never been available on Intel and has been removed on AMD since First Gen Ryzen. 3DNow! Extented has possibly never been implemented.

***NEON is required when compiling for ARM


Compiler Support as of 2021-11-15:

|Platform   | MSVC  | Intel (Windows)  | Intel (Linux)   |  AMD AOCC (Linux Only)  | Clang (Windows)  | Clang (Linux)   |GCC (Linux)    | NVCC |
|-----------|-------|------------------|-----------------|-------------------------|------------------|-----------------|---------------|------|
|AMD-64     |YES    |No, Coming Soon   |No, Coming Soon  |No, Coming Later         |No, Coming later  |No, Coming Later |No, Coming Soon| No   |
|x86        |Yes*   |No, Coming Soon   |No, Coming Soon  |No, Coming Later         |No, Coming later  |No, Coming Later |No, Coming Soon| No   |
|ARM-64     |Yes**  |No, Coming Soon   |No, Coming Soon  |No, Coming Later         |No, Coming later  |No, Coming Later |No, Coming Soon| No   |
|ARM-32***  |No     |No                |No               |No                       |No                |No               |No             | No   |

*Building for x86 is not officially supported on any platform, it may work but I do not intend to support or make any considerations for 32-bit platforms.

**Building for ARM-64 works, however very little NEON code is implemented at this time, so this library will only allow you to benefit from multi-threading as of 2021-11-15

***There will never be any support official or otherwise for building on ARM-32 as (eventually) NEON instructions will be a requirement of this library when building for ARM and there is no good way to detect the presence of the NEON instruction set extentions on ARM-32 at runtime, they are part of the base feature set of ARM-64.

The Nvidia CUDA C++ Compiler still has zero C++20 support as of 2021-12-06. Until C++20 is avaialble on NVCC, there will be no support official or unofficial for compiling with NVCC whatsoever. 
