# TPA
Truly Parallel Algorithms is a cross-platform SIMD acceleration library featuring a parallelized and vectorized implementation of std::algorithms.

Pre-Alpha Release.

|Features                                                                                    | Status                                        |
|--------------------------------------------------------------------------------------------|-----------------------------------------------|
|Runtime ISA extension detection                                                             |(complete)                                     |
|Runtime Detection of Hybrid Architecture                                                    |(complete)                                     |
|Cross-platform thread affinty and thread dispatch                                           |(not yet implemented)                          |
|An Optimized Thread Pool                                                                    |(complete, subject to change)                  |
|Functions to utilize intrinics directly on std::array-like and std::vector-like structures  |(incomplete, subject to change)                |
|Paralellized & vectorized basic math                                                        |(complete, subject to change)                  |
|Paralellized & vectorized comparisions                                                      |(complete, subject to change)                  |
|Paralellized & vectorized cmath functions                                                   |(complete, subject to change)                  |
|Paralellized & vectorized type converstion (static_cast)                                    |(not yet implemented)                          |
|Paralellized & vectorized implementation of std::algorithims                                |(very, very incomplete!, subject to change)    |
|Paralellized & vectorized implementation of std::numeric                                    |(very, very incomplete!, subject to change)    |

Requires C++20 or newer.

Detailed documentation of how to use this library will be provided at a later date, as all the information would be subject to major changes at this time. This is a pre-alpha release and this library is not intended to be widely usable at this time.

Compiler Support as of 2021-11-15:

|Platform   | MSVC  | Intel (Windows)  | Intel (Linux)   |  AMD AOCC (Linux Only)  | Clang (Windows)  | Clang (Linux)   |GCC (Linux)    |    
|-----------|-------|------------------|-----------------|-------------------------|------------------|-----------------|---------------|
|AMD-64     |YES    |No, Coming Soon   |No, Coming Soon  |No, Coming Later         |No, Coming later  |No, Coming Later |No, Coming Soon|
|x86        |Yes*   |No, Coming Soon   |No, Coming Soon  |No, Coming Later         |No, Coming later  |No, Coming Later |No, Coming Soon|
|ARM-64     |Yes**  |No, Coming Soon   |No, Coming Soon  |No, Coming Later         |No, Coming later  |No, Coming Later |No, Coming Soon|
|ARM-32***  |No     |No                |No               |No                       |No                |No               |No             |

*Building for x86 is not officially supported on any platform, it may work but I do not intend to support or make any considerations for 32-bit platforms.

**Building for ARM-64 works, however very little NEON code is implemented at this time, so this library will only allow you to benefit from multi-threading as of 2021-11-15

***There will never be any support official or otherwise for building on ARM-32 as (eventually) NEON instructions will be a requirement of this library when building for ARM and there is no good way to detect the presence of the NEON instruction set extentions on ARM-32 at runtime, they are part of the base feature set of ARM-64.


