#pragma once
/*
* Truly Parallel Algorithms Library - Main section and list of functions
* By: David Aaron Braun
* 2022-07-08
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/


/*
* TPA MACROS
*/
#ifdef _MSC_VER
#if defined(_M_IX86 ) || defined(_M_AMD64)
#define TPA_X86_64	//Intel x86 / AMD64 Arcitecture
#define TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
#define TPA_HAS_SVML //Intel Short Vector Math Library
#include <intrin.h>
#include <immintrin.h>
#elif defined(_M_ARM)
#define TPA_ARM	//ARM 32 / ARM 64 Architecture
#include "arm_neon.h"		
#elif defined(_M_ARM64)
#define TPA_ARM //ARM 32 / ARM 64 Architecture
#include "arm_neon.h"	
#include "arm64_neon.h"
#elif defined(_M_IA64)
#define TPA_IT_64	//Intel Itanium 64 Architecture 
#include "ia64intrin.h"
#elif defined(_M_PPC)
#define TPA_PPC_64	//Power PC 32 / 64 Architecture 
#include "altivec.h"
#elif defined(__mips__)
#define TPA_MIPS	//MIPS 32 / 64 Architecture 
#include <msa.h>
#else
#warning("TPA Warning : TPA may not support this architecture.")
#endif
#elif defined(__INTEL_COMPILER)
#if defined(_M_IX86 ) || defined(_M_AMD64)
#define TPA_X86_64 //Intel x86 / AMD64 Arcitecture
#define TPA_HAS_SVML //Intel Short Vector Math Library
#define TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
#include <immintrin.h>
#elif defined(_M_IA64)
#define TPA_IT_64	//Intel Itanium 64 Architecture 
#include "ia64intrin.h"
#else
#warning("TPA Warning : TPA may not support this architecture.")
#endif
#elif defined(__GNUC__)
#ifdef __x86_64__ 
#define TPA_X86_64 //Intel x86 / AMD64 Arcitecture
#define TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
#include <immintrin.h>
#elif defined(__arm__)
#define TPA_ARM //ARM 32 / ARM 64 Arcitecture
#include "arm_neon.h"
#elif defined(__aarch64__)
#define TPA_ARM //ARM 32 / ARM 64 Architecture
#include "arm_neon.h"
#elif defined(__IA64__)
#define TPA_IT_64	//Intel Itanium 64 Architecture 
#include "ia64intrin.h"
#elif defined(__PPC__)
#define TPA_PPC_64	//Power PC 32 / 64 Architecture 
#include "altivec.h"
#elif defined(__mips__)
#define TPA_MIPS	//MIPS 32 / 64 Architecture 
#include <msa.h>
#else
#warning("TPA Warning : TPA may not support this architecture.")
#endif
#elif defined(__clang__)
#ifdef __x86_64__
#define TPA_X86_64 //Intel x86 / AMD64 Arcitecture
#define TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
#include <immintrin.h>
#elif defined(__arm__)
#define TPA_ARM //ARM 32 / ARM 64 Arcitecture
#include "arm_neon.h"
#elif defined(__aarch64__)
#define TPA_ARM //ARM 32 / ARM 64 Architecture
#include "arm_neon.h"
#elif defined(__IA64__)
#define TPA_IT_64	//Intel Itanium 64 Architecture 
#include "ia64intrin.h"
#elif defined(__PPC__)
#define TPA_PPC_64	//Power PC 32 / 64 Architecture 
#include "altivec.h"
#elif defined(__mips__)
#define TPA_MIPS	//MIPS 32 / 64 Architecture 
#include <msa.h>
#else
#warning("TPA Warning : TPA may not support this architecture.")
#endif
#elif defined(__NVCC__) || defined(__CUDACC__)
#ifdef __x86_64__
#define TPA_X86_64 //Intel x86 / AMD64 Arcitecture
#define TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
#include <immintrin.h>
#elif defined(__arm__)
#define TPA_ARM	//ARM 32 / ARM 64 Architecture
#include "arm_neon.h"
#elif defined(__aarch64__)
#define TPA_ARM //ARM 32 / ARM 64 Architecture
#include "arm_neon.h"
#elif defined(_M_IA64)
#define TPA_IT_64	//Intel Itanium 64 Architecture 
#include "ia64intrin.h"
#elif defined(_M_PPC)
#define TPA_PPC_64	//Power PC 32 / 64 Architecture 
#include "altivec.h"
#elif defined(__mips__)
#define TPA_MIPS	//MIPS 32 / 64 Architecture 
#include <msa.h>
#else
#warning("TPA Warning : TPA may not support this architecture.")
#endif
#else
#warning("TPA Warning : This compiler may not be supported by the TPA library..")
#warning("TPA Warning : TPA may not support this architecture.")
#endif
