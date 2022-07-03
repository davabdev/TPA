#pragma once
/*
* Truly Parallel Algorithms Library - Main section and list of functions
* By: David Aaron Braun
* 2022-05-24
* List of functions
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
		#define TPA_X86_64
		#define TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
		#include <intrin.h>
		#include <immintrin.h>
	#elif defined(_M_ARM)
		#define TPA_ARM
		#include "arm_neon.h"		
	#elif defined(_M_ARM64)
		#define TPA_ARM
		#include "arm64_neon.h"
	#else
		#warning("TPA Warning : TPA may not support this architecture.")
	#endif
#elif defined(__INTEL_COMPILER)
	#if defined(_M_IX86 ) || defined(_M_AMD64)
		#define TPA_X86_64
		#define TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
		#include <immintrin.h>
	#else
		#warning("TPA Warning : TPA may not support this architecture.")
	#endif
#elif defined(__GNUC__)
	#ifdef __x86_64__
		#define TPA_X86_64
		#define TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
		#include <immintrin.h>
	#elif defined(__arm__)
		#define TPA_ARM
		#include "arm_neon.h"
	#elif defined(__aarch64__)
		#define TPA_ARM
		#include "arm_neon.h"
	#else
		#warning("TPA Warning : TPA may not support this architecture.")
	#endif
#elif defined(__clang__)
	#ifdef __x86_64__
		#define TPA_X86_64
		#define TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
		#include <immintrin.h>
	#elif defined(__arm__)
		#define TPA_ARM
		#include "arm_neon.h"
	#elif defined(__aarch64__)
		#define TPA_ARM
		#include "arm_neon.h"
	#else
		#warning("TPA Warning : TPA may not support this architecture.")
	#endif
#elif defined(__NVCC__) || defined(__CUDACC__)
	#ifdef __x86_64__
		#define TPA_X86_64
		#define TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
		#include <immintrin.h>
	#elif defined(__arm__)
		#define TPA_ARM
		#include "arm_neon.h"
	#elif defined(__aarch64__)
		#define TPA_ARM
	#include "arm_neon.h"
	#else
		#warning("TPA Warning : TPA may not support this architecture.")
	#endif
#else
	#warning("TPA Warning : This compiler may not be supported by the TPA library..")
	#warning("TPA Warning : TPA may not support this architecture.")
#endif


/*
* STD Headers
*/
#include <cstdlib>
#include <cstdint>

/*
* TPA headers
*/

#include "_util.hpp"				//Utility Functions
#include "size_t_lit.hpp"			//std::size_t literal suffix before C++23

#include "InstructionSet.hpp"		//CPUID

#include "numeric/iota.hpp"			//iota
#include "numeric/accumulate.hpp"	//accumulate

#include "algorithm/copy.hpp"		//copy
#include "algorithm/copy_if.hpp"	//copy_if
#include "algorithm/fill.hpp"		//fill
#include "algorithm/generate.hpp"	//generate
#include "algorithm/min_element.hpp"//min_element
#include "algorithm/max_element.hpp"//max_element
#include "algorithm/minmax_element.hpp"//minmax_element
#include "algorithm/count.hpp"		//count
#include "algorithm/count_if.hpp"	//count_if

#include "simd/simd.hpp"			//SIMD
#include "simd/fma.hpp"				//FMA
#include "simd/trigonometry.hpp"	//Trigonometry
#include "simd/bit_manip.hpp"		//Bit Manipulation & Bitwise operations
#include "simd/rounding_math.hpp"   //abs, floor, ceil, round, round_nearest
#include "simd/roots.hpp"			//sqrt, isqrt, cbrt, icbrt, nrt, inrt
#include "simd/exponent.hpp"		//exp, exp2, exp10, expm1
#include "simd/logarithm.hpp"		//log, log2, log10, loglp, logb
#include "simd/convert.hpp"			//static_cast
#include "simd/stat.hpp"			//Statistical Functions (mean, median, mode... etc.)
