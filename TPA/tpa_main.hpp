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
* STD Headers
*/
#include <cstdlib>
#include <cstdint>

/*
* TPA headers
*/
#include "tpa.hpp"					//Thread Pool
#include "tpa_macros.hpp"			//TPA Lib Macros 
#include "predicates.hpp"			//Predicate Enums
#include "tpa_concepts.hpp"			//TPA Lib Concepts
#include "Timer.hpp"				//Scope-Based Timer Class
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

#include "simd/simd.hpp"			//SIMD Utility Functions
#include "simd/basic_math.hpp"		//Basic Math
#include "simd/fma.hpp"				//FMA
#include "simd/trigonometry.hpp"	//Trigonometry
#include "simd/bit_manip.hpp"		//Bit Manipulation & Bitwise operations
#include "simd/rounding_math.hpp"   //abs, floor, ceil, round, round_nearest
#include "simd/roots.hpp"			//sqrt, isqrt, cbrt, icbrt, nrt, inrt
#include "simd/exponent.hpp"		//exp, exp2, exp10, expm1
#include "simd/logarithm.hpp"		//log, log2, log10, loglp, logb
#include "simd/convert.hpp"			//static_cast
#include "simd/stat.hpp"			//Statistical Functions (mean, median, mode... etc.)
