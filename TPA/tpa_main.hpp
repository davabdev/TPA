#pragma once
/*
* Truly Parallel Algorithms Library - Main section and list of functions
* By: David Aaron Braun
* 2021-05-20
* List of functions
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

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

#include "simd/simd.hpp"			//SIMD
#include "simd/fma.hpp"				//FMA
#include "simd/trigonometry.hpp"	//Trigonometry
#include "simd/bit_manip.hpp"		//Bit Manipulation
#include "simd/rounding_math.hpp"   //abs, floor, ceil, round, round_nearest
#include "simd/roots.hpp"			//sqrt, isqrt, cbrt, icbrt, nrt, inrt
#include "simd/exponent.hpp"		//exp, exp2, exp10, expm1
#include "simd/logarithm.hpp"		//log, log2, log10, loglp, logb