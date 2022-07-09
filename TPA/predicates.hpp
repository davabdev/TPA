#pragma once
/*
* Truly Parallel Algorithms Library -
* By: David Aaron Braun
* 2022-07-08
* List of predicate enums
/*

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include "tpa_macros.hpp"
#include <cfenv>

namespace tpa
{
	/// <summary>
	/// Provides a list of valid SIMD arithmetic operation predicates.
	/// </summary>
	const enum class op {
		ADD,
		SUBTRACT,
		MULTIPLY,
		DIVIDE,
		MODULO,
		MIN, //Functionally identical to tpa::simd::compare<tpa::op::MIN>(...)
		MAX, //Functionally identical to tpa::simd::compare<tpa::op::MAX>(...)
		AVERAGE,
		POWER
	};//End of op

	/// <summary>
	/// Provides a list of valid SIMD bit wise operation predicates.
	/// </summary>
	const enum class bit {
		AND,
		OR,
		XOR,
		AND_NOT
	};//End of bit

	/// <summary>
	/// Provides a list of valid SIMD bit modification operation predicates.
	/// </summary>
	const enum class bit_mod {
		SET,					//Sets the specified bit to 1
		SET_ALL,				//Sets all the bits to 1
		CLEAR,					//Clears the specified bit to 0
		CLEAR_ALL,				//Clears all the bits to 0
		TOGGLE,					//Toggles (flips) the specified bit
		TOGGLE_ALL,				//Toggles (flips) all the bits
		REVERSE,				//Reverses the order of the bits
		SET_TRAILING_ZEROS,		//Sets all the trailing 0s to 1s
		CLEAR_TRAILING_ONES,	//Clears all the trailing 1s to 0s
		SET_LEADING_ZEROS,		//Sets all the leading 0s to 1s
		CLEAR_LEADING_ONES		//Clears all the leading 1s to 0s
	};//End of bit_mod

	/// <summary>
	/// Provides a list of valid SIMD bit counting operation predicates.
	/// </summary>
	const enum class bit_count {
		POP_COUNT,
		ONE_COUNT = POP_COUNT,
		ZERO_COUNT,
		LEADING_ZERO_COUNT,
		TRAILING_ZERO_COUNT,
		LEADING_ONE_COUNT,
		TRAILING_ONE_COUNT,
		BIT_ISLAND_COUNT,
	};//End of bit

	/// <summary>
	/// Provides a list of valid SIMD Trigonometric predicate functions
	/// </summary>
	const enum class trig
	{
		SINE,
		HYPERBOLIC_SINE,
		INVERSE_SINE,
		INVERSE_HYPERBOLIC_SINE,

		COSINE,
		HYPERBOLIC_COSINE,
		INVERSE_COSINE,
		INVERSE_HYPERBOLIC_COSINE,

		TANGENT,
		HYPERBOLIC_TANGENT,
		INVERSE_TANGENT,
		INVERSE_HYPERBOLIC_TANGENT
	};//End of trig

	/// <summary>
	/// Provides a list of valid root functions
	/// </summary>
	const enum class rt
	{
		SQUARE,
		INVERSE_SQUARE,
		CUBE,
		INVERSE_CUBE,
		N_ROOT,		   //Warning, can be very, very slow, use SQUARE / CUBE instead of nroot 2 / nroot 3
		INVERSE_N_ROOT //Warning, can be very, very slow, use SQUARE / CUBE instead of niroot 2/ niroot  3
	};//End of rt

	/// <summary>
	/// Provides a list of units of emasure for angles 
	/// </summary>
	const enum class angle {
		DEGREES,
		RADIANS
	};//End of angle

	/// <summary>
	/// <para>Provides a list of valid floating-point SIMD rounding modes</para>
	/// <para>Please note that some ARM CPUs do not support IEEE-754 rounding modes</para>
	/// </summary>
	const enum class rnd {
#if defined(TPA_X86_64)
		NEAREST_INT = _MM_FROUND_TO_NEAREST_INT,//SIMD eqivilant of FE_TONEAREST
		DOWN = _MM_FROUND_TO_NEG_INF,//SIMD eqivilant of FE_DOWNWARD
		UP = _MM_FROUND_TO_POS_INF,//SIMD equivilant of FE_UPWARD
		TRUNCATE_TO_ZERO = _MM_FROUND_TO_ZERO //SIMD eqivilant of FE_TOWARDZERO
#elif defined(TPA_ARM)
		NEAREST_INT = FE_TONEAREST,
		DOWN = FE_DOWNWARD,
		UP = FE_UPWARD,
		TRUNCATE_TO_ZERO = FE_TOWARDZERO
#else
		NEAREST_INT = FE_TONEAREST,
		DOWN = FE_DOWNWARD,
		UP = FE_UPWARD,
		TRUNCATE_TO_ZERO = FE_TOWARDZERO
#endif
	};

	/// <summary>
	/// Provides a list of valid SIMD comparison operation predicates.
	/// </summary>
	const enum class comp {
		EQUAL,
		NOT_EQUAL,
		LESS_THAN,
		LESS_THAN_OR_EQUAL,
		GREATER_THAN,
		GREATER_THAN_OR_EQUAL,
		MIN,//Functionally identical to tpa::simd::calculate<tpa::op::MIN>(...)
		MAX //Functionally identical to tpa::simd::calculate<tpa::op::MAX>(...)
	};//End of comp

	/// <summary>
	/// Provides a list of valid SIMD conditional predicates.
	/// </summary>
	const enum class cond {
		EVEN,
		ODD,
		PRIME,
		EQUAL_TO,
		NOT_EQUAL_TO,
		LESS_THAN,
		LESS_THAN_OR_EQUAL_TO,
		GREATER_THAN,
		GREATER_THAN_OR_EQUAL_TO,
		POWER_OF,
		DIVISIBLE_BY,
		FACTOR = DIVISIBLE_BY,
		MULTIPLE = DIVISIBLE_BY,
		PERFECT_SQUARE,
		FIBONACCI,
		TRIBONOCCI,
		PERFECT,
		SYLVESTER
	};//End of cond

	/// <summary>
	/// Provides a list of valid SIMD generation predicates.
	/// </summary>
	const enum class gen
	{
		EVEN,	//Generates a sequence of even numbers starting at the specifed number in param
		ODD,	//Generates a sequence of odd numbers starting at the specified number in param

		ALL_LESS_THAN,//Generates a sequence of all numbers less than the specified param upto the item_count (3rd parameter) or the container's size if not specified
		ALL_GREATER_THAN,//Generates a sequence of all numbers greater than the specified param upto the item_count (3rd parameter) or the container's size if not specified, functionally equivilent to tpa::iota<T>()

		XOR_SHIFT,//'param' is the minimum random number and 'param2' is the maximum random number.
		STD_RAND,//'param' is the minimum random number and 'param2' is the maximum random number.
		SECURE_RAND, //'param' is the minimum random number and 'param2' is the maximum random number. Uses RD_RAND & RD_SEED where available, VERY, VERY slow!
		UNIFORM,//'param' is the minimum random number and 'param2' is the maximum random number.
		BERNOULLI,//'param' is the frequency of 'truths'
		BINOMIAL,//'param' is the number of trials, 'param2' is the frequncy of of success
		NEGATIVE_BINOMIAL,//'param' is the number of trials, 'param2' is the frequncy of of success
		GEOMETRIC,//'param' is the number of coin tosses that are requried to get heads
		POISSON,//'param' is the mean
		EXPONENTIAL,//'param' is the constant time
		GAMMA,//'param' is the alpha, 'param2' is the beta
		WEIBULL,//'param' is the shape, 'param2' is the scale
		EXTREME_VALUE,//'param' is the location, 'param2' is the scale
		NORMAL,//'param' is the mean, 'param2' is the standard deviation
		LOG_NORMAL,//'param' is the mean, 'param2' is the standard deviation
		CHI_SQUARED,//'param' is the degress of freedom
		CAUCHY,//'param' is the location, 'param2' is the scale
		FISHER_F,//'param' is the first degree of freedom, 'param2' is the second degree of freedom
		STUDENT_T//'param' is the number of degrees of freedom
	};//End of gen

	/// <summary>
	/// Provides a list of valid sequences to generate
	/// </summary>
	const enum class seq {
		PRIME,
		PARTITION_NUMBERS,
		POWERS,
		FACTORIAL,
		DIVISOR_FUNCTION,
		PRIME_POWERS,
		KOLAKOSKI,
		EULER_TOTIENT,
		LUCAS_NUMBERS,
		FIBONACCI,
		TRIBONOCCI,
		SYLVESTER,
		POLYMINOES,
		CATALAN,
		BELL_NUMBERS,
		EULER_ZIG_ZAG,
		LAZY_CATERERS_NUMBERS,
		CENTRAL_POLYGONAL_NUMBERS,
		PELL_NUMBERS,
		DERANGEMENTS,
		FERMAT_NUMBERS,
		POLYTREES,
		PERFECT_NUMBERS,
		RAMANUJAN_TAU_FUNCTION,
		LANDAU_FUNCTION,
		NARAYANS_COWS,
		PADOVAN,
		EUCLID_MULLIN,
		LUCKY_NUMBERS,
		CENTRAL_BINOMIAL_CO,
		MOTZKIN_NUMBERS,
		JACOBSTHAL_NUMBERS,
		SUM_OF_PROPER_DIVISORS,
		WEDDERBURN_ETHERINGTON_NUMBERS,
		GOULD,
		SEMI_PRIMES,
		GOLOMB,
		PERRIN_NUMBERS,
		CULLEN_NUMBERS,
		PRIMORIALS,
		COMPOSITE_NUMBERS,
		HIGHLY_COMPOSITE_INTEGERS,
		SUPERIOR_HIGHLY_COMPOSITE_INTEGERS,
		PRONIC_NUMBERS,
		MARKOV_NUMBERS,
		ULAM_NUMBERS,
		PRIME_KNOTS,
		CARMICHAEL_NUMBERS,
		WOODALL_NUMBERS,
		ARITHMETIC_NUMBERS,
		ABUNDANT_NUMBERS,
		COLOSSALLY_ABUNDANT_NUMBERS,
		ALCUIN,
		UNTOUCHABLE_NUMBERS,
		RECAMAN,
		LOOK_AND_SAY,
		PRACTICAL_NUMBERS,
		ALTERNATING_FACTORIAL,
		FORTUNATE_NUMBERS,
		SEMI_PERFECT_NUMBERS,
		MAGIC_CONSTANTS,
		WEIRD_NUMBERS,
		FAREY_NUMERATORS,
		FAREY_DENUMERATORS,
		EUCLID_NUMBERS,
		KAPREKAR_NUMBERS,
		SPHENIC_NUMBERS,
		GUIGA_NUMBERS,
		RADICAL_OF_INTEGER,
		THUE_MORSE,
		REGULAR_PAPERFOLDING,
		BLUM_INTEGERS,
		MAGIC_NUMBERS,
		SUPER_PERFECT_NUMBERS,
		BERNOULLI_NUMBERS,
		HYPER_PERFECT_NUMBERS,
		ACHILLES_NUMBERS,
		PRIMARY_PSEUDO_PERFECT_NUMBERS,
		ERDOS_WOODS_NUMBERS,
		SIERPINSKI_NUMBERS,
		RIESEL_NUMBERS,
		BAUM_SWEET,
		GIJSWIT,
		CAROL_NUMBERS,
		JUGGLER,
		HIGHLY_TOTIENT_NUMBERS,
		EULER_NUMBERS,
		POLITE_NUMBERS,
		ERDOS_NICOLAS_NUMBERS,
		STAR_NUMBERS,
		STELLA_OCTAGULA_NUMBERS,
		ARONSON,
		HARSHAD_NUMBERS,
		FACTORIONS,
		UNDULATING_NUMBERS,
		EQUIDIGITAL_NUMBERS,
		EXTRAVAGANT_NUMBERS,
		PANDIGITAL_NUMBERS,
		TRIANGULAR,
		SQUARE,
		CUBE,
		PALINDROMIC,
		PERMUTABLE_PRIMES,
		CIRCULAR_PRIMES,
		HOME_PRIMES
	};//End of seq
}//End of namespace
