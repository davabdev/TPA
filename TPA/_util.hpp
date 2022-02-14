#pragma once
/*
* Truely Parallel Algorithms Library - Utility Functions
* By: David Aaron Braun
* 2021-04-08
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include <type_traits>
#include <concepts>
#include <iterator>
#include <vector>
#include <forward_list>
#include <thread>
#include <utility>
#include <iostream>
#include <chrono>
#include <mutex>
#include <numbers>
#include <bitset>
#include <bit>

#include "ThreadPool.hpp"
#include "excepts.hpp"
#include "size_t_lit.hpp"

#ifdef _M_AMD64
	#include <immintrin.h>
#elif defined (_M_ARM64)
#ifdef _WIN32
#include "arm64_neon.h"
#else
#include "arm_neon.h"
#endif
#endif

#undef max
#undef min
#undef abs
#undef fabs
#undef pow
#undef sqrt
#undef cbrt

/// <summary>
/// The tpa::util namespace provides utility functions for TPA, it is not inteded to be accessed by users of this library. However you may find something useful.
/// </summary>
namespace tpa::util 
{
	std::mutex consoleMtx;

	template<typename CONT, typename ITER = CONT::iterator, typename VAL = CONT::value_type>
	/// <summary>
	/// <para> concept contiguous_seqeunce requires: </para>
	/// <para> size() function returning an integer convertible to std::size_t </para>
	/// <para> An implementation of the subscript operator[] returning a const T &amp; </para>
	/// <para> Container's iterator must satisfy all the requirements of std::contiguous_iterator </para>
	/// </summary>
	concept contiguous_seqeunce = requires(const CONT & cont, const std::size_t index) {
		{cont.size()} -> std::convertible_to<std::size_t>;
		{cont[index]} -> std::convertible_to<const VAL&>;
		std::contiguous_iterator<ITER>;
	};	

	template<typename T>
	/// <summary>
	/// <para>Concept calculatable requires: </para>
	/// <para>operator overload  for = </para>
	/// <para>operator overloads for ==  and != </para>
	/// <para>operator overloads for +, -, *, and / </para>
	/// <para>operator overloads for (POST) ++, -- </para>
	/// <para>operator overloads for (PRE) ++, -- //</para>
	/// <para>operator overloads for +=, -=, *=, and /= </para>
	/// </summary>
	concept calculatable = requires(T t)
	{
		t == t;
		t != t;

		t = t;

		t + t;
		t - t;
		t * t;
		t / t;

		++t;
		t++;
		
		--t;
		t--;
		
		t += t;
		t -= t;
		t *= t;
		t /= t;		
	};

	/// <summary>
	/// <para>Converts any variable or object to a bitset</para>
	/// <para>Padding bits may be included depending on your compiler.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="var"></param>
	template<typename T, size_t SIZE = (sizeof(T) * CHAR_BIT)>
	std::bitset<SIZE> as_bits(T var) noexcept
	{
		if constexpr (SIZE < 32)//Size in bits
		{
			int32_t temp = 0;
			std::memmove(&temp, &var, sizeof(T));

			std::bitset<SIZE> bits = var;
			return bits;
		}//End if
		else
		{
			std::bitset<SIZE> bits = std::bit_cast<std::bitset<SIZE>, T>(var);
			return bits;
		}//End else
	}//End of as_bits

	/// <summary>
	/// <para>Branchless min value function</para>
	/// <para> Caution usually not any faster than std::min (hint on MSVC use '#undef min' to use the actual std::min and not the macro)</para>
	/// </summary>
	/// <typeparam name="A"></typeparam>
	/// <typeparam name="B"></typeparam>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns></returns>
	template<typename A, typename B>
	[[nodiscard]] inline constexpr auto min(const A a, const B b) noexcept -> decltype(a+b)
	{
		return (a * (a < b) + b * (b <= a));
	}//End of min

	/// <summary>
	/// <para>Branchless max value function</para>
	/// <para> Caution usually not any faster than std::max (hint on MSVC use '#undef max' to use the actual std::max and not the macro)</para>
	/// </summary>
	/// <typeparam name="A"></typeparam>
	/// <typeparam name="B"></typeparam>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns></returns>
	template<typename A, typename B>
	[[nodiscard]] inline constexpr auto max(const A a, const B b) noexcept -> decltype(a + b)
	{
		return ((a > b) * a + (a <= b) * b);
	}//End of max

	/// <summary>
	/// <para>Calculate powers</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="base"></param>
	/// <param name="exp"></param>
	/// <returns>T</returns>
	template <typename T, typename E>
	[[nodiscard]] inline constexpr T pow(const T base, const E exp) noexcept
	{
		if constexpr (std::is_floating_point<T>() || std::is_floating_point<E>())
		{
			return static_cast<T>(std::pow(base, exp));
		}//End if
		else
		{
			if (exp == 0) return static_cast<T>(1);
			else if (exp == 1) return base;
			else if (exp == 2) return static_cast<T>(base * base);
			else
			{
				T temp = 1;
				for (size_t i = 1; i <= exp; ++i)
				{
					temp *= base;
				}//End for
				return static_cast<T>(temp);
			}//End else
		}//End else
	}//End of power

	/// <summary>
	/// <para>Computes e (Euler's number, 2.7182818...) raised to the power 'pow' and returns the result</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="pow"></param>
	/// <returns>T</returns>
	template<typename T>
	[[nodiscard]] inline constexpr T exp(const T pow) noexcept
	{
		if constexpr (std::is_floating_point<T>())
		{
			return static_cast<T>(std::exp(pow));
		}//End if
		else
		{
			return static_cast<T>(tpa::util::pow(std::numbers::e, pow));
		}//End else
	}//End of exp

	/// <summary>
	/// <para>Computes 2 raised to the given power 'pow'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="pow"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T exp2(const T pow) noexcept
	{
		if constexpr (std::is_floating_point<T>())
		{
			return static_cast<T>(std::exp2(pow));
		}//End if
		else
		{
			return static_cast<T>(tpa::util::pow(2ull, pow));
		}//End else
	}//End of exp2

	/// <summary>
	/// <para>Computes 10 raised to the given power 'pow'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="pow"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T exp10(const T pow) noexcept
	{
		if constexpr (std::is_floating_point<T>())
		{
			return static_cast<T>(std::pow(10.0, pow));
		}//End if
		else
		{
			return static_cast<T>(tpa::util::pow(10ull, pow));
		}//End else
	}//End of exp10

	/// <summary>
	/// <para>Computes e (Euler's number, 2.7182818...) raised to the given power 'pow' minus 1</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="pow"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T expm1(const T pow) noexcept
	{
		if constexpr (std::is_floating_point<T>())
		{
			return static_cast<T>(std::expm1(pow));
		}//End if
		else
		{
			return static_cast<T>(tpa::util::pow(std::numbers::e, pow) - 1ull);
		}//End else
	}//End of expm1

	/// <summary>
	/// Get the size of an std::forward_list
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="lst"></param>
	/// <returns></returns>
	template <typename T>
	[[nodiscard]] inline constexpr size_t size(const std::forward_list<T>& lst) noexcept
	{
		return static_cast<size_t>(std::distance(lst.begin(), lst.end()));
	}//End of size()

	/// <summary>
	/// Checks is a number is a prime number
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="n"></param>
	/// <returns></returns>
	template <typename T>
	inline constexpr bool isPrime(const T n) noexcept
	{
		if (n < 2) 
		{
			return false;
		}//End if

		size_t i = 2uz;

		for (; static_cast<size_t>(i * i) <= n; ++i) 
		{
			if (n % i == 0)
			{
				return false;
			}//End if
		}//End for

		return true;
	}//End of isPrime()

	/// <summary>
	/// <para>Returns true if the value is an even number.</para>
	/// <para>If type is non-standard, requires that the type has implemented operator modulo ( % )</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="n"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr bool isEven(const T n) noexcept
	{
		if constexpr (std::is_floating_point<T>())
		{
			return std::fmod(n, 2) == 0.0;
		}//End if
		else
		{
			return n % 2 == 0;
		}//End else
	}//End of isEven

	/// <summary>
	/// <para>Returns true if the value is an odd number.</para>
	/// <para>If type is non-standard, requires that the type has implemented operator modulo ( % )</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="n"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr bool isOdd(const T n) noexcept
	{
		if constexpr (std::is_floating_point<T>())
		{
			return std::fmod(n, 2) != 0.0;
		}//End if
		else
		{
			return n % 2 != 0;
		}//End else
	}//End of isOdd

	/// <summary>
	/// Branchless Absolute Value function
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="num"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T abs(const T num) noexcept
	{
		return num * ((num > 0) - (num < 0));
	}//End of abs

	/// <summary>
	/// <para>Rounds an integer number to the nearest multiple specified in 'mult'</para>
	/// <para>Negative Numbers may only round up.</para>
	/// <para>The multiple 'mult' must be an integral type or causes rounding errors.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <typeparam name="MULT"></typeparam>
	/// <param name="num"></param>
	/// <param name="mult"></param>
	/// <returns></returns>
	template<typename T, typename MULT>
	[[nodiscard]] inline constexpr T round_to_nearest(const T num, const MULT mult) noexcept
	{
		static_assert(!std::is_floating_point<MULT>(), "You cannot specify a floating-point number as a multiple");

		fesetround(FE_TONEAREST);

		const MULT half = mult / 2;

		//2-Layer static-cast to force integer divison on floats
		return static_cast<T>(static_cast<MULT>((num + half) / mult) * mult);
	}//End of round_to_nearest

#pragma region roots

	/// <summary>
	/// <para>Calculates n root of a number</para>
	/// <para>Example:</para>
	/// <para>Square Root: n_root(number, 2)</para>
	/// <para>Cube Root: n_root(number, 3)</para>
	/// <para>Non-standard types will work.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <typeparam name="R"></typeparam>
	/// <param name="num"></param>
	/// <param name="root"></param>
	/// <returns></returns>
	template<typename T, typename R>
	[[nodiscard]] inline constexpr T n_root(const T num, const R root) noexcept
	{
		return static_cast<T>(tpa::util::pow(num, static_cast<R>(1.0 / root)));
	}//End of n_root

	/// <summary>
	/// <para>Calculates n Inverse Root of a number</para>
	/// <para>Example:</para>
	/// <para>Inverse Square Root: n_iroot(number, 2)</para>
	/// <para>Inverse Cube Root: n_iroot(number, 3)</para>
	/// <para>Non-standard types will work.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <typeparam name="R"></typeparam>
	/// <param name="num"></param>
	/// <param name="root"></param>
	/// <returns></returns>
	template<typename T, typename R = uint32_t>
	[[nodiscard]] inline constexpr T n_iroot(const T num, const R root) noexcept
	{
		return static_cast<T>(1.0 / tpa::util::n_root(num, root));
	}//End of n_iroot

	/// <summary>
	/// <para>Optimimally Computes the Square Root of a number</para>
	/// <para>Non-standard types will likely be truncated.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="num"></param>
	/// <param name="inaccurateOptimization"> true by default, uses SSE instructions if available at the expense of some accuracy</param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T sqrt(const T num, const bool inaccurateOptimization = true) noexcept
	{
		if constexpr (std::is_same<T, float>())
		{
#ifdef _M_AMD64
			if (inaccurateOptimization && tpa::has_SSE)
			{
				__m128 _num = _mm_set1_ps(num);

				_num = _mm_sqrt_ps(_num);

#ifdef _WIN32
				return static_cast<T>(_num.m128_f32[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(std::sqrt(num));
#ifdef _M_AMD64
			}//End if
#endif
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
#ifdef _M_AMD64
			if (inaccurateOptimization && tpa::has_SSE2)
			{
				__m128d _num = _mm_set1_pd(num);

				_num = _mm_sqrt_pd(_num);

#ifdef _WIN32
				return static_cast<T>(_num.m128d_f64[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(std::sqrt(num));
#ifdef _M_AMD64
			}//End if
#endif
		}//End if		
		else
		{
			return static_cast<T>(std::sqrt(num));
		}//End else
	}//End of sqrt

	/// <summary>
	/// <para>Optimimally Computes the Inverse Square Root of a number</para>
	/// <para>Non-standard types will likely be truncated.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="num"></param>
	/// <param name="inaccurateOptimization"> true by default, uses SSE instructions if available at the expense of some accuracy</param>
	/// <returns></returns>
	template<typename T> 
	[[nodiscard]] inline constexpr T isqrt(const T num, const bool inaccurateOptimization = true) noexcept
	{
		if constexpr (std::is_same<T, float>())
		{
#ifdef _M_AMD64
			if (inaccurateOptimization && tpa::has_SSE)
			{				
				__m128 _num = _mm_set1_ps(num);

				_num = _mm_invsqrt_ps(_num);

#ifdef _WIN32
				return static_cast<T>(_num.m128_f32[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(1.0f / std::sqrt(num));
#ifdef _M_AMD64
			}//End if
#endif
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
#ifdef _M_AMD64
			if (inaccurateOptimization && tpa::has_SSE2)
			{
				__m128d _num = _mm_set1_pd(num);

				_num = _mm_invsqrt_pd(_num);

#ifdef _WIN32
				return static_cast<T>(_num.m128d_f64[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(1.0 / std::sqrt(num));
#ifdef _M_AMD64
			}//End if
#endif
		}//End if		
		else
		{
			return static_cast<T>(1.0 / std::sqrt(num));
		}//End else
	}//End of isqrt

	/// <summary>
	/// <para>Optimimally computes the Cube Root</para>
	/// <para>Non-standard types will likely be truncated.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="num"></param>
	/// <param name="inaccurateOptimization"> true by default, uses SEE instructions if available at the expense of some accuracy</param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T cbrt(const T num, const bool inaccurateOptimization = true) noexcept
	{
		if constexpr (std::is_same<T, float>())
		{
#ifdef _M_AMD64
			if (inaccurateOptimization && tpa::has_SSE)
			{
				__m128 _num = _mm_set1_ps(num);

				_num = _mm_cbrt_ps(_num);

#ifdef _WIN32
				return static_cast<T>(_num.m128_f32[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(std::cbrt(num));
#ifdef _M_AMD64
			}//End if
#endif
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
#ifdef _M_AMD64
			if (inaccurateOptimization && tpa::has_SSE2)
			{
				__m128d _num = _mm_set1_pd(num);

				_num = _mm_cbrt_pd(_num);

#ifdef _WIN32
				return static_cast<T>(_num.m128d_f64[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(std::cbrt(num));
#ifdef _M_AMD64
			}//End if
#endif
		}//End if		
		else
		{
			return static_cast<T>(std::cbrt(num));
		}//End else
	}//End of cbrt

	/// <summary>
	/// <para>Optimimally computes the Inverse Cube Root</para>
	/// <para>Non-standard types will likely be truncated.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="num"></param>
	/// <param name="inaccurateOptimization"> true by default, uses SEE instructions if available at the expense of some accuracy</param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T icbrt(const T num, const bool inaccurateOptimization = true) noexcept
	{
		if constexpr (std::is_same<T, float>())
		{
#ifdef _M_AMD64
			if (inaccurateOptimization && tpa::has_SSE)
			{
				__m128 _num = _mm_set1_ps(num);

				_num = _mm_invcbrt_ps(_num);

#ifdef _WIN32
				return static_cast<T>(_num.m128_f32[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(1.0f / std::cbrt(num));
#ifdef _M_AMD64
			}//End if
#endif
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
#ifdef _M_AMD64
			if (inaccurateOptimization && tpa::has_SSE2)
			{
				__m128d _num = _mm_set1_pd(num);

				_num = _mm_invcbrt_pd(_num);

#ifdef _WIN32
				return static_cast<T>(_num.m128d_f64[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(1.0 / std::cbrt(num));
#ifdef _M_AMD64
			}//End if
#endif
		}//End if		
		else
		{
			return static_cast<T>(1.0 / std::cbrt(num));
		}//End else
	}//End of icbrt
#pragma endregion

#pragma region degrees_and_radians

	/// <summary>
	/// <para>List of radian-to-degree and degree-to-radian offsets.</para>
	/// <para>Avoids divide instruction</para>
	/// </summary>
	namespace deg_rad 
	{
		inline constexpr double d2r_offset = static_cast<double>(std::numbers::pi / 180.0);
		inline constexpr float f_d2r_offset = static_cast<float>(std::numbers::pi / 180.0f);

		inline constexpr double r2d_offset = static_cast<double>(180.0 / std::numbers::pi);
		inline constexpr float f_r2d_offset = static_cast<float>(180.0f / std::numbers::pi);

#ifdef _M_AMD64	

		inline constexpr __m512 avx512_f_r2d_offset = { f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset };

		inline constexpr __m512d avx512_d_r2d_offset = { r2d_offset, r2d_offset, r2d_offset, r2d_offset, r2d_offset, r2d_offset, r2d_offset, r2d_offset };

		inline constexpr __m256 avx256_f_r2d_offset = { f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset };

		inline constexpr __m256d avx256_d_r2d_offset = { r2d_offset, r2d_offset, r2d_offset, r2d_offset };

		inline constexpr __m128 sse_f_r2d_offset = { f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset};

		inline constexpr __m128d sse2_d_r2d_offset = { r2d_offset, r2d_offset };

		inline constexpr __m512 avx512_f_d2r_offset = { f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset};

		inline constexpr __m512d avx512_d_d2r_offset = { d2r_offset, d2r_offset, d2r_offset, d2r_offset, d2r_offset, d2r_offset, d2r_offset, d2r_offset };

		inline constexpr __m256 avx256_f_d2r_offset = { f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset };

		inline constexpr __m256d avx256_d_d2r_offset = { d2r_offset, d2r_offset, d2r_offset, d2r_offset };

		inline constexpr __m128 sse_f_d2r_offset = { f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset };

		inline constexpr __m128d sse2_d_d2r_offset = { d2r_offset, d2r_offset };
#endif
	}//End of namespace

	/// <summary>
	/// <para>Converts degrees to radians</para>
	/// <para>Includes support for AVX-512, AVX-256 and SSE registers</para>
	/// <para>If a SIMD register (such as __m256) is passed to this function and the CPU does not support the necessary instruction set this function is unsafe and may either generate an "Unsupported Instruction Exception" or crash the program.</para>
	/// <para>If the compiler cannot deduce the type of argument "degree" you must specify it yourself     </para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="degree"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T degrees_to_radians(const T& degree) noexcept
	{
#ifdef _M_AMD64
		static_assert(!std::is_same<T, __m512i>() && !std::is_same<T, __m256i>() && !std::is_same<T, __m128i>(), "Integer-type SIMD registers are not supported as there is no way to automatically determin thier data type. Use tpa::simd::calculate to compute: (n * (pi / 180)) instead.");

		if constexpr (std::is_same<T, __m512>())
		{
			return _mm512_mul_ps(degree, deg_rad::avx512_f_d2r_offset);
		}//End of AVX-512 float
		else if constexpr (std::is_same<T, __m512d>())
		{
			return _mm512_mul_pd(degree, deg_rad::avx512_d_d2r_offset);
		}//End of AVX-512 double
		else if constexpr (std::is_same<T, __m256>())
		{
			return _mm256_mul_ps(degree, deg_rad::avx256_f_d2r_offset);
		}//End of AVX-256 float
		else if constexpr (std::is_same<T, __m256d>())
		{
			return _mm256_mul_pd(degree, deg_rad::avx256_d_d2r_offset);
		}//End of AVX-256 double
		else if constexpr (std::is_same<T, __m128>())
		{
			return _mm_mul_ps(degree, deg_rad::sse_f_d2r_offset);
		}//End of SSE float
		else if constexpr (std::is_same<T, __m128d>())
		{
			return _mm_mul_pd(degree, deg_rad::sse2_d_d2r_offset);
		}//End of SSE2 double
		else if constexpr(std::is_same<T, float>())
		{
			return (degree * deg_rad::f_d2r_offset);
		}//End scaler float
		else if constexpr (std::is_same<T, double>())
		{
			return (degree * deg_rad::d2r_offset);
		}//End scaler double
		else
		{
			return static_cast<T>(degree * deg_rad::d2r_offset);
		}//End else
#else
		if constexpr (std::is_same<T, float>())
		{
			return (degree * deg_rad::f_d2r_offset);
		}//End scaler float
		else if constexpr (std::is_same<T, double>())
		{
			return (degree * deg_rad::d2r_offset);
		}//End scaler double
		else
		{
			return static_cast<T>(degree * deg_rad::d2r_offset);
		}//End else
#endif
	}//End of degrees_to_radians	

	/// <summary>
	/// <para>Converts radians to degrees</para>
	/// <para>Includes support for AVX-512, AVX-256 and SSE registers</para>
	/// <para>If a SIMD register (such as __m256) is passed to this function and the CPU does not support the necessary instruction set this function is unsafe and may either generate an "Unsupported Instruction Exception" or crash the program.</para>
	/// <para>If the compiler cannot deduce the type of argument "radian" you must specify it yourself     </para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="radian"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T radians_to_degrees(const T& radian) noexcept
	{
#ifdef _M_AMD64
		static_assert(!std::is_same<T, __m512i>() && !std::is_same<T, __m256i>() && !std::is_same<T, __m128i>(), "Integer-type SIMD registers are not supported as there is no way to automatically determin thier data type. Use tpa::simd::calculate to compute: (n * (180 / pi)) instead.");

		if constexpr (std::is_same<T, __m512>())
		{
			return _mm512_mul_ps(radian, deg_rad::avx512_f_r2d_offset);
		}//End of AVX-512 float
		else if constexpr (std::is_same<T, __m512d>())
		{
			return _mm512_mul_pd(radian, deg_rad::avx512_d_r2d_offset);
		}//End of AVX-512 double
		else if constexpr (std::is_same<T, __m256>())
		{
			return _mm256_mul_ps(radian, deg_rad::avx256_f_r2d_offset);
		}//End of AVX-256 float
		else if constexpr (std::is_same<T, __m256d>())
		{
			return _mm256_mul_pd(radian, deg_rad::avx256_d_r2d_offset);
		}//End of AVX-256 double
		else if constexpr (std::is_same <T, __m128>())
		{
			return _mm_mul_ps(radian, deg_rad::sse_f_r2d_offset);
		}//End of SSE float
		else if constexpr (std::is_same<T, __m128d>())
		{
			return _mm_mul_pd(radian, deg_rad::sse2_d_r2d_offset);
		}//End of SSE2 double
		else if constexpr (std::is_same<T, float>())
		{
			return (radian * deg_rad::f_r2d_offset);
		}//End scaler float
		else if constexpr (std::is_same<T, double>())
		{
			return (radian * deg_rad::r2d_offset);
		}//End scaler double
		else
		{
			return static_cast<T>(radian * deg_rad::r2d_offset);
		}//End else
#else
		if constexpr (std::is_same<T, float>())
		{
			return (radian * deg_rad::f_r2d_offset);
		}//End scaler float
		else if constexpr (std::is_same<T, double>())
		{
			return (radian * deg_rad::r2d_offset);
		}//End scaler double
		else
		{
			return static_cast<T>(radian * deg_rad::r2d_offset);
		}//End else
#endif		
	}//End of radians_to_degrees
#pragma endregion

	/// <summary>
	/// Creates a vector of pairs(size_t,size_t) to serve as a list of sections, 1 section for each thread
	/// </summary>
	inline void prepareThreading(std::vector<std::pair<size_t,size_t>>& sections, const size_t arr_size)
	{
		try
		{
			#ifdef _DEBUG
				if (arr_size < 1uz) [[unlikely]]
				{
					throw tpa::exceptions::EmptyArray();
				}//End if
			#endif

			sections.resize(tpa::nThreads);

			size_t StartOff = 0uz;
			size_t EndOff = 0uz;
			size_t i = 0uz;

			for (; i != tpa::nThreads; ++i)
			{
				StartOff = static_cast<size_t>(i * arr_size / tpa::nThreads);
				EndOff = static_cast<size_t>((i + 1uz) * arr_size / tpa::nThreads);
				sections[i] = {StartOff, EndOff};
			}//End for
		}//End try
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(consoleMtx);
			std::cerr << "Exception thrown in tpa::util::prepareThreading: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(consoleMtx);
			std::cerr << "Exception thrown in tpa::util::prepareThreading: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(consoleMtx);
			std::cerr << "Exception thrown in tpa::util::prepareThreading: unknown!\n";
		}//End catch
	}//End of prepareThreading

	/// <summary>
	/// <para>Provides a scope-based timer (stop watch) class for benchmarking purposes</para>
	/// <para>Outputs time taken to console in nanoseconds.</para>
	/// <para>Requires iostream and chrono.</para>
	/// </summary>
	class Timer
	{
	public:
		Timer()
		{
			start = std::chrono::high_resolution_clock::now();
		}//End of constructor

		Timer(Timer const&) = delete;
		Timer& operator=(Timer const&) = delete;
		Timer(Timer&&) = delete;
		Timer& operator=(Timer&&) = delete;

		~Timer()
		{
			end = std::chrono::high_resolution_clock::now();
			std::chrono::high_resolution_clock::duration d = end - start;
			std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(d).count() << "ns\n";
		}//End of destructor

	private:
		std::chrono::high_resolution_clock::time_point start;
		std::chrono::high_resolution_clock::time_point end;
	};//End of class Timer
}//End of namespace

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
	/// Provides a list of valid SIMD-enabled equations
	/// </summary>
	const enum class eqt {
		SUM,
		DIFFERENCE_, //has an underscore in the name because 'DIFFERENCE' is a pre-defined macro in msvc
		PRODUCT,
		QUOTIENT,
		REMAINDER
	};

	/// <summary>
	/// <para>Provides a list of valid floating-point SIMD rounding modes</para>
	/// <para>Please note that some ARM CPUs do not support IEEE-754 rounding modes</para>
	/// </summary>
	const enum class rnd {
#if defined(_M_AMD64)
		NEAREST_INT = _MM_FROUND_TO_NEAREST_INT,//SIMD eqivilant of FE_TONEAREST
		DOWN = _MM_FROUND_TO_NEG_INF,//SIMD eqivilant of FE_DOWNWARD
		UP = _MM_FROUND_TO_POS_INF,//SIMD equivilant of FE_UPWARD
		TRUNCATE_TO_ZERO = _MM_FROUND_TO_ZERO //SIMD eqivilant of FE_TOWARDZERO
#elif defined(_M_ARM64)
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
	/// Provides a list of valid SIMD copy-if predicates.
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
		FACTOR_OF,
		POWER_OF,
		DIVISIBLE_BY
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
	const enum class seq{
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

namespace tpa::util {

#pragma region floating_point_bitwise
	/// <summary>
	/// <para>Performs genuine bit-wise operations on standard floats and doubles.</para>
	/// <para>Requires the SSE2 or NEON instrunction set! If not present at runtime will throw an exception and return 0.</para>
	/// <para>Templated predicate takes 1 of these predicates: tpa::bit</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::bit::AND</para>			
	/// <para>tpa::bit::OR</para>
	/// <para>tpa::bit::XOR</para>
	/// <para>tpa::bit::AND_NOT</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="num1"></param>
	/// <param name="num2"></param>
	/// <returns></returns>
	template<tpa::bit INSTR, typename T>
	[[nodiscard]] inline constexpr T fp_bitwise(T num1, T num2)
	{
		try
		{
#ifdef _M_AMD64
			if constexpr (std::is_same<T, float>())
			{
				//Use SSE for better defined behavior if possible
				if (tpa::has_SSE)
				{
					const __m128 f1 = _mm_set1_ps(num1);
					const __m128 f2 = _mm_set1_ps(num2);
					__m128 result = _mm_setzero_ps();

					if constexpr (INSTR == tpa::bit::AND)
					{
						result = _mm_and_ps(f1, f2);
					}//End if
					else if constexpr (INSTR == tpa::bit::OR)
					{
						result = _mm_or_ps(f1, f2);
					}//End if
					else if constexpr (INSTR == tpa::bit::XOR)
					{
						result = _mm_xor_ps(f1, f2);
					}//End if
					else if constexpr (INSTR == tpa::bit::AND_NOT)
					{
						result = _mm_andnot_ps(f1, f2);
					}//End if
					else
					{
						[] <bool flag = false>()
						{
							static_assert(flag, "INVALID PREDICATED passed in tpa::util::fp_bitwise()");
						}();
					}
#ifdef _WIN32
					return static_cast<T>(result.m128_f32[0]);
#else	
					return static_cast<T>(result[0]);
#endif
				}//End if hasSSE
				else
				{
					static_assert(sizeof(float) == sizeof(int32_t), "Size of float must equal int32_t");

					const int32_t num1_as_int = *reinterpret_cast<int32_t*>(&num1);
					const int32_t num2_as_int = *reinterpret_cast<int32_t*>(&num2);
					int32_t int_res = {};

					if constexpr (INSTR == tpa::bit::AND)
					{
						int_res = num1_as_int & num2_as_int;
					}//End if
					else if constexpr (INSTR == tpa::bit::OR)
					{
						int_res = num1_as_int | num2_as_int;
					}//End if
					else if constexpr (INSTR == tpa::bit::XOR)
					{
						int_res = num1_as_int ^ num2_as_int;
					}//End if
					else if constexpr (INSTR == tpa::bit::AND_NOT)
					{
						int_res = ~num1_as_int & num2_as_int;
					}//End if
					else
					{
						[] <bool flag = false>()
						{
							static_assert(flag, "INVALID PREDICATED passed in tpa::util::fp_bitwise()");
						}();
					}//End else

					return *reinterpret_cast<float*>(&int_res);
				}//End else
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				if (tpa::has_SSE2)
				{
					const __m128d d1 = _mm_set1_pd(num1);
					const __m128d d2 = _mm_set1_pd(num2);
					__m128d result = _mm_setzero_pd();

					if constexpr (INSTR == tpa::bit::AND)
					{
						result = _mm_and_pd(d1, d2);
					}//End if
					else if constexpr (INSTR == tpa::bit::OR)
					{
						result = _mm_or_pd(d1, d2);
					}//End if
					else if constexpr (INSTR == tpa::bit::XOR)
					{
						result = _mm_xor_pd(d1, d2);
					}//End if
					else if constexpr (INSTR == tpa::bit::AND_NOT)
					{
						result = _mm_andnot_pd(d1, d2);
					}//End if
					else
					{
						[] <bool flag = false>()
						{
							static_assert(flag, "INVALID PREDICATED passed in tpa::util::fp_bitwise()");
						}();
					}//End else
#ifdef _WIN32
					return static_cast<T>(result.m128d_f64[0]);
#else	
					return static_cast<T>(result[0]);
#endif
				}//End if has SSE2
				else
				{
					static_assert(sizeof(double) == sizeof(int64_t), "Size of double must equal int64_t");

					const int64_t num1_as_int = *reinterpret_cast<int64_t*>(&num1);
					const int64_t num2_as_int = *reinterpret_cast<int64_t*>(&num2);
					int64_t int_res = {};

					if constexpr (INSTR == tpa::bit::AND)
					{
						int_res = num1_as_int & num2_as_int;
					}//End if
					else if constexpr (INSTR == tpa::bit::OR)
					{
						int_res = num1_as_int | num2_as_int;
					}//End if
					else if constexpr (INSTR == tpa::bit::XOR)
					{
						int_res = num1_as_int ^ num2_as_int;
					}//End if
					else if constexpr (INSTR == tpa::bit::AND_NOT)
					{
						int_res = ~num1_as_int & num2_as_int;
					}//End if
					else
					{
						[] <bool flag = false>()
						{
							static_assert(flag, "INVALID PREDICATED passed in tpa::util::fp_bitwise()");
						}();
					}//End else

					return reinterpret_cast<double*>(&int_res);
				}//End if
			}//End else
			else
			{
				[] <bool flag = false>()
				{
					static_assert(flag, "tpa::util::fp_bitwise() requires float or double.");
				}();
			}//End else
#else
			if constexpr (std::is_same<T, float>())
			{
				static_assert(sizeof(float) == sizeof(int32_t), "Size of float must equal int32_t");

				const int32_t num1_as_int = *reinterpret_cast<int32_t*>(&num1);
				const int32_t num2_as_int = *reinterpret_cast<int32_t*>(&num2);
				int32_t int_res = {};

				if constexpr (INSTR == tpa::bit::AND)
				{
					int_res = num1_as_int & num2_as_int;
				}//End if
				else if constexpr (INSTR == tpa::bit::OR)
				{
					int_res = num1_as_int | num2_as_int;
				}//End if
				else if constexpr (INSTR == tpa::bit::XOR)
				{
					int_res = num1_as_int ^ num2_as_int;
				}//End if
				else if constexpr (INSTR == tpa::bit::AND_NOT)
				{
					int_res = ~num1_as_int & num2_as_int;
				}//End if
				else
				{
					[] <bool flag = false>()
					{
						static_assert(flag, "INVALID PREDICATED passed in tpa::util::fp_bitwise()");
					}();
				}//End else

				return *reinterpret_cast<T*>(&int_res);
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				static_assert(sizeof(double) == sizeof(int64_t), "Size of double must equal int64_t");

				const int64_t num1_as_int = *reinterpret_cast<int64_t*>(&num1);
				const int64_t num2_as_int = *reinterpret_cast<int64_t*>(&num2);
				int64_t int_res = {};

				if constexpr (INSTR == tpa::bit::AND)
				{
					int_res = num1_as_int & num2_as_int;
				}//End if
				else if constexpr (INSTR == tpa::bit::OR)
				{
					int_res = num1_as_int | num2_as_int;
				}//End if
				else if constexpr (INSTR == tpa::bit::XOR)
				{
					int_res = num1_as_int ^ num2_as_int;
				}//End if
				else if constexpr (INSTR == tpa::bit::AND_NOT)
				{
					int_res = ~num1_as_int & num2_as_int;
				}//End if
				else
				{
					[] <bool flag = false>()
					{
						static_assert(flag, "INVALID PREDICATED passed in tpa::util::fp_bitwise()");
					}();
				}//End else

				return *reinterpret_cast<T*>(&int_res);
			}//End if
			else
			{
				[] <bool flag = false>()
				{
					static_assert(flag, "tpa::util::fp_bitwise() requires float or double.");
				}();
			}//End else
#endif
		}//End try
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(consoleMtx);
			std::cerr << "Exception thrown in tpa::util::fp_bitwise: " << ex.what() << "\n";
			return static_cast<T>(0);
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(consoleMtx);
			std::cerr << "Exception thrown in tpa::util::fp_bitwise: " << ex.what() << "\n";
			return static_cast<T>(0);
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(consoleMtx);
			std::cerr << "Exception thrown in tpa::util::fp_bitwise: unknown!\n";
			return static_cast<T>(0);
		}//End catch
	}//End of fp_bitwise

	/// <summary>
	/// <para>Performs genuine bit-wise not (~) on standard floats and doubles.</para>
	/// <para>Requires the SSE2 instrunction set! If not present at runtime will throw an exception and return 0.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name=""></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T fp_bitwise_not(T num)
	{
		return tpa::util::fp_bitwise<tpa::bit::XOR>(num, std::numeric_limits<T>::max());
	}//End of fp_bitwise_not
#pragma endregion

#pragma region misc_avx
#ifdef _M_AMD64

	///<summary>
	///<para> Multiply Packed 64-Bit Integers (Signed and Unsigned) in 'a' by 'b' and returns 'product' using AVX2</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///<para>This function is based on Agner Fog's Vector Class Library: </para>
	/// <see href="https://www.agner.org/optimize/#vectorclass">VCL</see>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_mul_epi64(const __m256i& a, const __m256i& b) noexcept
	{
		const __m256i bswap = _mm256_shuffle_epi32(b, 0xB1);		// swap H<->L
		const __m256i prodlh = _mm256_mullo_epi32(a, bswap);		// 32 bit L*H products
		const __m256i zero = _mm256_setzero_si256();				// 0
		const __m256i prodlh2 = _mm256_hadd_epi32(prodlh, zero);    // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
		const __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2, 0x73);// 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
		const __m256i prodll = _mm256_mul_epu32(a, b);              // a0Lb0L,a1Lb1L, 64 bit unsigned products
		const __m256i prod = _mm256_add_epi64(prodll, prodlh3);     // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
		return  prod;			
	}//End of _mm256_mul_epi64

	///<summary>
	///<para> Multiply Packed 64-Bit Integers (Signed and Unsigned) in 'a' by 'b' and returns 'product' using SSE4.1</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE4.1.</para>
	///<para>This function is based on Agner Fog's Vector Class Library: </para>
	/// <see href="https://www.agner.org/optimize/#vectorclass">VCL</see>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_mul_epi64(const __m128i& a, const __m128i& b) noexcept
	{
		const __m128i bswap = _mm_shuffle_epi32(b, 0xB1);       // b0H,b0L,b1H,b1L (swap H<->L)
		const __m128i prodlh = _mm_mullo_epi32(a, bswap);       // a0Lb0H,a0Hb0L,a1Lb1H,a1Hb1L, 32 bit L*H products
		const __m128i zero = _mm_setzero_si128();                // 0
		const __m128i prodlh2 = _mm_hadd_epi32(prodlh, zero);    // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
		const __m128i prodlh3 = _mm_shuffle_epi32(prodlh2, 0x73);// 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
		const __m128i prodll = _mm_mul_epu32(a, b);             // a0Lb0L,a1Lb1L, 64 bit unsigned products
		const __m128i prod = _mm_add_epi64(prodll, prodlh3);    // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
		return  prod;
	}//End of _mm_mul_epi64

	///<summary>
	///<para> Multiply Packed 32-Bit SIGNED Integers in 'a' by 'b' and returns 'product' using SSE2</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///<para>This function is based on the Intel Developers' Guide</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_mul_epi32(const __m128i& a, const __m128i& b) noexcept
	{
		__m128i tmp1 = _mm_mul_epu32(a, b); /* mul 2,0*/
		__m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4)); /* mul 3,1 */
		return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))); /* shuffle results to [63..0] and pack */
	}//End of _mm_mul_epi32

	///<summary>
	///<para> Computes the absolute value of floats stored in 'x' using SSE</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128</returns>
	[[nodiscard]] inline __m128 _mm_abs_ps(const __m128& x) noexcept
	{
		constexpr __m128 sign_mask = {-0.0f, -0.0f, -0.0f, -0.0f};
		return _mm_andnot_ps(sign_mask, x);
	}//End of _mm_abs_ps

	///<summary>
	///<para> Computes the absolute value of doubles stored in 'x' using SSE2</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128d</returns>
	[[nodiscard]] inline __m128d _mm_abs_pd(const __m128d& x) noexcept
	{
		constexpr __m128d sign_mask = {-0.0, -0.0};
		return _mm_andnot_pd(sign_mask, x);
	}//End of _mm_abs_pd

	///<summary>
	///<para> Computes the absolute value of floats stored in 'x' using AVX</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256</returns>
	[[nodiscard]] inline __m256 _mm256_abs_ps(const __m256& x) noexcept
	{
		constexpr __m256 sign_mask = {-0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f};
		return _mm256_andnot_ps(sign_mask, x);
	}//End of _mm256_abs_ps

	///<summary>
	///<para> Computes the absolute value of doubles stored in 'x' using AVX</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256d</returns>
	[[nodiscard]] inline __m256d _mm256_abs_pd(const __m256d& x) noexcept
	{
		constexpr __m256d sign_mask = {-0.0, -0.0, -0.0, -0.0};
		return _mm256_andnot_pd(sign_mask, x);
	}//End of _mm256_abs_pd

	///<summary>
	/// <para>Converts packed uint64_t in 'x' to packed doubles</para>
	/// <para>Full Range Supported.</para>
	/// <para>Requires SSE4.1 at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128d</returns>
	[[nodiscard]] inline __m128d _mm_cvtepu64_pd(const __m128i& x) noexcept
	{
		__m128i xH = _mm_srli_epi64(x, 32ll);
		xH = _mm_or_si128(xH, _mm_castpd_si128(_mm_set1_pd(19342813113834066795298816.0)));          //  2^84
		__m128i xL = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0xcc);   //  2^52
		__m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(19342813118337666422669312.));     //  2^84 + 2^52
		
		return _mm_add_pd(f, _mm_castsi128_pd(xL));
	}//End of _mm_cvtepu64_pd

	///<summary>
	/// <para>Converts packed uint64_t in 'x' to packed doubles</para>
	/// <para>Full Range Supported.</para>
	/// <para>Requires AVX2 at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256d</returns>
	[[nodiscard]] inline __m256d _mm256_cvtepu64_pd(const __m256i& x) noexcept
	{
		__m256i xH = _mm256_srli_epi64(x, 32ll);
		xH = _mm256_or_si256(xH, _mm256_castpd_si256(_mm256_set1_pd(19342813113834066795298816.0)));          //  2^84
		__m256i xL = _mm256_blend_epi16(x, _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)), 0xcc);   //  2^52
		__m256d f = _mm256_sub_pd(_mm256_castsi256_pd(xH), _mm256_set1_pd(19342813118337666422669312.));     //  2^84 + 2^52

		return _mm256_add_pd(f, _mm256_castsi256_pd(xL));
	}//End of _mm256_cvtepu64_pd
	
	///<summary>
	/// <para>Converts packed int64_t in 'x' to packed doubles</para>
	/// <para>Full Range Supported.</para>
	/// <para>Requires SSE4.1 at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128d</returns>
	[[nodiscard]] inline __m128d _mm_cvtepi64_pd(const __m128i& x) noexcept
	{
		__m128i xH = _mm_srai_epi32(x, 16);
		xH = _mm_blend_epi16(xH, _mm_setzero_si128(), 0x33);
		xH = _mm_add_epi64(xH, _mm_castpd_si128(_mm_set1_pd(442721857769029238784.)));              //  3*2^67
		__m128i xL = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0x88);   //  2^52
		__m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(442726361368656609280.));          //  3*2^67 + 2^52
		
		return _mm_add_pd(f, _mm_castsi128_pd(xL));
	}//End of _mm_cvtepi64_pd

	///<summary>
	/// <para>Converts packed int64_t in 'x' to packed doubles</para>
	/// <para>Full Range Supported.</para>
	/// <para>Requires AVX2 at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256d</returns>
	[[nodiscard]] inline __m256d _mm256_cvtepi64_pd(const __m256i& x) noexcept
	{
		__m256i xH = _mm256_srai_epi32(x, 16);
		xH = _mm256_blend_epi16(xH, _mm256_setzero_si256(), 0x33);
		xH = _mm256_add_epi64(xH, _mm256_castpd_si256(_mm256_set1_pd(442721857769029238784.0)));              //  3*2^67
		__m256i xL = _mm256_blend_epi16(x, _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)), 0x88);   //  2^52
		__m256d f = _mm256_sub_pd(_mm256_castsi256_pd(xH), _mm256_set1_pd(442726361368656609280.0));          //  3*2^67 + 2^52

		return _mm256_add_pd(f, _mm256_castsi256_pd(xL));
	}//End of _mm256_cvtepi64_pd

	///<summary>
	/// <para>Converts packed doubles in 'x' to packed int64_t</para>
	/// <para>Range: [0, 2^51]</para>
	/// <para>Requires SSE2 at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_cvtpd_epu64(__m128d& x) noexcept
	{
		x = _mm_add_pd(x, _mm_set1_pd(0x0010000000000000));
		return _mm_xor_si128(
			_mm_castpd_si128(x),
			_mm_castpd_si128(_mm_set1_pd(0x0010000000000000))
		);
	}//End of _mm_cvtpd_epu64

	///<summary>
	/// <para>Converts packed doubles in 'x' to packed uint64_t</para>
	/// <para>Range: [0, 2^51]</para>
	/// <para>Requires AVX2 at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256d</returns>
	[[nodiscard]] inline __m256i _mm256_cvtpd_epu64(__m256d& x) noexcept
	{
		x = _mm256_add_pd(x, _mm256_set1_pd(0x0010000000000000));
		return _mm256_xor_si256(
			_mm256_castpd_si256(x),
			_mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000))
		);
	}//End of _mm256_cvtpd_epu64

	///<summary>
	/// <para>Converts packed doubles in 'x' to packed uint64_t</para>
	/// <para>Range: [-2^51, 2^51]</para>
	/// <para>Requires SSE2 at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256d</returns>
	[[nodiscard]] inline __m128i _mm_cvtpd_epi64(__m128d& x) noexcept
	{
		x = _mm_add_pd(x, _mm_set1_pd(0x0018000000000000));
		return _mm_sub_epi64(
			_mm_castpd_si128(x),
			_mm_castpd_si128(_mm_set1_pd(0x0018000000000000))
		);
	}//End of _mm_cvtpd_epi64

	///<summary>
	/// <para>Converts packed doubles in 'x' to packed uint64_t</para>
	/// <para>Range: [-2^51, 2^51]</para>
	/// <para>Requires AVX2 at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256d</returns>
	[[nodiscard]] inline __m256i _mm256_cvtpd_epi64(__m256d& x) noexcept
	{
		x = _mm256_add_pd(x, _mm256_set1_pd(0x0018000000000000));
		return _mm256_sub_epi64(
			_mm256_castpd_si256(x),
			_mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000))
		);
	}//End of _mm256_cvtpd_epi64

	///<summary>
	/// <para>Narrow numbers in 'bits' to a certain range.</para>
	/// <para>Much faster replacement for _mm_rem_epi32 / _mm_rem_epu32 for random number range generation</para>
	/// <para>Requires SSE2 instruction set at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="bits"></param>
	/// <param name="range"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_narrow_epi32(const __m128i& bits, const uint32_t range) noexcept
	{
		const __m128i mantissaMask = _mm_set1_epi32(0x7FFFFF);
		const __m128i mantissa = _mm_and_si128(bits, mantissaMask);
		const __m128 one = _mm_set1_ps(1.0f);
		__m128 val = _mm_or_ps(_mm_castsi128_ps(mantissa), one);

		const __m128 rf = _mm_set1_ps((float)range);
		val = _mm_mul_ps(val, rf);
		val = _mm_sub_ps(val, rf);

		return _mm_cvttps_epi32(val);
	}//End of _mm_narrow_epi32

	///<summary>
	/// <para>Narrow numbers in 'bits' to a certain range.</para>
	/// <para>Much faster replacement for _mm_rem_epi64 / _mm_rem_epu64 for random number range generation</para>
	/// <para>Requires SSE2 instruction set at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="bits"></param>
	/// <param name="range"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_narrow_epi64(const __m128i& bits, const uint64_t range) noexcept
	{
		const __m128i mantissaMask = _mm_set1_epi64x(0x7FFFFFFFFFFFFFFF);
		const __m128i mantissa = _mm_and_si128(bits, mantissaMask);
		const __m128d one = _mm_set1_pd(1.0);
		__m128d val = _mm_or_pd(_mm_castsi128_pd(mantissa), one);

		const __m128d rf = _mm_set1_pd((double)range);
		val = _mm_mul_pd(val, rf);
		val = _mm_sub_pd(val, rf);

		return tpa::util::_mm_cvtpd_epi64(val);
	}//End of _mm_narrow_epi64

	///<summary>
	/// <para>Narrow numbers in 'bits' to a certain range.</para>
	/// <para>Much faster replacement for _mm256_rem_epi32 / _mm256_rem_epu32 for random number range generation</para>
	/// <para>Requires AVX2 and FMA instruction sets at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="bits"></param>
	/// <param name="range"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_narrow_epi32(const __m256i& bits, const uint32_t range) noexcept
	{		
		const __m256i mantissaMask = _mm256_set1_epi32(0x7FFFFF);
		const __m256i mantissa = _mm256_and_si256(bits, mantissaMask);
		const __m256 one = _mm256_set1_ps(1.0f);
		__m256 val = _mm256_or_ps(_mm256_castsi256_ps(mantissa), one);

		const __m256 rf = _mm256_set1_ps((float)range);
		val = _mm256_fmsub_ps(val, rf, rf);

		return _mm256_cvttps_epi32(val);
	}//End of _mm256_narrow_epi32

	///<summary>
	/// <para>Narrow numbers in 'bits' to a certain range.</para>
	/// <para>Much faster replacement for _mm256_rem_epi64 / _mm256_rem_epu64 for random number range generation</para>
	/// <para>Requires AVX2 and FMA instruction sets at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="bits"></param>
	/// <param name="range"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_narrow_epi64(const __m256i& bits, const uint64_t range) noexcept
	{
		const __m256i mantissaMask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
		const __m256i mantissa = _mm256_and_si256(bits, mantissaMask);
		const __m256d one = _mm256_set1_pd(1.0);
		__m256d val = _mm256_or_pd(_mm256_castsi256_pd(mantissa), one);

		const __m256d rf = _mm256_set1_pd((double)range);
		val = _mm256_fmsub_pd(val, rf, rf);

		return tpa::util::_mm256_cvtpd_epi64(val);
	}//End of _mm256_narrow_epi64

	///<summary>
	/// <para>Narrow numbers in 'bits' to a certain range.</para>
	/// <para>Much faster replacement for _mm512_rem_epi32 / _mm512_rem_epu32 for random number range generation</para>
	/// <para>Requires AVX-512 (foundation) instruction set at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="bits"></param>
	/// <param name="range"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_narrow_epi32(const __m512i& bits, const uint32_t range) noexcept
	{
		const __m512i mantissaMask = _mm512_set1_epi64(0x7FFFFF);
		const __m512i mantissa = _mm512_and_si512(bits, mantissaMask);
		const __m512 one = _mm512_set1_ps(1.0f);
		__m512 val = _mm512_or_ps(_mm512_castsi512_ps(mantissa), one);

		const __m512 rf = _mm512_set1_ps((float)range);
		val = _mm512_fmsub_ps(val, rf, rf);

		return _mm512_cvttps_epi32(val);
	}//End of _mm512_narrow_epi32

	///<summary>
	/// <para>Narrow numbers in 'bits' to a certain range.</para>
	/// <para>Much faster replacement for _mm512_rem_epi64 / _mm512_rem_epu64 for random number range generation</para>
	/// <para>Requires AVX-512 (foundation) and AVX-512 (Double Word Quad Word) instruction set at runtime.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	///</summary>
	/// <param name="bits"></param>
	/// <param name="range"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_narrow_epi64(const __m512i& bits, const uint64_t range) noexcept
	{
		const __m512i mantissaMask = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
		const __m512i mantissa = _mm512_and_si512(bits, mantissaMask);
		const __m512d one = _mm512_set1_pd(1.0);
		__m512d val = _mm512_or_pd(_mm512_castsi512_pd(mantissa), one);

		const __m512d rf = _mm512_set1_pd((double)range);
		val = _mm512_fmsub_pd(val, rf, rf);

		return _mm512_cvttpd_epi64(val);
	}//End of _mm512_narrow_epi64

	/// <summary>
	/// <para>Sums the values stored in an __m128i vector of int32_t</para>
	/// <para>Requires SSE2 at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="x"></param>
	/// <returns></returns>
	inline uint32_t _mm_sum_epi32(__m128i& x)
	{
		__m128i hi64 = _mm_unpackhi_epi64(x, x);           
		__m128i sum64 = _mm_add_epi32(hi64, x);
		__m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
		__m128i sum32 = _mm_add_epi32(sum64, hi32);
		return _mm_cvtsi128_si32(sum32);
	}//End of _mm_sum_epi32
	
	/// <summary>
	/// <para>Sums the values stored in an __m256i vector of int32_t</para>
	/// <para>Requires AVX2 at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline uint32_t _mm256_sum_epi32(__m256i& v)
	{
		__m128i sum128 = _mm_add_epi32(
			_mm256_castsi256_si128(v),
			_mm256_extracti128_si256(v, 1));
		return _mm_sum_epi32(sum128);
	}//End of _mm256_sum_epi32

	/// <summary>
	/// <para>Sums the values stored in an __m512i vector of int32_t</para>
	/// <para>Requires AVX512 at runtime </para>
	/// <para>AVX512's _mm512_reduce_add_epi32 may be faster in some cases.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline uint32_t _mm512_sum_epi32(__m512i& v)
	{
		__m256i sum256 = _mm256_add_epi32(
			_mm512_castsi512_si256(v),
			_mm512_extracti64x4_epi64(v, 1));
		return _mm256_sum_epi32(sum256);
	}//End of _mm512_sum_epi32

	/// <summary>
	/// <para>Sums the values stored in an __m128i vector of int64_t</para>
	/// <para>Requires SSE2 at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="x"></param>
	/// <returns></returns>
	inline uint64_t _mm_sum_epi64(__m128i& x)
	{
		__m128i hi64 = _mm_unpackhi_epi64(x, x);
		__m128i sum64 = _mm_add_epi64(hi64, x);
		return _mm_cvtsi128_si64(sum64);
	}//End of _mm_sum_epi64

	/// <summary>
	/// <para>Sums the values stored in an __m256i vector of int64_t</para>
	/// <para>Requires AVX2 at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline uint64_t _mm256_sum_epi64(__m256i& v)
	{
		__m128i sum128 = _mm_add_epi64(
			_mm256_castsi256_si128(v),
			_mm256_extracti128_si256(v, 1));
		return _mm_sum_epi64(sum128);
	}//End of _mm256_sum_epi64

	/// <summary>
	/// <para>Sums the values stored in an __m512i vector of int64_t</para>
	/// <para>Requires AVX512 at runtime </para>
	/// <para>AVX512's _mm512_reduce_add_epi64 may be faster in some cases.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline uint64_t _mm512_sum_epi64(__m512i& v)
	{
		__m256i sum256 = _mm256_add_epi64(
			_mm512_castsi512_si256(v),
			_mm512_extracti64x4_epi64(v, 1));
		return _mm256_sum_epi64(sum256);
	}//End of _mm512_sum_epi64

	/// <summary>
	/// <para>Sums the values stored in an __m128 vector of floats</para>
	/// <para>Requires SSE1 at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="x"></param>
	/// <returns></returns>
	inline float _mm_sum_ps(__m128& x)
	{
		__m128 shuff = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
		__m128 sums = _mm_add_ps(x, shuff);
		shuff = _mm_movehl_ps(shuff, sums);
		sums = _mm_add_ss(sums, shuff);
		return _mm_cvtss_f32(sums);
	}//End of _mm_sum_ps

	/// <summary>
	/// <para>Sums the values stored in an __m256 vector of float</para>
	/// <para>Requires AVX at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline float _mm256_sum_ps(__m256& x)
	{
		__m128 vlow = _mm256_castps256_ps128(x);
		__m128 vhigh = _mm256_extractf128_ps(x, 1);
		__m128 v128 = _mm_add_ps(vlow, vhigh);
		return _mm_sum_ps(v128);
	}//End of _mm256_sum_ps

	/// <summary>
	/// <para>Sums the values stored in an __m512 vector of floats</para>
	/// <para>Requires AVX512 Foundation at runtime </para>
	/// <para>AVX512's _mm512_reduce_add_ps may be faster in some cases.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline float _mm512_sum_ps(__m512& x)
	{
		__m256 vlow = _mm512_castps512_ps256(x);
		__m256 vhigh = _mm512_extractf32x8_ps(x, 1);
		__m256 v256 = _mm256_add_ps(vlow, vhigh);
		return _mm256_sum_ps(v256);
	}//End of _mm512_sum_ps

	/// <summary>
	/// <para>Sums the values stored in an __m128d vector of doubles</para>
	/// <para>Requires SSE2 at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="x"></param>
	/// <returns></returns>
	inline double _mm_sum_pd(__m128d& x)
	{
		__m128 undef = _mm_undefined_ps();                       
		__m128 shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(x));
		__m128d shuf = _mm_castps_pd(shuftmp);
		return  _mm_cvtsd_f64(_mm_add_sd(x, shuf));
	}//End of _mm_sum_pd

	/// <summary>
	/// <para>Sums the values stored in an __m256d vector of doubles</para>
	/// <para>Requires AVX at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline double _mm256_sum_pd(__m256d& x)
	{
		__m128d vlow = _mm256_castpd256_pd128(x);
		__m128d vhigh = _mm256_extractf128_pd(x, 1);
		__m128d v128 = _mm_add_pd(vlow, vhigh);
		return _mm_sum_pd(v128);
	}//End of _mm256_sum_pd

	/// <summary>
	/// <para>Sums the values stored in an __m512d vector of doubles</para>
	/// <para>Requires AVX512 Foundation at runtime </para>
	/// <para>AVX512's _mm512_reduce_add_pd may be faster in some cases.</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline double _mm512_sum_pd(__m512d& x)
	{
		__m256d vlow = _mm512_castpd512_pd256(x);
		__m256d vhigh = _mm512_extractf64x4_pd(x, 1);
		__m256d v256 = _mm256_add_pd(vlow, vhigh);
		return _mm256_sum_pd(v256);
	}//End of _mm512_sum_pd

	/// <summary>
	/// <para>Sums the values stored in an __m128i vector of int8_t</para>
	/// <para>Requires SSE2 at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="x"></param>
	/// <returns></returns>
	inline uint32_t _mm_sum_epi8(__m128i& x)
	{
		__m128i v = _mm_sad_epu8(x, _mm_setzero_si128());
		return _mm_cvtsi128_si32(v) + _mm_extract_epi16(v, 4);
	}//End of _mm_sum_epi8

	/// <summary>
	/// <para>Sums the values stored in an __m256i vector of int8_t</para>
	/// <para>Requires AVX2 at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline uint32_t _mm256_sum_epi8(__m256i& x)
	{
		__m128i sum128 = _mm_add_epi32(
			_mm256_castsi256_si128(x),
			_mm256_extracti128_si256(x, 1));
		return _mm_sum_epi8(sum128);
	}//End of _mm256_sum_epi8

	/// <summary>
	/// <para>Sums the values stored in an __m512i vector of int8_t</para>
	/// <para>Requires AVX512 Byte and Word at runtime </para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline uint32_t _mm512_sum_epi8(__m512i& v)
	{
		__m256i sum256 = _mm256_add_epi32(
			_mm512_castsi512_si256(v),
			_mm512_extracti64x4_epi64(v, 1));
		return _mm256_sum_epi8(sum256);
	}//End of _mm512_sum_epi8

	/// <summary>
	/// <para>Sums the values stored in an __m128i vector of int16_t</para>
	/// <para>Requires SSE2 at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="x"></param>
	/// <returns></returns>
	inline uint32_t _mm_sum_epi16(__m128i& x)
	{
		__m128i _temp = _mm_madd_epi16(x, _mm_set1_epi16(1));
		return tpa::util::_mm_sum_epi32(_temp);
	}//End of _mm_sum_epi16

	/// <summary>
	/// <para>Sums the values stored in an __m256i vector of int16_t</para>
	/// <para>Requires AVX2 at runtime</para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline uint32_t _mm256_sum_epi16(__m256i& x)
	{
		__m256i _temp = _mm256_madd_epi16(x, _mm256_set1_epi16(1));
		return tpa::util::_mm256_sum_epi32(_temp);
	}//End of _mm256_sum_epi16

	/// <summary>
	/// <para>Sums the values stored in an __m512i vector of int16_t</para>
	/// <para>Requires AVX512 Byte & Word at runtime </para>
	/// <para>Note: This is a function which is part of TPA and is not an instruction or intrinsic.</para>
	/// </summary>
	/// <param name="v"></param>
	/// <returns></returns>
	inline uint32_t _mm512_sum_epi16(__m512i& x)
	{
		__m512i _temp = _mm512_madd_epi16(x, _mm512_set1_epi16(1));
		return tpa::util::_mm512_sum_epi32(_temp);
	}//End of _mm512_sum_epi16
#endif
#pragma endregion
}//End of namespace



