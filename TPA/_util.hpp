#pragma once
/*
* Truely Parallel Algorithms Library - Utility Functions
* By: David Aaron Braun
* 2022-07-08
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
#include <array>

#include "ThreadPool.hpp"
#include "excepts.hpp"
#include "size_t_lit.hpp"
#include "tpa_macros.hpp"
#include "predicates.hpp"
#include "tpa_concepts.hpp"
#include "InstructionSet.hpp"

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
#ifdef TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
		return (a < b) ? a : b;
#else
		return (a * (a < b) + b * (b <= a));
#endif
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
#ifdef TPA_ARCHITECTURE_PROBABLY_HAS_CMOV
		return (a > b) ? a : b;
#else
		return ((a > b) * a + (a <= b) * b);
#endif
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
	/// <para>Returns true if 'n' is a Prime Number</para>
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
	/// <para>Returns true if 'x' is a Perfect Square</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr bool isPerfectSquare(const T x) noexcept
	{
		T s = static_cast<T>(std::sqrt(x));
		return (s * s == x);
	}//End of isPerfectSquare

	/// <summary>
	/// <para>Returns true if 'x' is a Fibonacci Number</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr bool isFibonacci(const T x) noexcept
	{
		T mult = ((5 * x) * x);
		return tpa::util::isPerfectSquare(mult + 4) ||
			tpa::util::isPerfectSquare(mult - 4);
	}//End of isFibonacci 

	/// <summary>
	/// <para>An std::array<uint64_t> containing the first 7 numbers of the Sylvester Sequence.</para>
	/// </summary>
	static constexpr std::array<std::uint64_t, 7uz> sylvester_seq = {2ull,3ull,7ull,43ull,1807ull,3263443ull,10650056950807ull};

	/// <summary>
	/// <para>Returns true if 'x' is a member of the Sylvester 
	/// <para>Only works on the first 7 numbers of the sequence as the 8th number and higher in the Sylvester would require a type larger than uint64_t to represtent them..</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr bool isSylvester(const T x) noexcept
	{
		const uint64_t xx = static_cast<uint64_t>(x);

		if (xx == sylvester_seq[0] ||  
			xx == sylvester_seq[1] ||
			xx == sylvester_seq[2] ||
			xx == sylvester_seq[3] ||
			xx == sylvester_seq[4] ||
			xx == sylvester_seq[5] ||
			xx == sylvester_seq[6] ||
			xx == sylvester_seq[7])
		{
			return true;
		}//End if
		else
		{
			return false;
		}//End else		
	}//End of isSylvester

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
	/// <para>Returns true if 'p' is a power of 'n'</para>
	/// </summary>
	/// <typeparam name="N"></typeparam>
	/// <typeparam name="POW"></typeparam>
	/// <param name="n"></param>
	/// <param name="p"></param>
	/// <returns></returns>
	template<typename N, typename POW>
	inline constexpr bool isPower(const N n, const POW p) noexcept
	{
		double x = std::log(static_cast<double>(n)) / std::log(static_cast<double>(p));

		return (x - std::trunc(x)) < 0.000001;
	}//End of isPower

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
#ifdef TPA_X86_64
			if (inaccurateOptimization && tpa::has_SSE)
			{
				__m128 _num = _mm_set1_ps(num);

				_num = _mm_sqrt_ps(_num);

#ifdef _MSC_VER
				return static_cast<T>(_num.m128_f32[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(std::sqrt(num));
#ifdef TPA_X86_64
			}//End if
#endif
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
#ifdef TPA_X86_64
			if (inaccurateOptimization && tpa::has_SSE2)
			{
				__m128d _num = _mm_set1_pd(num);

				_num = _mm_sqrt_pd(_num);

#ifdef _MSC_VER
				return static_cast<T>(_num.m128d_f64[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(std::sqrt(num));
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
			if (inaccurateOptimization && tpa::has_SSE)
			{				
				__m128 _num = _mm_set1_ps(num);

				_num = _mm_invsqrt_ps(_num);

#ifdef _MSC_VER
				return static_cast<T>(_num.m128_f32[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(1.0f / std::sqrt(num));
#ifdef TPA_X86_64
			}//End if
#endif
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
#ifdef TPA_X86_64
			if (inaccurateOptimization && tpa::has_SSE2)
			{
				__m128d _num = _mm_set1_pd(num);

				_num = _mm_invsqrt_pd(_num);

#ifdef _MSC_VER
				return static_cast<T>(_num.m128d_f64[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(1.0 / std::sqrt(num));
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
			if (inaccurateOptimization && tpa::has_SSE)
			{
				__m128 _num = _mm_set1_ps(num);

				_num = _mm_cbrt_ps(_num);

#ifdef _MSC_VER
				return static_cast<T>(_num.m128_f32[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(std::cbrt(num));
#ifdef TPA_X86_64
			}//End if
#endif
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
#ifdef TPA_X86_64
			if (inaccurateOptimization && tpa::has_SSE2)
			{
				__m128d _num = _mm_set1_pd(num);

				_num = _mm_cbrt_pd(_num);

#ifdef _MSC_VER
				return static_cast<T>(_num.m128d_f64[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(std::cbrt(num));
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
			if (inaccurateOptimization && tpa::has_SSE)
			{
				__m128 _num = _mm_set1_ps(num);

				_num = _mm_invcbrt_ps(_num);

#ifdef _MSC_VER
				return static_cast<T>(_num.m128_f32[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(1.0f / std::cbrt(num));
#ifdef TPA_X86_64
			}//End if
#endif
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
#ifdef TPA_X86_64
			if (inaccurateOptimization && tpa::has_SSE2)
			{
				__m128d _num = _mm_set1_pd(num);

				_num = _mm_invcbrt_pd(_num);

#ifdef _MSC_VER
				return static_cast<T>(_num.m128d_f64[0]);
#else	
				return static_cast<T>(_num[0]);
#endif
			}//End if
			else
			{
#endif
				return static_cast<T>(1.0 / std::cbrt(num));
#ifdef TPA_X86_64
			}//End if
#endif
		}//End if		
		else
		{
			return static_cast<T>(1.0 / std::cbrt(num));
		}//End else
	}//End of icbrt
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
}//End of namespace
