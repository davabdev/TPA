#pragma once
/*
* Truely Parallel Algorithms Library - SIMD Utility Functions
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

#include "../_util.hpp"
#include "../ThreadPool.hpp"
#include "../excepts.hpp"
#include "../size_t_lit.hpp"
#include "../tpa_macros.hpp"
#include "../predicates.hpp"
#include "../InstructionSet.hpp"

/// <summary>
/// TPA SIMD Utility Functions
/// </summary>
namespace tpa::simd {
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

#ifdef TPA_X86_64	

		inline constexpr __m512 avx512_f_r2d_offset = { f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset };

		inline constexpr __m512d avx512_d_r2d_offset = { r2d_offset, r2d_offset, r2d_offset, r2d_offset, r2d_offset, r2d_offset, r2d_offset, r2d_offset };

		inline constexpr __m256 avx256_f_r2d_offset = { f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset };

		inline constexpr __m256d avx256_d_r2d_offset = { r2d_offset, r2d_offset, r2d_offset, r2d_offset };

		inline constexpr __m128 sse_f_r2d_offset = { f_r2d_offset, f_r2d_offset, f_r2d_offset, f_r2d_offset };

		inline constexpr __m128d sse2_d_r2d_offset = { r2d_offset, r2d_offset };

		inline constexpr __m512 avx512_f_d2r_offset = { f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset, f_d2r_offset };

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
#ifdef TPA_X86_64
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
		else if constexpr (std::is_same<T, float>())
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef _MSC_VER
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
#ifdef _MSC_VER
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
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::util::fp_bitwise: " << ex.what() << "\n";
			return static_cast<T>(0);
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::util::fp_bitwise: " << ex.what() << "\n";
			return static_cast<T>(0);
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
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
		return tpa::simd::fp_bitwise<tpa::bit::XOR>(num, std::numeric_limits<T>::max());
	}//End of fp_bitwise_not
#pragma endregion

#pragma region misc_avx
#ifdef TPA_X86_64	

	///<summary>
	///<para> Bitwise Not of __m128i Vector using SSE2</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_not_si128(const __m128i& x) noexcept
	{
		return _mm_xor_si128(x, _mm_set1_epi64x(-1LL));
	}//End of _mm_not_si128

	///<summary>
	///<para> Bitwise Not of __m256i Vector using AVX2</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_not_si256(const __m256i& x) noexcept
	{
		return _mm256_xor_si256(x, _mm256_set1_epi64x(-1LL));
	}//End of _mm256_not_si256

	///<summary>
	///<para> Bitwise Not of __m512i Vector using AVX-512</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX-512.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_not_si512(const __m512i& x) noexcept
	{
		return _mm512_xor_si512(x, _mm512_set1_epi64(-1LL));
	}//End of _mm512_not_si512

	///<summary>
	///<para>Extract highest set 1 bits of 16 bit ints in an __m128i</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_exthsb_epi16(const __m128i& x) noexcept
	{
		const __m128i _one = _mm_set1_epi16(static_cast<int16_t>(1));

		__m128i _ret = _mm_setzero_si128();
		__m128i _temp = _mm_setzero_si128();
		__m128i _temp2 = _mm_setzero_si128();

		_ret = _mm_or_si128(x, _mm_srli_epi16(x, 1));
		_ret = _mm_or_si128(_ret, _mm_srli_epi16(_ret, 2));
		_ret = _mm_or_si128(_ret, _mm_srli_epi16(_ret, 4));
		_ret = _mm_or_si128(_ret, _mm_srli_epi16(_ret, 8));

		_temp = _mm_add_epi16(_ret, _one);
		_temp = _mm_srli_epi16(_temp, 1);

		_temp2 = _mm_slli_epi16(_one, 15);
		_temp2 = _mm_and_si128(_ret, _temp2);

		_ret = _mm_or_si128(_temp, _temp2);

		return _ret;
	}//End of _mm_exthsb_epi16

	///<summary>
	///<para>Extract highest set 1 bits of 16 bit ints in an __m256i</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_exthsb_epi16(const __m256i& x) noexcept
	{		
		const __m256i _one = _mm256_set1_epi16(static_cast<int16_t>(1));

		__m256i _ret = _mm256_setzero_si256();
		__m256i _temp = _mm256_setzero_si256();
		__m256i _temp2 = _mm256_setzero_si256();
		
		_ret = _mm256_or_si256(x, _mm256_srli_epi16(x, 1));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi16(_ret, 2));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi16(_ret, 4));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi16(_ret, 8));

		_temp = _mm256_add_epi16(_ret, _one);
		_temp = _mm256_srli_epi16(_temp, 1);

		_temp2 = _mm256_slli_epi16(_one, 15);
		_temp2 = _mm256_and_si256(_ret, _temp2);

		_ret = _mm256_or_si256(_temp, _temp2);

		return _ret;
	}//End of _mm256_exthsb_epi16

	///<summary>
	///<para>Extract highest set 1 bits of 16 bit ints in an __m512i</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX512F and AVX512BW.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_exthsb_epi16(const __m512i& x) noexcept
	{
		const __m512i _one = _mm512_set1_epi16(static_cast<int16_t>(1));

		__m512i _ret = _mm512_setzero_si512();
		__m512i _temp = _mm512_setzero_si512();
		__m512i _temp2 = _mm512_setzero_si512();

		_ret = _mm512_or_si512(x, _mm512_srli_epi16(x, 1));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi16(_ret, 2));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi16(_ret, 4));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi16(_ret, 8));

		_temp = _mm512_add_epi16(_ret, _one);
		_temp = _mm512_srli_epi16(_temp, 1);

		_temp2 = _mm512_slli_epi16(_one, 15);
		_temp2 = _mm512_and_si512(_ret, _temp2);

		_ret = _mm512_or_si512(_temp, _temp2);

		return _ret;
	}//End of _mm512_exthsb_epi16

	///<summary>
	///<para>Extract highest set 1 bits of 32 bit ints in an __m128i</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_exthsb_epi32(const __m128i& x) noexcept
	{
		const __m128i _one = _mm_set1_epi32(1);

		__m128i _ret = _mm_setzero_si128();
		__m128i _temp = _mm_setzero_si128();
		__m128i _temp2 = _mm_setzero_si128();

		_ret = _mm_or_si128(x, _mm_srli_epi32(x, 1));
		_ret = _mm_or_si128(_ret, _mm_srli_epi32(_ret, 2));
		_ret = _mm_or_si128(_ret, _mm_srli_epi32(_ret, 4));
		_ret = _mm_or_si128(_ret, _mm_srli_epi32(_ret, 8));
		_ret = _mm_or_si128(_ret, _mm_srli_epi32(_ret, 16));

		_temp = _mm_add_epi32(_ret, _one);
		_temp = _mm_srli_epi32(_temp, 1);

		_temp2 = _mm_slli_epi32(_one, 15);
		_temp2 = _mm_and_si128(_ret, _temp2);

		_ret = _mm_or_si128(_temp, _temp2);

		return _ret;
	}//End of _mm_exthsb_epi32

	///<summary>
	///<para>Extract highest set 1 bits of 32 bit ints in an __m256i</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_exthsb_epi32(const __m256i& x) noexcept
	{
		const __m256i _one = _mm256_set1_epi16(1);

		__m256i _ret = _mm256_setzero_si256();
		__m256i _temp = _mm256_setzero_si256();
		__m256i _temp2 = _mm256_setzero_si256();

		_ret = _mm256_or_si256(x, _mm256_srli_epi32(x, 1));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi32(_ret, 2));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi32(_ret, 4));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi32(_ret, 8));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi32(_ret, 16));

		_temp = _mm256_add_epi32(_ret, _one);
		_temp = _mm256_srli_epi32(_temp, 1);

		_temp2 = _mm256_slli_epi32(_one, 15);
		_temp2 = _mm256_and_si256(_ret, _temp2);

		_ret = _mm256_or_si256(_temp, _temp2);

		return _ret;
	}//End of _mm256_exthsb_epi32

	///<summary>
	///<para>Extract highest set 1 bits of 32 bit ints in an __m512i</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX512F.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_exthsb_epi32(const __m512i& x) noexcept
	{
		const __m512i _one = _mm512_set1_epi32(1);

		__m512i _ret = _mm512_setzero_si512();
		__m512i _temp = _mm512_setzero_si512();
		__m512i _temp2 = _mm512_setzero_si512();

		_ret = _mm512_or_si512(x, _mm512_srli_epi32(x, 1));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi32(_ret, 2));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi32(_ret, 4));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi32(_ret, 8));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi32(_ret, 16));

		_temp = _mm512_add_epi32(_ret, _one);
		_temp = _mm512_srli_epi32(_temp, 1);

		_temp2 = _mm512_slli_epi32(_one, 15);
		_temp2 = _mm512_and_si512(_ret, _temp2);

		_ret = _mm512_or_si512(_temp, _temp2);

		return _ret;
	}//End of _mm512_exthsb_epi32

	///<summary>
	///<para>Extract highest set 1 bits of 64 bit ints in an __m128i</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_exthsb_epi64(const __m128i& x) noexcept
	{
		const __m128i _one = _mm_set1_epi64x(1ll);

		__m128i _ret = _mm_setzero_si128();
		__m128i _temp = _mm_setzero_si128();
		__m128i _temp2 = _mm_setzero_si128();

		_ret = _mm_or_si128(x, _mm_srli_epi64(x, 1));
		_ret = _mm_or_si128(_ret, _mm_srli_epi64(_ret, 2));
		_ret = _mm_or_si128(_ret, _mm_srli_epi64(_ret, 4));
		_ret = _mm_or_si128(_ret, _mm_srli_epi64(_ret, 8));
		_ret = _mm_or_si128(_ret, _mm_srli_epi64(_ret, 16));
		_ret = _mm_or_si128(_ret, _mm_srli_epi64(_ret, 32));

		_temp = _mm_add_epi64(_ret, _one);
		_temp = _mm_srli_epi64(_temp, 1);

		_temp2 = _mm_slli_epi64(_one, 15);
		_temp2 = _mm_and_si128(_ret, _temp2);

		_ret = _mm_or_si128(_temp, _temp2);

		return _ret;
	}//End of _mm_exthsb_epi64

	///<summary>
	///<para>Extract highest set 1 bits of 64 bit ints in an __m256i</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_exthsb_epi64(const __m256i& x) noexcept
	{
		const __m256i _one = _mm256_set1_epi64x(1ll);

		__m256i _ret = _mm256_setzero_si256();
		__m256i _temp = _mm256_setzero_si256();
		__m256i _temp2 = _mm256_setzero_si256();

		_ret = _mm256_or_si256(x, _mm256_srli_epi64(x, 1));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi64(_ret, 2));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi64(_ret, 4));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi64(_ret, 8));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi64(_ret, 16));
		_ret = _mm256_or_si256(_ret, _mm256_srli_epi64(_ret, 32));

		_temp = _mm256_add_epi64(_ret, _one);
		_temp = _mm256_srli_epi64(_temp, 1);

		_temp2 = _mm256_slli_epi64(_one, 15);
		_temp2 = _mm256_and_si256(_ret, _temp2);

		_ret = _mm256_or_si256(_temp, _temp2);

		return _ret;
	}//End of _mm256_exthsb_epi64

	///<summary>
	///<para>Extract highest set 1 bits of 64 bit ints in an __m512i</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX512F.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_exthsb_epi64(const __m512i& x) noexcept
	{
		const __m512i _one = _mm512_set1_epi64(1ll);

		__m512i _ret = _mm512_setzero_si512();
		__m512i _temp = _mm512_setzero_si512();
		__m512i _temp2 = _mm512_setzero_si512();

		_ret = _mm512_or_si512(x, _mm512_srli_epi64(x, 1u));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi64(_ret, 2u));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi64(_ret, 4u));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi64(_ret, 8u));
		_ret = _mm512_or_si512(_ret, _mm512_srli_epi64(_ret, 16u));

		_temp = _mm512_add_epi64(_ret, _one);
		_temp = _mm512_srli_epi64(_temp, 1u);

		_temp2 = _mm512_slli_epi64(_one, 15u);
		_temp2 = _mm512_and_si512(_ret, _temp2);

		_ret = _mm512_or_si512(_temp, _temp2);

		return _ret;
	}//End of _mm512_exthsb_epi64 

	///<summary>
	///<para> Set leading zeros of 16-bit integers in an __m128i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_lzset_epi16(const __m128i& x) noexcept
	{
		__m128i _mask = _mm_or_si128(x, _mm_srli_epi16(x, 1));

		_mask = _mm_or_si128(_mask, _mm_srli_epi16(_mask, 2));
		_mask = _mm_or_si128(_mask, _mm_srli_epi16(_mask, 4));
		_mask = _mm_or_si128(_mask, _mm_srli_epi16(_mask, 8));

		return _mm_or_si128(x, _mm_not_si128(_mask));
	}//End of _mm_lzset_epi16

	///<summary>
	///<para> Set leading zeros of 32-bit integers in an __m128i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_lzset_epi32(const __m128i& x) noexcept
	{
		__m128i _mask = _mm_or_si128(x, _mm_srli_epi32(x, 1));

		_mask = _mm_or_si128(_mask, _mm_srli_epi32(_mask, 2));
		_mask = _mm_or_si128(_mask, _mm_srli_epi32(_mask, 4));
		_mask = _mm_or_si128(_mask, _mm_srli_epi32(_mask, 8));
		_mask = _mm_or_si128(_mask, _mm_srli_epi32(_mask, 16));

		return _mm_or_si128(x, _mm_not_si128(_mask));
	}//End of _mm_lzset_epi32

	///<summary>
	///<para> Set leading zeros of 64-bit integers in an __m128i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_lzset_epi64(const __m128i& x) noexcept
	{
		__m128i _mask = _mm_or_si128(x, _mm_srli_epi64(x, 1));

		_mask = _mm_or_si128(_mask, _mm_srli_epi64(_mask, 2));
		_mask = _mm_or_si128(_mask, _mm_srli_epi64(_mask, 4));
		_mask = _mm_or_si128(_mask, _mm_srli_epi64(_mask, 8));
		_mask = _mm_or_si128(_mask, _mm_srli_epi64(_mask, 16));
		_mask = _mm_or_si128(_mask, _mm_srli_epi64(_mask, 32));

		return _mm_or_si128(x, _mm_not_si128(_mask));
	}//End of _mm_lzset_epi64

	///<summary>
	///<para> Set leading zeros of 16-bit integers in an __m256i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_lzset_epi16(const __m256i& x) noexcept
	{
		__m256i _mask = _mm256_or_si256(x, _mm256_srli_epi16(x, 1));

		_mask = _mm256_or_si256(_mask, _mm256_srli_epi16(_mask, 2));
		_mask = _mm256_or_si256(_mask, _mm256_srli_epi16(_mask, 4));
		_mask = _mm256_or_si256(_mask, _mm256_srli_epi16(_mask, 8));

		return _mm256_or_si256(x, _mm256_not_si256(_mask));
	}//End of _mm256_lzset_epi16

	///<summary>
	///<para> Set leading zeros of 32-bit integers in an __m256i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_lzset_epi32(const __m256i& x) noexcept
	{
		__m256i _mask = _mm256_or_si256(x, _mm256_srli_epi32(x, 1));

		_mask = _mm256_or_si256(_mask, _mm256_srli_epi32(_mask, 2));
		_mask = _mm256_or_si256(_mask, _mm256_srli_epi32(_mask, 4));
		_mask = _mm256_or_si256(_mask, _mm256_srli_epi32(_mask, 8));
		_mask = _mm256_or_si256(_mask, _mm256_srli_epi32(_mask, 16));

		return _mm256_or_si256(x, _mm256_not_si256(_mask));
	}//End of _mm256_lzset_epi32

	///<summary>
	///<para> Set leading zeros of 64-bit integers in an __m128i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_lzset_epi64(const __m256i& x) noexcept
	{
		__m256i _mask = _mm256_or_si256(x, _mm256_srli_epi64(x, 1));

		_mask = _mm256_or_si256 (_mask, _mm256_srli_epi64(_mask, 2));
		_mask = _mm256_or_si256 (_mask, _mm256_srli_epi64(_mask, 4));
		_mask = _mm256_or_si256 (_mask, _mm256_srli_epi64(_mask, 8));
		_mask = _mm256_or_si256 (_mask, _mm256_srli_epi64(_mask, 16));
		_mask = _mm256_or_si256 (_mask, _mm256_srli_epi64(_mask, 32));

		return _mm256_or_si256(x, _mm256_not_si256(_mask));
	}//End of _mm256_lzset_epi64

	///<summary>
	///<para> Set leading zeros of 16-bit integers in an __m512i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX-512.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_lzset_epi16(const __m512i& x) noexcept
	{
		__m512i _mask = _mm512_or_si512(x, _mm512_srli_epi16(x, 1));

		_mask = _mm512_or_si512(_mask, _mm512_srli_epi16(_mask, 2));
		_mask = _mm512_or_si512(_mask, _mm512_srli_epi16(_mask, 4));
		_mask = _mm512_or_si512(_mask, _mm512_srli_epi16(_mask, 8));

		return _mm512_or_si512(x, _mm512_not_si512(_mask));
	}//End of _mm512_lzset_epi16

	///<summary>
	///<para> Set leading zeros of 32-bit integers in an __m512i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX-512.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_lzset_epi32(const __m512i& x) noexcept
	{
		__m512i _mask = _mm512_or_si512(x, _mm512_srli_epi32(x, 1));

		_mask = _mm512_or_si512(_mask, _mm512_srli_epi32(_mask, 2));
		_mask = _mm512_or_si512(_mask, _mm512_srli_epi32(_mask, 4));
		_mask = _mm512_or_si512(_mask, _mm512_srli_epi32(_mask, 8));
		_mask = _mm512_or_si512(_mask, _mm512_srli_epi32(_mask, 16));

		return _mm512_or_si512(x, _mm512_not_si512(_mask));
	}//End of _mm512_lzset_epi32

	///<summary>
	///<para> Set leading zeros of 64-bit integers in an __m512i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX-512.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_lzset_epi64(const __m512i& x) noexcept
	{
		__m512i _mask = _mm512_or_si512(x, _mm512_srli_epi64(x, 1));

		_mask = _mm512_or_si512(_mask, _mm512_srli_epi64(_mask, 2));
		_mask = _mm512_or_si512(_mask, _mm512_srli_epi64(_mask, 4));
		_mask = _mm512_or_si512(_mask, _mm512_srli_epi64(_mask, 8));
		_mask = _mm512_or_si512(_mask, _mm512_srli_epi64(_mask, 16));
		_mask = _mm512_or_si512(_mask, _mm512_srli_epi64(_mask, 32));

		return _mm512_or_si512(x, _mm512_not_si512(_mask));
	}//End of _mm512_lzset_epi64

	///<summary>
	///<para> Compares 8-bit integers in an __m128i for not equality.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_cmpneq_epi8(const __m128i& a, const __m128i& b) noexcept
	{
		const __m128i _NEG_ONE = _mm_set1_epi8(static_cast<int8_t>(-1));
		__m128i _mask = _mm_setzero_si128();

		_mask = _mm_cmpeq_epi8(a, b);
		_mask = _mm_xor_si128(_mask, _NEG_ONE);//Not Equal

		return _mask;
	}//End of _mm_cmpneq_epi8

	///<summary>
	///<para> Compares 16-bit integers in an __m128i for not equality.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_cmpneq_epi16(const __m128i& a, const __m128i& b) noexcept
	{
		const __m128i _NEG_ONE = _mm_set1_epi16(static_cast<int16_t>(-1));
		__m128i _mask = _mm_setzero_si128();

		_mask = _mm_cmpeq_epi16(a, b);
		_mask = _mm_xor_si128(_mask, _NEG_ONE);//Not Equal

		return _mask;
	}//End of _mm_cmpneq_epi16

	///<summary>
	///<para> Compares 32-bit integers in an __m128i for not equality.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_cmpneq_epi32(const __m128i& a, const __m128i& b) noexcept
	{
		const __m128i _NEG_ONE = _mm_set1_epi32(-1);
		__m128i _mask = _mm_setzero_si128();

		_mask = _mm_cmpeq_epi32(a, b);
		_mask = _mm_xor_si128(_mask, _NEG_ONE);//Not Equal

		return _mask;
	}//End of _mm_cmpneq_epi32

	///<summary>
	///<para> Compares 64-bit integers in an __m128i for not equality.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE4.1.</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_cmpneq_epi64(const __m128i& a, const __m128i& b) noexcept
	{
		const __m128i _NEG_ONE = _mm_set1_epi64x(static_cast<int64_t>(-1));
		__m128i _mask = _mm_setzero_si128();

		_mask = _mm_cmpeq_epi64(a, b);
		_mask = _mm_xor_si128(_mask, _NEG_ONE);//Not Equal

		return _mask;
	}//End of _mm_cmpneq_epi64

	///<summary>
	///<para> Compares 8-bit integers in an __m256i for not equality.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_cmpneq_epi8(const __m256i& a, const __m256i& b) noexcept
	{
		const __m256i _NEG_ONE = _mm256_set1_epi8(static_cast<int8_t>(-1));
		__m256i _mask = _mm256_setzero_si256();

		_mask = _mm256_cmpeq_epi8(a, b);
		_mask = _mm256_xor_si256(_mask, _NEG_ONE);//Not Equal

		return _mask;
	}//End of _mm256_cmpneq_epi8

	///<summary>
	///<para> Compares 16-bit integers in an __m256i for not equality.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_cmpneq_epi16(const __m256i& a, const __m256i& b) noexcept
	{
		const __m256i _NEG_ONE = _mm256_set1_epi16(static_cast<int16_t>(-1));
		__m256i _mask = _mm256_setzero_si256();

		_mask = _mm256_cmpeq_epi16(a, b);
		_mask = _mm256_xor_si256(_mask, _NEG_ONE);//Not Equal

		return _mask;
	}//End of _mm256_cmpneq_epi16

	///<summary>
	///<para> Compares 32-bit integers in an __m256i for not equality.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_cmpneq_epi32(const __m256i& a, const __m256i& b) noexcept
	{
		const __m256i _NEG_ONE = _mm256_set1_epi32(-1);
		__m256i _mask = _mm256_setzero_si256();

		_mask = _mm256_cmpeq_epi32(a, b);
		_mask = _mm256_xor_si256(_mask, _NEG_ONE);//Not Equal

		return _mask;
	}//End of _mm256_cmpneq_epi32

	///<summary>
	///<para> Compares 64-bit integers in an __m256i for not equality.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE4.1.</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_cmpneq_epi64(const __m256i& a, const __m256i& b) noexcept
	{
		const __m256i _NEG_ONE = _mm256_set1_epi64x(static_cast<int64_t>(-1));
		__m256i _mask = _mm256_setzero_si256();

		_mask = _mm256_cmpeq_epi64(a, b);
		_mask = _mm256_xor_si256(_mask, _NEG_ONE);//Not Equal

		return _mask;
	}//End of _mm_cmpneq_epi64

	///<summary>
	///<para>Shifts 8-bit integers in an _m256i vector 'a' left by 'count' bits</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="count"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_sllv_epi8(const __m256i& a, const __m256i& count) 
	{
		const __m256i mask_hi = _mm256_set1_epi32(0xFF00FF00);
		const __m256i multiplier_lut = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 128, 64, 32, 16, 8, 4, 2, 1);
		
		const __m256i count_sat = _mm256_min_epu8(count, _mm256_set1_epi8(8));     
		const __m256i multiplier = _mm256_shuffle_epi8(multiplier_lut, count_sat); 
		const __m256i x_lo = _mm256_mullo_epi16(a, multiplier);               
		
		const __m256i multiplier_hi = _mm256_srli_epi16(multiplier, 8);       
		const __m256i a_hi = _mm256_and_si256(a, mask_hi);                    
		const __m256i x_hi = _mm256_mullo_epi16(a_hi, multiplier_hi);
		const __m256i x = _mm256_blendv_epi8(x_lo, x_hi, mask_hi);   

		return x;
	}//End of _mm256_sllv_epi8

	///<summary>
	///<para>Shifts 8-bit integers in an _m256i vector 'a' right by 'count' bits</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="count"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_srlv_epi8(const __m256i& a, const __m256i& count) 
	{
		const __m256i mask_hi = _mm256_set1_epi32(0xFF00FF00);
		const __m256i multiplier_lut = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, 128);

		const __m256i count_sat = _mm256_min_epu8(count, _mm256_set1_epi8(8));
		const __m256i multiplier = _mm256_shuffle_epi8(multiplier_lut, count_sat);
		const __m256i a_lo = _mm256_andnot_si256(mask_hi, a);
		const __m256i multiplier_lo = _mm256_andnot_si256(mask_hi, multiplier);
		__m256i x_lo = _mm256_mullo_epi16(a_lo, multiplier_lo);				
		x_lo = _mm256_srli_epi16(x_lo, 7);									

		const __m256i multiplier_hi = _mm256_and_si256(mask_hi, multiplier);
		__m256i x_hi = _mm256_mulhi_epu16(a, multiplier_hi);				
		x_hi = _mm256_slli_epi16(x_hi, 1);									
		const __m256i x = _mm256_blendv_epi8(x_lo, x_hi, mask_hi);

		return x;
	}//End of _mm256_srlv_epi8

	///<summary>
	///<para>Shifts 16-bit integers in an _m256i vector 'a' left by 'count' bits</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="count"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_sllv_epi16(const __m128i& a, const __m128i& count)
	{
		const __m128i mask = _mm_set1_epi32(0xffff0000);

		__m128i low_half = _mm_sllv_epi32(a, _mm_andnot_si128(mask, count));
		__m128i high_half = _mm_sllv_epi32(_mm_and_si128(mask, a), _mm_srli_epi32(count, 16));

		return _mm_blend_epi16(low_half, high_half, 0xaa);
	}//End of _mm_sllv_epi16

	///<summary>
	///<para>Shifts 16-bit integers in an _m256i vector 'a' left by 'count' bits</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="count"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_sllv_epi16(const __m256i& a, const __m256i& count) 
	{
		const __m256i mask = _mm256_set1_epi32(0xffff0000);

		__m256i low_half = _mm256_sllv_epi32(a,	_mm256_andnot_si256(mask, count));
		__m256i high_half = _mm256_sllv_epi32(_mm256_and_si256(mask, a),_mm256_srli_epi32(count, 16));
		
		return _mm256_blend_epi16(low_half, high_half, 0xaa);
	}//End of _mm256_sllv_epi16

	///<summary>
	///<para>Shifts 16-bit integers in an _m128i vector 'a' right by 'count' bits</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="count"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_srlv_epi16(const __m128i& a, const __m128i& count)
	{
		const __m128i mask = _mm_set1_epi32(0x0000ffff);

		__m128i low_half = _mm_srl_epi32(_mm_and_si128(mask, a), _mm_and_si128(mask, count));
		__m128i high_half = _mm_srl_epi32(a, _mm_srli_epi32(count, 16));

		return _mm_blend_epi16(low_half, high_half, 0xaa);
	}//End of _mm_srlv_epi16

	///<summary>
	///<para>Shifts 16-bit integers in an _m256i vector 'a' right by 'count' bits</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2</para>
	///</summary>
	/// <param name="a"></param>
	/// <param name="count"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_srlv_epi16(const __m256i& a, const __m256i& count) 
	{
		const __m256i mask = _mm256_set1_epi32(0x0000ffff);

		__m256i low_half = _mm256_srlv_epi32(_mm256_and_si256(mask, a),	_mm256_and_si256(mask, count));
		__m256i high_half = _mm256_srlv_epi32(a,_mm256_srli_epi32(count, 16));

		return _mm256_blend_epi16(low_half, high_half, 0xaa);
	}//End of _mm256_srlv_epi16

	///<summary>
	///<para> Returns the indexes of the lowest set bits of 16-bit integers in an __m128i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_bsf_epi16(__m128i x) noexcept
	{
		const __m128i x0000 = _mm_setzero_si128();
		const __m128i x5555 = _mm_set1_epi16(0x5555);
		const __m128i x3333 = _mm_set1_epi16(0x3333);
		const __m128i x0F0F = _mm_set1_epi16(0x0F0F);
		const __m128i x00FF = _mm_set1_epi16(0x00FF);

		__m128i r;
		x = _mm_and_si128(x, _mm_sub_epi16(x0000, x));
		r = _mm_slli_epi16(_mm_cmpeq_epi16(_mm_and_si128(x5555, x), x0000), 15);
		r = _mm_avg_epu16(r, _mm_cmpeq_epi16(_mm_and_si128(x3333, x), x0000));
		r = _mm_avg_epu16(r, _mm_cmpeq_epi16(_mm_and_si128(x0F0F, x), x0000));
		r = _mm_avg_epu16(r, _mm_cmpeq_epi16(_mm_and_si128(x00FF, x), x0000));
		r = _mm_sub_epi16(_mm_srli_epi16(r, 12), _mm_cmpeq_epi16(x, x0000));

		return r;
	}//End of _mm_bsf_epi16

	///<summary>
	///<para> Returns the indexes of the lowest set bits of 16-bit integers in an __m256i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_bsf_epi16(__m256i x) noexcept
	{
		const __m256i x0000 = _mm256_setzero_si256();
		const __m256i x5555 = _mm256_set1_epi16(0x5555);
		const __m256i x3333 = _mm256_set1_epi16(0x3333);
		const __m256i x0F0F = _mm256_set1_epi16(0x0F0F);
		const __m256i x00FF = _mm256_set1_epi16(0x00FF);

		__m256i r;
		x = _mm256_and_si256(x, _mm256_sub_epi16(x0000, x));
		r = _mm256_slli_epi16(_mm256_cmpeq_epi16(_mm256_and_si256(x5555, x), x0000), 15);
		r = _mm256_avg_epu16(r, _mm256_cmpeq_epi16(_mm256_and_si256(x3333, x), x0000));
		r = _mm256_avg_epu16(r, _mm256_cmpeq_epi16(_mm256_and_si256(x0F0F, x), x0000));
		r = _mm256_avg_epu16(r, _mm256_cmpeq_epi16(_mm256_and_si256(x00FF, x), x0000));
		r = _mm256_sub_epi16(_mm256_srli_epi16(r, 12), _mm256_cmpeq_epi16(x, x0000));

		return r;
	}//End of _mm256_bsf_epi16

	///<summary>
	///<para> Returns the indexes of the lowest set bits of 16-bit integers in an __m512i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX-512, AVX-512 BW and AVX-512 Bit Algorithms (BITALG).</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_bsf_epi16(__m512i x) noexcept
	{
		const __m512i _one = _mm512_set1_epi16(static_cast<int16_t>(1));

		return _mm512_add_epi16(_mm512_popcnt_epi16(_mm512_and_si512(_mm512_not_si512(x), _mm512_sub_epi16(x, _one))), _one);
	}//End of _mm512_bsf_epi16

	///<summary>
	///<para> Returns the indexes of the lowest set bits of 32-bit integers in an __m128i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without SSE2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m128i</returns>
	[[nodiscard]] inline __m128i _mm_bsf_epi32(const __m128i& x) noexcept
	{
		const __m128i xFFFF = _mm_set1_epi32(0xffff0000);
		const __m128i xFF00 = _mm_set1_epi32(0xff00ff00);
		const __m128i xF0F0 = _mm_set1_epi32(0xf0f0f0f0);
		const __m128i xCCCC = _mm_set1_epi32(0xcccccccc);
		const __m128i xAAAA = _mm_set1_epi32(0xaaaaaaaa);

		const __m128i _zero = _mm_set1_epi32(0);
		const __m128i _16 = _mm_set1_epi32(16);
		const __m128i _8 = _mm_set1_epi32(8);
		const __m128i _4 = _mm_set1_epi32(4);
		const __m128i _2 = _mm_set1_epi32(2);
		const __m128i _1 = _mm_set1_epi32(1);

		__m128i _index = _mm_setzero_si128();
		__m128i _mask = _mm_setzero_si128();

		_mask = _mm_and_si128(x, _mm_sub_epi32(_zero, x));

		_mask = _mm_and_si128(_mask, xFFFF);
		_mask = tpa::simd::_mm_cmpneq_epi32(_mask, _zero);
		_index = _mm_and_si128(_mask, _mm_add_epi32(_index, _16));

		_mask = _mm_and_si128(_mask, xFFFF);
		_mask = tpa::simd::_mm_cmpneq_epi32(_mask, _zero);
		_index = _mm_and_si128(_mask, _mm_add_epi32(_index, _8));

		_mask = _mm_and_si128(_mask, xFFFF);
		_mask = tpa::simd::_mm_cmpneq_epi32(_mask, _zero);
		_index = _mm_and_si128(_mask, _mm_add_epi32(_index, _4));

		_mask = _mm_and_si128(_mask, xFFFF);
		_mask = tpa::simd::_mm_cmpneq_epi32(_mask, _zero);
		_index = _mm_and_si128(_mask, _mm_add_epi32(_index, _2));

		_mask = _mm_and_si128(_mask, xFFFF);
		_mask = tpa::simd::_mm_cmpneq_epi32(_mask, _zero);
		_index = _mm_and_si128(_mask, _mm_add_epi32(_index, _1));

		return _index;
	}//End of _mm_bsf_epi32

	///<summary>
	///<para> Returns the indexes of the lowest set bits of 32-bit integers in an __m256i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX2.</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m256i</returns>
	[[nodiscard]] inline __m256i _mm256_bsf_epi32(const __m256i& x) noexcept
	{
		const __m256i xFFFF = _mm256_set1_epi32(0xffff0000);
		const __m256i xFF00 = _mm256_set1_epi32(0xff00ff00);
		const __m256i xF0F0 = _mm256_set1_epi32(0xf0f0f0f0);
		const __m256i xCCCC = _mm256_set1_epi32(0xcccccccc);
		const __m256i xAAAA = _mm256_set1_epi32(0xaaaaaaaa);
				 
		const __m256i _zero = _mm256_set1_epi32(0);
		const __m256i _16 = _mm256_set1_epi32(16);
		const __m256i _8 = _mm256_set1_epi32(8);
		const __m256i _4 = _mm256_set1_epi32(4);
		const __m256i _2 = _mm256_set1_epi32(2);
		const __m256i _1 = _mm256_set1_epi32(1);

		__m256i _index = _mm256_setzero_si256();
		__m256i _mask = _mm256_setzero_si256();

		_mask = _mm256_and_si256(x, _mm256_sub_epi32(_zero, x));

		_mask = _mm256_and_si256(_mask, xFFFF);
		_mask = tpa::simd::_mm256_cmpneq_epi32(_mask, _zero);
		_index = _mm256_and_si256(_mask, _mm256_add_epi32(_index, _16));

		_mask = _mm256_and_si256(_mask, xFFFF);
		_mask = tpa::simd::_mm256_cmpneq_epi32(_mask, _zero);
		_index = _mm256_and_si256(_mask, _mm256_add_epi32(_index, _8));

		_mask = _mm256_and_si256(_mask, xFFFF);
		_mask = tpa::simd::_mm256_cmpneq_epi32(_mask, _zero);
		_index = _mm256_and_si256(_mask, _mm256_add_epi32(_index, _4));

		_mask = _mm256_and_si256(_mask, xFFFF);
		_mask = tpa::simd::_mm256_cmpneq_epi32(_mask, _zero);
		_index = _mm256_and_si256(_mask, _mm256_add_epi32(_index, _2));

		_mask = _mm256_and_si256(_mask, xFFFF);
		_mask = tpa::simd::_mm256_cmpneq_epi32(_mask, _zero);
		_index = _mm256_and_si256(_mask, _mm256_add_epi32(_index, _1));

		return _index;
	}//End of _mm256_bsf_epi32

	///<summary>
	///<para> Returns the indexes of the lowest set bits of 32-bit integers in an __m512i.</para>
	///<para>Note: This is not a hardware intrinsic, it is a function consisting of several instructions.</para>
	///<para>Note: This function is a part of TPA and is not available by default within 'immintrin.h' or SVML.</para>
	/// <para>Warning: This function is unsafe if called on a platform without AVX-512, AVX-512 BW and AVX-512 Bit Algorithms (BITALG).</para>
	///</summary>
	/// <param name="x"></param>
	/// <returns>__m512i</returns>
	[[nodiscard]] inline __m512i _mm512_bsf_epi32(__m512i x) noexcept
	{
		const __m512i _one = _mm512_set1_epi32(static_cast<int16_t>(1));

		return _mm512_add_epi32(_mm512_popcnt_epi32(_mm512_and_si512(_mm512_not_si512(x), _mm512_sub_epi32(x, _one))), _one);
	}//End of _mm512_bsf_epi32

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
		constexpr __m128 sign_mask = { -0.0f, -0.0f, -0.0f, -0.0f };
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
		constexpr __m128d sign_mask = { -0.0, -0.0 };
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
		constexpr __m256 sign_mask = { -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f };
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
		constexpr __m256d sign_mask = { -0.0, -0.0, -0.0, -0.0 };
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

		return tpa::simd::_mm_cvtpd_epi64(val);
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

		return tpa::simd::_mm256_cvtpd_epi64(val);
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
		return tpa::simd::_mm_sum_epi32(_temp);
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
		return tpa::simd::_mm256_sum_epi32(_temp);
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
		return tpa::simd::_mm512_sum_epi32(_temp);
	}//End of _mm512_sum_epi16
#endif
#pragma endregion
};
