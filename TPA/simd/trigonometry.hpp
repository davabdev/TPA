#pragma once
/*
*	Trigonometry functions for TPA Library
*	By: David Aaron Braun
*	2021-10-22
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include <type_traits>
#include <concepts>
#include <utility>
#include <mutex>
#include <future>
#include <iostream>
#include <functional>
#include <numbers>
#include <cmath>
#include <vector>
#include <array>

#ifdef _M_AMD64
#include <immintrin.h>
#elif defined (_M_ARM64)
#ifdef _WIN32
#include "arm64_neon.h"
#else
#include "arm_neon.h"
#endif
#endif

#include "../_util.hpp"
#include "../ThreadPool.hpp"
#include "../excepts.hpp"

#undef min
#undef max
#undef abs
#undef sin
#undef cos
#undef tan
#undef acos
#undef asin
#undef atan
#undef atan2
#undef cosh
#undef sinh
#undef tanh
#undef acosh
#undef asinh
#undef atanh
#undef hypot

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>SIMD Matrix Math Functions.</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa::simd
{
#pragma region generic

	/// <summary>
	/// <para>Computes trigonometic functions on an std::vector storing the result in a 2nd std::vector.</para>
	/// <para>This function can only take advantage of SIMD if both containers' value_type is float or both double (much, much faster!). </para>
	/// <para>Else uses only multi-threading and the results are static_cast to the value_type of the destination container. Passing containers of non-standard value_types is allowed but may deliver truncted or incorrect results as this function relies on standard cmath functions.</para>
	/// <para>Containers do not have to be a particular size</para>
	/// <para>It is recommened to use Radians as opposed to Degrees as Degrees often have to be converted to Radians and thus a small performance hit is incurred.</para>
	/// <para>Takes 2 templated predicates: tpa::trig and tpa::angle</para>
	/// <para>tpa::trig::SINE, tpa::angle::RADIANS / tpa::angle::DEGREES</para>
	/// <para>tpa::trig::COSINE, tpa::angle::RADIANS / tpa::angle::DEGREES</para>
	/// <para>tpa::trig::TANGENT, tpa::angle::RADIANS / tpa::angle::DEGREES</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	template<tpa::trig INSTR, tpa::angle ANG, typename CONTAINER_A, typename DEST>
	inline constexpr void trigonometry(const CONTAINER_A& source, DEST& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<DEST>
	{		
		uint32_t complete = 0;
		size_t smallest = source.size();
		try
		{
			using T = CONTAINER_A::value_type;
			using RES = DEST::value_type;

			if (dest.size() < smallest)
			{
				throw tpa::exceptions::ArrayTooSmall();
			}//End if
		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region int32-float
						if constexpr (std::is_same<T, int32_t>() && std::is_same<RES, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _dest, _source;
								__m512i _nums;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_nums = _mm512_load_epi32(&source[i]);
									_source = _mm512_cvtepi32_ps(_nums);
									_dest = _mm512_setzero_ps();

									//Sine
									if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_sin_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm512_sind_ps(_source));
									}//End if

									//Hyperbolic Sine
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_sinh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_sinh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Sine
									else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_asin_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_asin_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Sine
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_asinh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_asinh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Cosine
									else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_cos_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm512_cosd_ps(_source));
									}//End if

									//Hyperbolic Cosine
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_cosh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_cosh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Cosine
									else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_acos_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_acos_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Cosine
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_acosh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_acosh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Tangent
									else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_tan_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm512_tand_ps(_source));
									}//End if

									//Hyperbolic Tangent
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_tanh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_tanh_ps(_source);

										_dest = tpa::util::radians_to_degrees<__m512>(_dest);
									}//End if

									//Inverse Tangent
									else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_atan_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_atan_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Tangent
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_atanh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_atanh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _source, _dest;
								__m256i _nums;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_nums = _mm256_load_si256((__m256i*)&source[i]);
									_source = _mm256_cvtepi32_ps(_nums);
									_dest = _mm256_setzero_ps();

									//Sine
									if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_sin_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm256_sind_ps(_source));
									}//End if

									//Hyperbolic Sine
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_sinh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm256_sinh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Sine
									else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_asin_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm256_asin_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Sine
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_asinh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm256_asinh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Cosine
									else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_cos_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm256_cosd_ps(_source));
									}//End if

									//Hyperbolic Cosine
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_cosh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm256_cosh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Cosine
									else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_acos_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians<__m256>(_source);

										_dest = _mm256_acos_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Cosine
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_acosh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm256_acosh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Tangent
									else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_tan_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm256_tand_ps(_source));
									}//End if

									//Hyperbolic Tangent
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_tanh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm256_tanh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Tangent
									else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_atan_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm256_atan_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Tangent
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm256_atanh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm256_atanh_ps(_source);

										_dest = tpa::util::radians_to_degrees<__m256>(_dest);
									}//End if

									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result									
									_mm256_store_ps(& dest[i], _dest);

								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE2)
							{
								__m128 _source, _dest;
								__m128i _nums;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_nums = _mm_load_si128((__m128i*)&source[i]);
									_source = _mm_cvtepi32_ps(_nums);
									_dest = _mm_setzero_ps();

									//Sine
									if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_sin_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm_sind_ps(_source));
									}//End if

									//Hyperbolic Sine
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_sinh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm_sinh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Sine
									else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_asin_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm_asin_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Sine
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_asinh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm_asinh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Cosine
									else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_cos_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm_cosd_ps(_source));
									}//End if

									//Hyperbolic Cosine
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_cosh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm_cosh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Cosine
									else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_acos_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm_acos_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Cosine
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_acosh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm_acosh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Tangent
									else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_tan_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm_tand_ps(_source));
									}//End if

									//Hyperbolic Tangent
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_tanh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm_tanh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Tangent
									else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_atan_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm_atan_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Tangent
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm_atanh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm_atanh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result									
									_mm_store_ps(&dest[i], _dest);

								}//End for
							}//End if has_SSE
#endif						
						}//End if
#pragma endregion
#pragma region uint32-float
						else if constexpr (std::is_same<T, uint32_t>() && std::is_same<RES, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _dest, _source;
								__m512i _nums;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_nums = _mm512_load_epu32(&source[i]);
									_source = _mm512_cvtepu32_ps(_nums);
									_dest = _mm512_setzero_ps();

									//Sine
									if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_sin_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm512_sind_ps(_source));
									}//End if

									//Hyperbolic Sine
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_sinh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_sinh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Sine
									else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_asin_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_asin_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Sine
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_asinh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_asinh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Cosine
									else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_cos_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm512_cosd_ps(_source));
									}//End if

									//Hyperbolic Cosine
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_cosh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_cosh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Cosine
									else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_acos_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_acos_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Cosine
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_acosh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_acosh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Tangent
									else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_tan_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
									{
										_dest = tpa::util::radians_to_degrees(_mm512_tand_ps(_source));
									}//End if

									//Hyperbolic Tangent
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_tanh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_tanh_ps(_source);

										_dest = tpa::util::radians_to_degrees<__m512>(_dest);
									}//End if

									//Inverse Tangent
									else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_atan_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_atan_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									//Inverse Hyperbolic Tangent
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
									{
										_dest = _mm512_atanh_ps(_source);
									}//End if
									else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
									{
										_source = tpa::util::degrees_to_radians(_source);

										_dest = _mm512_atanh_ps(_source);

										_dest = tpa::util::radians_to_degrees(_dest);
									}//End if

									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
#endif
						}//End if
#pragma endregion
#pragma region float
					else if constexpr (std::is_same<T, float>() && std::is_same<RES, float>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512 _source, _dest;

							for (; (i+16) < end; i += 16)
							{
								//Set Values
								_source = _mm512_load_ps(&source[i]);
								_dest = _mm512_setzero_ps();

								//Sine
								if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_sin_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_sind_ps(_source));
								}//End if

								//Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_sinh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_sinh_ps(_source);
									
									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_asin_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_asin_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_asinh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_asinh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Cosine
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_cos_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_cosd_ps(_source));
								}//End if

								//Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_cosh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_cosh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_acos_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_acos_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_acosh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_acosh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Tangent
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_tan_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_tand_ps(_source));
								}//End if

								//Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_tanh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_tanh_ps(_source);

									_dest = tpa::util::radians_to_degrees<__m512>(_dest);
								}//End if

								//Inverse Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_atan_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_atan_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_atanh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_atanh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm512_store_ps(&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							__m256 _source, _dest;

							for (; (i+8) < end; i += 8)
							{
								//Set Values
								_source = _mm256_load_ps(&source[i]);
								_dest = _mm256_setzero_ps();

								//Sine
								if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_sin_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_sind_ps(_source));
								}//End if

								//Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_sinh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_sinh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_asin_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_asin_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_asinh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_asinh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Cosine
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_cos_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_cosd_ps(_source));
								}//End if

								//Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_cosh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_cosh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_acos_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians<__m256>(_source);

									_dest = _mm256_acos_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_acosh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_acosh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Tangent
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_tan_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_tand_ps(_source));
								}//End if

								//Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_tanh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_tanh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_atan_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_atan_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_atanh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_atanh_ps(_source);

									_dest = tpa::util::radians_to_degrees<__m256>(_dest);
								}//End if

								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm256_store_ps(&dest[i], _dest);
							}//End for
						}//End if hasAVX
						else if (tpa::has_SSE)
						{
						__m128 _source, _dest;

						for (; (i+4) < end; i += 4)
						{
							//Set Values
							_source = _mm_load_ps(&source[i]);
							_dest = _mm_setzero_ps();

							//Sine
							if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_sin_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm_sind_ps(_source));
							}//End if

							//Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_sinh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_sinh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_asin_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_asin_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_asinh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_asinh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Cosine
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_cos_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm_cosd_ps(_source));
							}//End if

							//Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_cosh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_cosh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_acos_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_acos_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_acosh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_acosh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Tangent
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_tan_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm_tand_ps(_source));
							}//End if

							//Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_tanh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_tanh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_atan_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_atan_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_atanh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_atanh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
								}();
							}//End else

							//Store Result
							_mm_store_ps(&dest[i], _dest);
						}//End for
						}//End if has_SSE
#endif						
					}//End if
#pragma endregion
#pragma region float-int32
					else if constexpr (std::is_same<T, float>() && std::is_same<RES, int32_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512 _source, _dest;
							__m512i _res;

							for (; (i + 16) < end; i += 16)
							{
								//Set Values
								_source = _mm512_load_ps(&source[i]);
								_dest = _mm512_setzero_ps();

								//Sine
								if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_sin_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_sind_ps(_source));
								}//End if

								//Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_sinh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_sinh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_asin_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_asin_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_asinh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_asinh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Cosine
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_cos_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_cosd_ps(_source));
								}//End if

								//Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_cosh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_cosh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_acos_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_acos_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_acosh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_acosh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Tangent
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_tan_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_tand_ps(_source));
								}//End if

								//Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_tanh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_tanh_ps(_source);

									_dest = tpa::util::radians_to_degrees<__m512>(_dest);
								}//End if

								//Inverse Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_atan_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_atan_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_atanh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_atanh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_res = _mm512_cvt_roundps_epi32(_dest, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
								_mm512_store_epi32(&dest[i], _res);

							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							__m256 _source, _dest;
							__m256i _res;

							for (; (i + 8) < end; i += 8)
							{
								//Set Values
								_source = _mm256_load_ps(&source[i]);
								_dest = _mm256_setzero_ps();

								//Sine
								if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_sin_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_sind_ps(_source));
								}//End if

								//Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_sinh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_sinh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_asin_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_asin_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_asinh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_asinh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Cosine
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_cos_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_cosd_ps(_source));
								}//End if

								//Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_cosh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_cosh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_acos_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians<__m256>(_source);

									_dest = _mm256_acos_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_acosh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_acosh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Tangent
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_tan_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_tand_ps(_source));
								}//End if

								//Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_tanh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_tanh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_atan_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_atan_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_atanh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_atanh_ps(_source);

									_dest = tpa::util::radians_to_degrees<__m256>(_dest);
								}//End if

								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_res = _mm256_cvtps_epi32(_mm256_round_ps(_dest, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
								_mm256_store_si256((__m256i*)&dest[i], _res);

							}//End for
						}//End if hasAVX
						else if (tpa::has_SSE41)
						{
							__m128 _source, _dest;
							__m128i _res;

							for (; (i + 4) < end; i += 4)
							{
								//Set Values
								_source = _mm_load_ps(&source[i]);
								_dest = _mm_setzero_ps();

								//Sine
								if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_sin_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm_sind_ps(_source));
								}//End if

								//Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_sinh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm_sinh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_asin_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm_asin_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_asinh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm_asinh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Cosine
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_cos_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm_cosd_ps(_source));
								}//End if

								//Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_cosh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm_cosh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_acos_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm_acos_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_acosh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm_acosh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Tangent
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_tan_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm_tand_ps(_source));
								}//End if

								//Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_tanh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm_tanh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_atan_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm_atan_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_atanh_ps(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm_atanh_ps(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_res = _mm_cvtps_epi32(_mm_round_ps(_dest, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
								_mm_store_si128((__m128i*)& dest[i], _res);

							}//End for
						}//End if has_SSE
#endif						
					}//End if
#pragma endregion
#pragma region float-uint32
					else if constexpr (std::is_same<T, float>() && std::is_same<RES, uint32_t>())
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512 _source, _dest;
						__m512i _res;

						for (; (i + 16) < end; i += 16)
						{
							//Set Values
							_source = _mm512_load_ps(&source[i]);
							_dest = _mm512_setzero_ps();

							//Sine
							if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_sin_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_sind_ps(_source));
							}//End if

							//Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_sinh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_sinh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_asin_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_asin_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_asinh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_asinh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Cosine
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_cos_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_cosd_ps(_source));
							}//End if

							//Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_cosh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_cosh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_acos_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_acos_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_acosh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_acosh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Tangent
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_tan_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_tand_ps(_source));
							}//End if

							//Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_tanh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_tanh_ps(_source);

								_dest = tpa::util::radians_to_degrees<__m512>(_dest);
							}//End if

							//Inverse Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_atan_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_atan_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_atanh_ps(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_atanh_ps(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
								}();
							}//End else

							//Store Result
							_res = _mm512_cvt_roundps_epu32(_dest, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
							_mm512_store_epu32(&dest[i], _res);

						}//End for
					}//End if hasAVX512
#endif						
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>() && std::is_same<RES, double>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512d _source, _dest;

							for (; (i+8) < end; i += 8)
							{
								//Set Values
								_source = _mm512_load_pd(&source[i]);
								_dest = _mm512_setzero_pd();

								//Sine
								if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_sin_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_sind_pd(_source));
								}//End if

								//Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_sinh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_sinh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_asin_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_asin_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_asinh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_asinh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Cosine
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_cos_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_cosd_pd(_source));
								}//End if

								//Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_cosh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_cosh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_acos_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_acos_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_acosh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_acosh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Tangent
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_tan_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_tand_pd(_source));
								}//End if

								//Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_tanh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_tanh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_atan_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_atan_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_atanh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm512_atanh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
									}();
								}//End else

								//Store Result
								_mm512_store_pd(&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							__m256d _source, _dest;

							for (; (i+4) < end; i += 4)
							{
								//Set Values
								_source = _mm256_load_pd(&source[i]);
								_dest = _mm256_setzero_pd();

								//Sine
								if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_sin_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_sind_pd(_source));
								}//End if

								//Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_sinh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_sinh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_asin_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_asin_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Sine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_asinh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_asinh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Cosine
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_cos_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_cosd_pd(_source));
								}//End if

								//Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_cosh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_cosh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_acos_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_acos_pd(_source);

									_source = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Cosine
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_acosh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_acosh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Tangent
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_tan_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_tand_pd(_source));
								}//End if

								//Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_tanh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_tanh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_atan_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_atan_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								//Inverse Hyperbolic Tangent
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_atanh_pd(_source);
								}//End if
								else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
								{
									_source = tpa::util::degrees_to_radians(_source);

									_dest = _mm256_atanh_pd(_source);

									_dest = tpa::util::radians_to_degrees(_dest);
								}//End if

								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
									}();
								}//End else

								//Store Result
								_mm256_store_pd(&dest[i], _dest);
							}//End for
						}//End if hasAVX
						else if (tpa::has_SSE2)
						{
						__m128d _source, _dest;

						for (; (i+2) < end; i += 2)
						{
							//Set Values
							_source = _mm_load_pd(&source[i]);
							_dest = _mm_setzero_pd();

							//Sine
							if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_sin_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm_sind_pd(_source));
							}//End if

							//Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_sinh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_sinh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_asin_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_asin_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_asinh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_asinh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Cosine
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_cos_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm_cosd_pd(_source));
							}//End if

							//Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_cosh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_cosh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_acos_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_acos_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_acosh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_acosh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Tangent
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_tan_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm_tand_pd(_source));
							}//End if

							//Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_tanh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_tanh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_atan_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_atan_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm_atanh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm_atanh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
								}();
							}//End else

							//Store Result
							_mm_store_pd(&dest[i], _dest);
						}//End for
						}//End if has_SSE2
#endif						
					}//End if
#pragma endregion
#pragma region double-int64
					else if constexpr (std::is_same<T, double>() && std::is_same<RES, int64_t>())
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512_DWQW)
					{
						__m512d _source, _dest;
						__m512i _res;

						for (; (i + 8) < end; i += 8)
						{
							//Set Values
							_source = _mm512_load_pd(&source[i]);
							_dest = _mm512_setzero_pd();

							//Sine
							if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_sin_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_sind_pd(_source));
							}//End if

							//Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_sinh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_sinh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_asin_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_asin_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_asinh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_asinh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Cosine
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_cos_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_cosd_pd(_source));
							}//End if

							//Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_cosh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_cosh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_acos_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_acos_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_acosh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_acosh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Tangent
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_tan_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_tand_pd(_source));
							}//End if

							//Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_tanh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_tanh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_atan_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_atan_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_atanh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_atanh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
								}();
							}//End else

							//Store Result
							_res = _mm512_cvt_roundpd_epi64(_dest, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
							_mm512_store_epi64(&dest[i], _res);

						}//End for
					}//End if hasAVX512
#endif						
					}//End if
#pragma endregion
#pragma region double-uint64
					else if constexpr (std::is_same<T, double>() && std::is_same<RES, uint64_t>())
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512_DWQW)
					{
						__m512d _source, _dest;
						__m512i _res;

						for (; (i + 8) < end; i += 8)
						{
							//Set Values
							_source = _mm512_load_pd(&source[i]);
							_dest = _mm512_setzero_pd();

							//Sine
							if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_sin_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_sind_pd(_source));
							}//End if

							//Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_sinh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_sinh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_asin_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_asin_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_asinh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_asinh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Cosine
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_cos_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_cosd_pd(_source));
							}//End if

							//Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_cosh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_cosh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_acos_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_acos_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_acosh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_acosh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Tangent
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_tan_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_tand_pd(_source));
							}//End if

							//Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_tanh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_tanh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_atan_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_atan_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_atanh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_atanh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
								}();
							}//End else

							//Store Result
							_res = _mm512_cvt_roundpd_epu64(_dest, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
							_mm512_store_epi64(&dest[i], _res);

						}//End for
					}//End if hasAVX512
#endif						
					}//End if
#pragma endregion
#pragma region int64-double
					else if constexpr (std::is_same<T, int64_t>() && std::is_same<RES, double>())
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512_DWQW)
					{
						__m512d _source, _dest;
						__m512i _nums;

						for (; (i + 8) < end; i += 8)
						{
							//Set Values
							_nums = _mm512_load_epi64(&source[i]);
							_source = _mm512_cvtepi64_pd(_nums);
							_dest = _mm512_setzero_ps();

							//Sine
							if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_sin_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_sind_pd(_source));
							}//End if

							//Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_sinh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_sinh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_asin_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_asin_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_asinh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_asinh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Cosine
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_cos_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_cosd_pd(_source));
							}//End if

							//Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_cosh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_cosh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_acos_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_acos_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_acosh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_acosh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Tangent
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_tan_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_tand_pd(_source));
							}//End if

							//Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_tanh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_tanh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_atan_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_atan_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_atanh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_atanh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
								}();
							}//End else

							//Store Result							
							_mm512_store_pd(&dest[i], _dest);

						}//End for
					}//End if hasAVX512
#endif						
					}//End if
#pragma endregion
#pragma region uint64-double
					else if constexpr (std::is_same<T, uint64_t>() && std::is_same<RES, double>())
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512_DWQW)
					{
						__m512d _source, _dest;
						__m512i _nums;

						for (; (i + 8) < end; i += 8)
						{
							//Set Values
							_nums = _mm512_load_epu64(&source[i]);
							_source = _mm512_cvtepu64_pd(_nums);
							_dest = _mm512_setzero_ps();

							//Sine
							if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_sin_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_sind_pd(_source));
							}//End if

							//Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_sinh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_sinh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_asin_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_asin_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_asinh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_asinh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Cosine
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_cos_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_cosd_pd(_source));
							}//End if

							//Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_cosh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_cosh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_acos_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_acos_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_acosh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_acosh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Tangent
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_tan_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
							{
								_dest = tpa::util::radians_to_degrees(_mm512_tand_pd(_source));
							}//End if

							//Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_tanh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_tanh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_atan_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_atan_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							//Inverse Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								_dest = _mm512_atanh_pd(_source);
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								_source = tpa::util::degrees_to_radians(_source);

								_dest = _mm512_atanh_pd(_source);

								_dest = tpa::util::radians_to_degrees(_dest);
							}//End if

							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
								}();
							}//End else

							//Store Result							
							_mm512_store_pd(&dest[i], _dest);

						}//End for
					}//End if hasAVX512
#endif						
					}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Sine
							if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::sin(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::SINE && ANG == tpa::angle::DEGREES)
							{		
								dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::sin(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							//Inverse Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG ==  tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::asin(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_SINE && ANG ==  tpa::angle::DEGREES)
							{
								dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::asin(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							//Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG ==  tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::sinh(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_SINE && ANG ==  tpa::angle::DEGREES)
							{
								dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::sinh(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							//Inverse Hyperbolic Sine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG ==  tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::asinh(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_SINE && ANG ==  tpa::angle::DEGREES)
							{
								dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::asinh(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							//Cosine
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::cos(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::COSINE && ANG == tpa::angle::DEGREES)
							{
								dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::cos(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							//Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::cosh(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::cosh(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							//Inverse Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::acos(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_COSINE && ANG == tpa::angle::DEGREES)
							{
								dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::acos(tpa::util::degrees_to_radians(source[i]))));
							}//End if
							
							//Inverse Hyperbolic Cosine
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::acosh(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_COSINE && ANG == tpa::angle::DEGREES)
							{
								dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::acosh(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							//Tangent
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::tan(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::TANGENT && ANG == tpa::angle::DEGREES)
							{
								dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::tan(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							//Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::tanh(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
								dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::tanh(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							//Inverse Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::atan(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_TANGENT && ANG == tpa::angle::DEGREES)
							{
							dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::atan(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							//Inverse Hyperbolic Tangent
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::RADIANS)
							{
								dest[i] = static_cast<RES>(std::atanh(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::trig::INVERSE_HYPERBOLIC_TANGENT && ANG == tpa::angle::DEGREES)
							{
							dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::atanh(tpa::util::degrees_to_radians(source[i]))));
							}//End if

							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::trigonometry<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
								}();
							}//End else
						}//End for
#pragma endregion
						return static_cast<uint32_t>(1);
					});//End of lambda

				results.emplace_back(std::move(temp));
			}//End for

			for (const auto& fut : results)
			{
				complete += fut.get();
			}//End for			

			//Check all threads completed
			if (complete != nThreads)
			{
				throw tpa::exceptions::NotAllThreadsCompleted(complete);
			}//End if

		}//End try
		catch (const tpa::exceptions::ArrayTooSmall& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::trigonometry: " << ex.what() << "\n";
			std::cerr << "tpa::simd::trigonometry will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::trigonometry: " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::trigonometry: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::trigonometry: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::trigonometry: unknown!\n";
		}//End catch
	}//End of trigonometry()
	
	/// <summary>
	/// <para>Computes the arc tangent of numbers in 'source1' and 'source2' using the signs of arguments to determine the correct quadrant storing the results in 'dest'</para>
	/// <para>This function can only take advantage of SIMD if all containers' value_type is float or both double (much, much faster!). </para>
	/// <para>Else uses only multi-threading and the results are static_cast to the value_type of the destination container. Passing containers of non-standard value_types is allowed but may deliver truncted or incorrect results as this function relies on standard cmath functions.</para>
	/// <para>Containers do not have to be a particular size</para>
	/// <para>It is recommened to use Radians as opposed to Degrees as Degrees often have to be converted to Radians and thus a small performance hit is incurred.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="DEST"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::angle ANG, typename CONTAINER_A, typename CONTAINER_B, typename DEST>
	inline constexpr void atan2(
		const CONTAINER_A& source1, 
		const CONTAINER_B& source2, 
		DEST& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>&&
		tpa::util::contiguous_seqeunce<DEST>
	{
		uint32_t complete = 0;
		size_t smallest = tpa::util::min(source1.size(), source2.size());
		try
		{
			using T = CONTAINER_A::value_type;
			using T2 = CONTAINER_B::value_type;
			using RES = DEST::value_type;

			if (dest.size() < smallest)
			{
				throw tpa::exceptions::ArrayTooSmall();
			}//End if
		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &source2, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region float
					if constexpr (std::is_same<T, float>() && std::is_same<T2, float>() && std::is_same<RES, float>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512 _source1, _source2, _dest;

							for (; (i + 16uz) < end; i += 16uz)
							{
								//Set Values
								_source1 = _mm512_load_ps(&source1[i]);
								_source2 = _mm512_load_ps(&source2[i]);
								_dest = _mm512_setzero_ps();

								//atan2
								if constexpr (ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_atan2_ps(_source1, _source2);
								}//End if
								else if constexpr (ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_atan2_ps(
										tpa::util::degrees_to_radians(_source1), 
										tpa::util::degrees_to_radians(_source2))
									);
								}//End else
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid angle prdicate in tpa::simd::atan2<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm512_store_ps(&dest[i], _dest);
							}//End for
						}//End of hasAVX512
						else if (tpa::hasAVX)
						{
							__m256 _source1, _source2, _dest;

							for (; (i + 8uz) < end; i += 8uz)
							{
								//Set Values
								_source1 = _mm256_load_ps(&source1[i]);
								_source2 = _mm256_load_ps(&source2[i]);
								_dest = _mm256_setzero_ps();

								//atan2
								if constexpr (ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_atan2_ps(_source1, _source2);
								}//End if
								else if constexpr (ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_atan2_ps(
										tpa::util::degrees_to_radians(_source1),
										tpa::util::degrees_to_radians(_source2))
									);
								}//End else
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid angle prdicate in tpa::simd::atan2<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm256_store_ps(&dest[i], _dest);
							}//End for
						}//End of hasAVX
						else if (tpa::has_SSE)
						{
							__m128 _source1, _source2, _dest;

							for (; (i + 4uz) < end; i += 4uz)
							{
								//Set Values
								_source1 = _mm_load_ps(&source1[i]);
								_source2 = _mm_load_ps(&source2[i]);
								_dest = _mm_setzero_ps();

								//atan2
								if constexpr (ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_atan2_ps(_source1, _source2);
								}//End if
								else if constexpr (ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm_atan2_ps(
										tpa::util::degrees_to_radians(_source1),
										tpa::util::degrees_to_radians(_source2))
									);
								}//End else
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid angle prdicate in tpa::simd::atan2<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm_store_ps(&dest[i], _dest);
							}//End for
						}//End of has_SSE
#endif
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>() && std::is_same<T2, double>() && std::is_same<RES, double>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512d _source1, _source2, _dest;

							for (; (i + 8uz) < end; i += 8uz)
							{
								//Set Values
								_source1 = _mm512_load_pd(&source1[i]);
								_source2 = _mm512_load_pd(&source2[i]);
								_dest = _mm512_setzero_pd();

								//atan2
								if constexpr (ANG == tpa::angle::RADIANS)
								{
									_dest = _mm512_atan2_pd(_source1, _source2);
								}//End if
								else if constexpr (ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm512_atan2_pd(
										tpa::util::degrees_to_radians(_source1),
										tpa::util::degrees_to_radians(_source2))
									);
								}//End else
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid angle prdicate in tpa::simd::atan2<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
									}();
								}//End else

								//Store Result
								_mm512_store_pd(&dest[i], _dest);
							}//End for
						}//End of hasAVX512
						else if (tpa::hasAVX)
						{
							__m256d _source1, _source2, _dest;

							for (; (i + 4uz) < end; i += 4uz)
							{
								//Set Values
								_source1 = _mm256_load_pd(&source1[i]);
								_source2 = _mm256_load_pd(&source2[i]);
								_dest = _mm256_setzero_pd();

								//atan2
								if constexpr (ANG == tpa::angle::RADIANS)
								{
									_dest = _mm256_atan2_pd(_source1, _source2);
								}//End if
								else if constexpr (ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm256_atan2_pd(
										tpa::util::degrees_to_radians(_source1),
										tpa::util::degrees_to_radians(_source2))
									);
								}//End else
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid angle prdicate in tpa::simd::atan2<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
									}();
								}//End else

								//Store Result
								_mm256_store_pd(&dest[i], _dest);
							}//End for
						}//End of hasAVX
						else if (tpa::has_SSE2)
						{
							__m128d _source1, _source2, _dest;

							for (; (i + 2uz) < end; i += 2uz)
							{
								//Set Values
								_source1 = _mm_load_pd(&source1[i]);
								_source2 = _mm_load_pd(&source2[i]);
								_dest = _mm_setzero_pd();

								//atan2
								if constexpr (ANG == tpa::angle::RADIANS)
								{
									_dest = _mm_atan2_pd(_source1, _source2);
								}//End if
								else if constexpr (ANG == tpa::angle::DEGREES)
								{
									_dest = tpa::util::radians_to_degrees(_mm_atan2_pd(
										tpa::util::degrees_to_radians(_source1),
										tpa::util::degrees_to_radians(_source2))
									);
								}//End else
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid angle prdicate in tpa::simd::atan2<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
									}();
								}//End else

								//Store Result
								_mm_store_pd(&dest[i], _dest);
							}//End for
						}//End of has_SSE
#endif
					}//End if
#pragma endregion
#pragma region generic
					for (; i != end; ++i)
					{
						//atan2
						if constexpr (ANG == tpa::angle::RADIANS)
						{
							dest[i] = static_cast<RES>(std::atan2(source1[i], source2[i]));
						}//End if
						else if constexpr (ANG == tpa::angle::DEGREES)
						{
							dest[i] = static_cast<RES>(tpa::util::radians_to_degrees(std::atan2(
								tpa::util::degrees_to_radians(source1[i]),
								tpa::util::degrees_to_radians(source2[i]))
							));
						}//End else
						else
						{
							[] <bool flag = false>()
							{
								static_assert(flag, " You have specifed an invalid angle prdicate in tpa::simd::atan2<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
							}();
						}//End else
					}//End for
#pragma endregion

						return static_cast<uint32_t>(1);
					});//End of lambda

				results.emplace_back(std::move(temp));
			}//End for

			for (const auto& fut : results)
			{
				complete += fut.get();
			}//End for			

			//Check all threads completed
			if (complete != nThreads)
			{
				throw tpa::exceptions::NotAllThreadsCompleted(complete);
			}//End if

		}//End try
		catch (const tpa::exceptions::ArrayTooSmall& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::atan2: " << ex.what() << "\n";
			std::cerr << "tpa::simd::atan2 will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::atan2: " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::atan2: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::atan2: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::atan2: unknown!\n";
		}//End catch
	}//End of atan2()

	/// <summary>
	/// <para>Computes the square root of the sum of the squares of numbers in 'source1' and 'source2', without undue overflow or underflow at intermediate stages of the computation storing the results in 'dest'.</para>
	/// <para>This function can only take advantage of SIMD if all containers' value_type is float or both double (much, much faster!). </para>
	/// <para>Else uses only multi-threading and the results are static_cast to the value_type of the destination container. Passing contianers of non-standard value_types is allowed but may deliver truncted or incorrect results as this function relies on standard cmath functions.</para>
	/// <para>Containers do not have to be a particular size</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// 	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="DEST"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<typename CONTAINER_A, typename CONTAINER_B, typename DEST>
	inline constexpr void hypot(
		const CONTAINER_A& source1,
		const CONTAINER_B& source2,
		DEST& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>&&
		tpa::util::contiguous_seqeunce<DEST>
	{
		uint32_t complete = 0;
		size_t smallest = tpa::util::min(source1.size(), source2.size());
		try
		{
			using T = CONTAINER_A::value_type;
			using T2 = CONTAINER_B::value_type;
			using RES = DEST::value_type;

			if (dest.size() < smallest)
			{
				throw tpa::exceptions::ArrayTooSmall();
			}//End if
		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &source2, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region float
						if constexpr (std::is_same<T, float>() && std::is_same<T2, float>() && std::is_same<RES, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _source1, _source2, _dest;

								for (; (i + 16uz) < end; i += 16uz)
								{
									//Set Values
									_source1 = _mm512_load_ps(&source1[i]);
									_source2 = _mm512_load_ps(&source2[i]);
									
									//hypot
									_dest = _mm512_hypot_ps(_source1, _source2);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End of hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _source1, _source2, _dest;

								for (; (i + 8uz) < end; i += 8uz)
								{
									//Set Values
									_source1 = _mm256_load_ps(&source1[i]);
									_source2 = _mm256_load_ps(&source2[i]);
									
									//hypot
									_dest = _mm256_hypot_ps(_source1, _source2);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End of hasAVX
							else if (tpa::has_SSE)
							{
								__m128 _source1, _source2, _dest;

								for (; (i + 4uz) < end; i += 4uz)
								{
									//Set Values
									_source1 = _mm_load_ps(&source1[i]);
									_source2 = _mm_load_ps(&source2[i]);
									
									//hypot
									_dest = _mm_hypot_ps(_source1, _source2);

									//Store Result
									_mm_store_ps(&dest[i], _dest);
								}//End for
							}//End of has_SSE
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() && std::is_same<T2, double>() && std::is_same<RES, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _source1, _source2, _dest;

								for (; (i + 8uz) < end; i += 8uz)
								{
									//Set Values
									_source1 = _mm512_load_pd(&source1[i]);
									_source2 = _mm512_load_pd(&source2[i]);
									
									//hypot
									_dest = _mm512_hypot_pd(_source1, _source2);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End of hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _source1, _source2, _dest;

								for (; (i + 4uz) < end; i += 4uz)
								{
									//Set Values
									_source1 = _mm256_load_pd(&source1[i]);
									_source2 = _mm256_load_pd(&source2[i]);
									
									//hypot
									_dest = _mm256_hypot_pd(_source1, _source2);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End of hasAVX
							else if (tpa::has_SSE2)
							{
								__m128d _source1, _source2, _dest;

								for (; (i + 2uz) < end; i += 2uz)
								{
									//Set Values
									_source1 = _mm_load_pd(&source1[i]);
									_source2 = _mm_load_pd(&source2[i]);
									
									//hypot
									_dest = _mm_hypot_pd(_source1, _source2);

									//Store Result
									_mm_store_pd(&dest[i], _dest);
								}//End for
							}//End of has_SSE2
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							dest[i] = static_cast<RES>(std::hypot(source1[i], source2[i]));
						}//End for
#pragma endregion

						return static_cast<uint32_t>(1);
					});//End of lambda

				results.emplace_back(std::move(temp));
			}//End for

			for (const auto& fut : results)
			{
				complete += fut.get();
			}//End for			

			//Check all threads completed
			if (complete != nThreads)
			{
				throw tpa::exceptions::NotAllThreadsCompleted(complete);
			}//End if

		}//End try
		catch (const tpa::exceptions::ArrayTooSmall& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::hypot: " << ex.what() << "\n";
			std::cerr << "tpa::simd::hypot will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::hypot: " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::hypot: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::hypot: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::hypot: unknown!\n";
		}//End catch
	}//End of hypot()
#pragma endregion
}//End of namespace
