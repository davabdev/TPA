#pragma once
/*
*	Vectorized Static Cast for TPA Library
*	By: David Aaron Braun
*	2021-10-05
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
#include <vector>
#include <array>
#include <cmath>
#include <numbers>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#include <float.h>
#if defined(_M_FP_PRECISE) || defined(_M_FP_STRICT) 
#pragma fenv_access (on)
#endif
#else
#pragma STDC FENV_ACCESS ON
#endif

#include "../_util.hpp"
#include "../ThreadPool.hpp"
#include "../excepts.hpp"
#include "../size_t_lit.hpp"
#include "../tpa_macros.hpp"
#include "../predicates.hpp"
#include "../tpa_concepts.hpp"
#include "../InstructionSet.hpp"
#include "simd.hpp"

#undef floor
#undef ceil
#undef round

namespace tpa{

	/// <summary>
	/// <para>Vectorized implementation of static_cast.</para>
	/// <para>Converts data in the 'source' container to the value_type of the 'dest' container </para>
	/// <para>and stores the converted data in the 'dest' container.</para>
	/// <para>When converting floating-point numbers to ints, this function uses the current rounding mode.</para>
	/// <para>To convert floats with a specific rounding mode use 'tpa::simd::round' or 'tpa::simd::floor' or tpa::simd::ceil'</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	template<typename CONTAINER_A, typename CONTAINER_B>
	inline constexpr void static_convert(
		const CONTAINER_A& source,
		CONTAINER_B& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
	tpa::util::contiguous_seqeunce<CONTAINER_B>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			using FROM_T = CONTAINER_A::value_type;
			using TO_T = CONTAINER_B::value_type;

			static_assert(!std::is_same<FROM_T, TO_T>(), "The 'source' and 'dest' containers are of the same value_type. Conversion is not necessary. If you wish to perform a copy, use 'tpa::copy'.");

			//Determin the smallest container
			smallest = source.size();

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

#pragma region int8-to-int16
					if constexpr (std::is_same<FROM_T, int8_t>() && std::is_same<TO_T, int16_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_ByteWord)
						{
							__m256i _from;
							__m512i _to;

							for (; (i + 32uz) < end; i += 32uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepi8_epi16(_from);

								_mm512_storeu_epi16(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm_load_si128((__m128i*) &source[i]);
								_to = _mm256_cvtepi8_epi16(_from);

								_mm256_store_si256((__m256i*) &dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi8_epi16(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int8-to-int32
					if constexpr (std::is_same<FROM_T, int8_t>() && std::is_same<TO_T, int32_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _from;
							__m512i _to;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm512_cvtepi8_epi32(_from);

								_mm512_store_epi32(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepi8_epi32(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi8_epi32(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int8-to-int64
					if constexpr (std::is_same<FROM_T, int8_t>() && std::is_same<TO_T, int64_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _from;
							__m512i _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm512_cvtepi8_epi64(_from);

								_mm512_store_epi64(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepi8_epi64(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi8_epi64(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int8-to-float32
					if constexpr (std::is_same<FROM_T, int8_t>() && std::is_same<TO_T, float>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _from;
							__m512i _to;
							__m512 _tof;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm512_cvtepi8_epi32(_from);
								_tof = _mm512_cvtepi32_ps(_to);

								_mm512_store_ps(&dest[i], _tof);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;
							__m256 _tof;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepi8_epi32(_from);
								_tof = _mm256_cvtepi32_ps(_to);

								_mm256_store_ps(&dest[i], _tof);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;
							__m128 _tof;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi8_epi32(_from);
								_tof = _mm_cvtepi32_ps(_to);

								_mm_store_ps(&dest[i], _tof);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int8-to-float64
					if constexpr (std::is_same<FROM_T, int8_t>() && std::is_same<TO_T, double>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _from;
							__m512i _to;
							__m256i _first, _second;
							__m512d _tof;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm512_cvtepi8_epi32(_from);

								_first = _mm256_load_si256((__m256i*) &_to);
								_second = _mm256_load_si256((__m256i*) &_to + 32);

								_tof = _mm512_cvtepi32_pd(_first);
								_mm512_store_pd(&dest[i], _tof);

								_tof = _mm512_cvtepi32_pd(_second);
								_mm512_store_pd(&dest[i+8], _tof);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from, _first, _second;
							__m256i _to;
							__m256d _tof;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) &source[i]);
								_to = _mm256_cvtepi8_epi32(_from);

								_first = _mm_load_si128((__m128i*) &_to);
								_second = _mm_load_si128((__m128i*) &_to + 16);

								_tof = _mm256_cvtepi32_pd(_first);
								_mm256_store_pd(&dest[i], _tof);

								_tof = _mm256_cvtepi32_pd(_second);
								_mm256_store_pd(&dest[i+4], _tof);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to, _low, _high;
							__m128d _tof;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi8_epi32(_from);

								_low = _mm_load_si128((__m128i*) & _to);
								_high = _mm_load_si128((__m128i*) & _to + 8);

								_tof = _mm_cvtepi32_pd(_low);
								_mm_store_pd(&dest[i], _tof);

								_tof = _mm_cvtepi32_pd(_high);
								_mm_store_pd(&dest[i+2], _tof);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint8-to-int16
					if constexpr (std::is_same<FROM_T, uint8_t>() && (std::is_same<TO_T, int16_t>() || std::is_same<TO_T, uint16_t>()))
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_ByteWord)
						{
							__m256i _from;
							__m512i _to;

							for (; (i + 32uz) < end; i += 32uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepu8_epi16(_from);

								_mm512_storeu_epi16(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepu8_epi16(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepu8_epi16(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint8-to-int32
					if constexpr (std::is_same<FROM_T, int8_t>() && (std::is_same<TO_T, int32_t>() || std::is_same<TO_T, uint32_t>()))
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _from;
							__m512i _to;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm512_cvtepu8_epi32(_from);

								_mm512_store_epi32(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepu8_epi32(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepu8_epi32(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint8-to-int64
					if constexpr (std::is_same<FROM_T, int8_t>() && (std::is_same<TO_T, int64_t>() || std::is_same<TO_T, uint64_t>()))
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _from;
							__m512i _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm512_cvtepu8_epi64(_from);

								_mm512_store_epi64(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepu8_epi64(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepu8_epi64(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint8-to-float32
					if constexpr (std::is_same<FROM_T, uint8_t>() && std::is_same<TO_T, float>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _from;
							__m512i _to;
							__m512 _tof;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm512_cvtepu8_epi32(_from);
								_tof = _mm512_cvtepi32_ps(_to);

								_mm512_store_ps(&dest[i], _tof);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;
							__m256 _tof;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepu8_epi32(_from);
								_tof = _mm256_cvtepi32_ps(_to);

								_mm256_store_ps(&dest[i], _tof);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;
							__m128 _tof;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepu8_epi32(_from);
								_tof = _mm_cvtepi32_ps(_to);

								_mm_store_ps(&dest[i], _tof);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint8-to-float64
					if constexpr (std::is_same<FROM_T, uint8_t>() && std::is_same<TO_T, double>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _from;
							__m512i _to;
							__m256i _first, _second;
							__m512d _tof;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm512_cvtepu8_epi32(_from);

								_first = _mm256_load_si256((__m256i*) & _to);
								_second = _mm256_load_si256((__m256i*) & _to + 32);

								_tof = _mm512_cvtepi32_pd(_first);
								_mm512_store_pd(&dest[i], _tof);

								_tof = _mm512_cvtepi32_pd(_second);
								_mm512_store_pd(&dest[i + 8], _tof);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from, _first, _second;
							__m256i _to;
							__m256d _tof;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepu8_epi32(_from);

								_first = _mm_load_si128((__m128i*) & _to);
								_second = _mm_load_si128((__m128i*) & _to + 16);

								_tof = _mm256_cvtepi32_pd(_first);
								_mm256_store_pd(&dest[i], _tof);

								_tof = _mm256_cvtepi32_pd(_second);
								_mm256_store_pd(&dest[i + 4], _tof);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to, _low, _high;
							__m128d _tof;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepu8_epi32(_from);

								_low = _mm_load_si128((__m128i*) & _to);
								_high = _mm_load_si128((__m128i*) & _to + 8);

								_tof = _mm_cvtepi32_pd(_low);
								_mm_store_pd(&dest[i], _tof);

								_tof = _mm_cvtepi32_pd(_high);
								_mm_store_pd(&dest[i + 2], _tof);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int16-to-int8
					if constexpr (std::is_same<FROM_T, int16_t>() && std::is_same<TO_T, int8_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_ByteWord)
						{
							__m256i _to;
							__m512i _from;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm512_loadu_epi16(&source[i]);
								_to = _mm512_cvtepi16_epi8(_from);

								_mm256_store_si256((__m256*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int16-to-int32
					if constexpr (std::is_same<FROM_T, int16_t>() && std::is_same<TO_T, int32_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _from;
							__m512i _to;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepi16_epi32(_from);

								_mm512_store_epi32(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepi16_epi32(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi16_epi32(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int16-to-int64
					if constexpr (std::is_same<FROM_T, int16_t>() && std::is_same<TO_T, int64_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _from;
							__m512i _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm512_cvtepi16_epi64(_from);

								_mm512_store_epi64(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepi16_epi64(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi16_epi64(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int16-to-float32
					if constexpr (std::is_same<FROM_T, int16_t>() && std::is_same<TO_T, float>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _from;
							__m512i _to;
							__m512 _tof;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepi16_epi32(_from);
								_tof = _mm512_cvtepi32_ps(_to);

								_mm512_store_ps(&dest[i], _tof);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;
							__m256 _tof;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepi16_epi32(_from);
								_tof = _mm256_cvtepi32_ps(_to);

								_mm256_store_ps(&dest[i], _tof);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;
							__m128 _tof;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi16_epi32(_from);
								_tof = _mm_cvtepi32_ps(_to);

								_mm_store_ps(&dest[i], _tof);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int16-to-float64
					if constexpr (std::is_same<FROM_T, int16_t>() && std::is_same<TO_T, double>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _from, _first, _second;
							__m512i _to;
							__m512d _tof;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepi16_epi32(_from);

								_first = _mm256_load_si256((__m256i*) &_to);
								_tof = _mm512_cvtepi32_pd(_first);
								_mm512_store_pd(&dest[i], _tof);

								_second = _mm256_load_si256((__m256i*) &_to + 32);
								_tof = _mm512_cvtepi32_pd(_first);
								_mm512_store_pd(&dest[i+8], _tof);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from, _first, _second;
							__m256i _to;
							__m256d _tof;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepi16_epi32(_from);

								_first = _mm_load_si128((__m128i*) &_to);
								_tof = _mm256_cvtepi32_pd(_first);
								_mm256_store_pd(&dest[i], _tof);

								_second = _mm_load_si128((__m128i*) &_to + 16);
								_tof = _mm256_cvtepi32_pd(_first);
								_mm256_store_pd(&dest[i+4], _tof);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;
							__m128d _tof;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi16_epi32(_from);
								_tof = _mm_cvtepi32_pd(_to);

								_mm_store_pd(&dest[i], _tof);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint16-to-int32
					if constexpr (std::is_same<FROM_T, uint16_t>() && (std::is_same<TO_T, int32_t>() || std::is_same<TO_T, uint32_t>()))
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _from;
							__m512i _to;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepu16_epi32(_from);

								_mm512_store_epi32(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepu16_epi32(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepu16_epi32(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint16-to-int64
					if constexpr (std::is_same<FROM_T, uint16_t>() && (std::is_same<TO_T, int64_t>() || std::is_same<TO_T, uint64_t>()))
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _from;
							__m512i _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm512_cvtepu16_epi64(_from);

								_mm512_store_epi64(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepu16_epi64(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepu8_epi64(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint16-to-float32
					if constexpr (std::is_same<FROM_T, uint16_t>() && std::is_same<TO_T, float>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _from;
							__m512i _to;
							__m512 _tof;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepu16_epi32(_from);
								_tof = _mm512_cvtepi32_ps(_to);

								_mm512_store_ps(&dest[i], _tof);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;
							__m256 _tof;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepu16_epi32(_from);
								_tof = _mm256_cvtepi32_ps(_to);

								_mm256_store_ps(&dest[i], _tof);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;
							__m128 _tof;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepu16_epi32(_from);
								_tof = _mm_cvtepi32_ps(_to);

								_mm_store_ps(&dest[i], _tof);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint16-to-float64
					if constexpr (std::is_same<FROM_T, uint16_t>() && std::is_same<TO_T, double>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _from, _first, _second;
							__m512i _to;
							__m512d _tof;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepu16_epi32(_from);

								_first = _mm256_load_si256((__m256i*) & _to);
								_tof = _mm512_cvtepi32_pd(_first);
								_mm512_store_pd(&dest[i], _tof);

								_second = _mm256_load_si256((__m256i*) & _to + 32);
								_tof = _mm512_cvtepi32_pd(_first);
								_mm512_store_pd(&dest[i + 8], _tof);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from, _first, _second;
							__m256i _to;
							__m256d _tof;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepu16_epi32(_from);

								_first = _mm_load_si128((__m128i*) & _to);
								_tof = _mm256_cvtepi32_pd(_first);
								_mm256_store_pd(&dest[i], _tof);

								_second = _mm_load_si128((__m128i*) & _to + 16);
								_tof = _mm256_cvtepi32_pd(_first);
								_mm256_store_pd(&dest[i + 4], _tof);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;
							__m128d _tof;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepu16_epi32(_from);
								_tof = _mm_cvtepi32_pd(_to);

								_mm_store_pd(&dest[i], _tof);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int32-to-int8
					if constexpr (std::is_same<FROM_T, int32_t>() && std::is_same<TO_T, int8_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _to;
							__m512i _from;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm512_load_epi32(&source[i]);
								_to = _mm512_cvtepi32_epi8(_from);

								_mm_store_si128((__m128*) &dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int32-to-int16
					if constexpr (std::is_same<FROM_T, int32_t>() && std::is_same<TO_T, int16_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _to;
							__m512i _from;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm512_load_epi32(&source[i]);
								_to = _mm512_cvtepi32_epi16(_from);

								_mm256_store_si256((__m256*) &dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int32-to-int64
					if constexpr (std::is_same<FROM_T, int32_t>() && std::is_same<TO_T, int64_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _from;
							__m512i _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepi32_epi64(_from);

								_mm512_store_epi64(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepi32_epi64(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi32_epi64(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int32-to-float32
					if constexpr (std::is_same<FROM_T, int32_t>() && std::is_same<TO_T, float>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m512i _from;
							__m512 _to;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm512_load_epi32(&source[i]);
								_to = _mm512_cvtepi32_ps(_from);

								_mm512_store_ps(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m256i _from;
							__m256 _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm256_cvtepi32_ps(_from);

								_mm256_store_ps(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE2)
						{
							__m128i _from;
							__m128 _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi32_ps(_from);

								_mm_store_ps(&dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int32-to-float64
					if constexpr (std::is_same<FROM_T, int32_t>() && std::is_same<TO_T, double>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _from;
							__m512d _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepi32_pd(_from);

								_mm512_store_pd(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256d _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepi32_pd(_from);

								_mm256_store_pd(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE2)
						{
							__m128i _from;
							__m128d _to;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi32_pd(_from);

								_mm_store_pd(&dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint32-to-int64
					if constexpr (std::is_same<FROM_T, uint32_t>() && std::is_same<TO_T, int64_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _from;
							__m512i _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepu32_epi64(_from);

								_mm512_store_epi64(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256i _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepu32_epi64(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE41)
						{
							__m128i _from, _to;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepu32_epi64(_from);

								_mm_store_si128((__m128i*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint32-to-float32
					if constexpr (std::is_same<FROM_T, uint32_t>() && std::is_same<TO_T, float>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m512i _from;
							__m512 _to;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm512_load_epi32(&source[i]);
								_to = _mm512_cvtepu32_ps(_from);

								_mm512_store_ps(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m256i _from;
							__m256 _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm256_cvtepi32_ps(_from);

								_mm256_store_ps(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE2)
						{
							__m128i _from;
							__m128 _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi32_ps(_from);

								_mm_store_ps(&dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint32-to-float64
					if constexpr (std::is_same<FROM_T, uint32_t>() && std::is_same<TO_T, double>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _from;
							__m512d _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = _mm512_cvtepu32_pd(_from);

								_mm512_store_pd(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m128i _from;
							__m256d _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm256_cvtepi32_pd(_from);

								_mm256_store_pd(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE2)
						{
							__m128i _from;
							__m128d _to;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = _mm_cvtepi32_pd(_from);

								_mm_store_pd(&dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int64-to-int8
					if constexpr (std::is_same<FROM_T, int64_t>() && std::is_same<TO_T, int8_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _to;
							__m512i _from;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm512_load_epi64(&source[i]);
								_to = _mm512_cvtepi64_epi8(_from);

								_mm_store_si128((__m128*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int64-to-int16
					if constexpr (std::is_same<FROM_T, int64_t>() && std::is_same<TO_T, int16_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m128i _to;
							__m512i _from;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm512_load_epi32(&source[i]);
								_to = _mm512_cvtepi64_epi16(_from);

								_mm_store_si128((__m128*) &dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region int64-to-int32
					if constexpr (std::is_same<FROM_T, int64_t>() && std::is_same<TO_T, int32_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m256i _to;
							__m512i _from;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm512_load_epi64(&source[i]);
								_to = _mm512_cvtepi64_epi32(_from);

								_mm512_store_epi32(&dest[i], _to);
							}//End for
						}//End if						
#endif
					}//End if
#pragma endregion
#pragma region int64-to-float32
					if constexpr (std::is_same<FROM_T, int64_t>() && std::is_same<TO_T, float>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_DWQW)
						{
							__m512i _from;
							__m256 _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm512_load_epi64(&source[i]);
								_to = _mm512_cvtepi64_ps(_from);

								_mm512_store_ps(&dest[i], _to);
							}//End for
						}//End if						
#endif
					}//End if
#pragma endregion
#pragma region int64-to-float64
					if constexpr (std::is_same<FROM_T, int64_t>() && std::is_same<TO_T, double>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_DWQW)
						{
							__m512i _from;
							__m512d _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm512_load_epi64(&source[i]);
								_to = _mm512_cvtepi64_pd(_from);

								_mm512_store_pd(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m256i _from;
							__m256d _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = tpa::simd::_mm256_cvtepi64_pd(_from);

								_mm256_store_pd(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE2)
						{
							__m128i _from;
							__m128d _to;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = tpa::simd::_mm_cvtepi64_pd(_from);

								_mm_store_pd(&dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region uint64-to-float64
					if constexpr (std::is_same<FROM_T, uint64_t>() && std::is_same<TO_T, double>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_DWQW)
						{
							__m512i _from;
							__m512d _to;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm512_load_epi64(&source[i]);
								_to = _mm512_cvtepu64_pd(_from);

								_mm512_store_pd(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m256i _from;
							__m256d _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm256_load_si256((__m256i*) & source[i]);
								_to = tpa::simd::_mm256_cvtepu64_pd(_from);

								_mm256_store_pd(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE2)
						{
							__m128i _from;
							__m128d _to;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_si128((__m128i*) & source[i]);
								_to = tpa::simd::_mm_cvtepu64_pd(_from);

								_mm_store_pd(&dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region float32-to-int32
					if constexpr (std::is_same<FROM_T, float>() && std::is_same<TO_T, int32_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m512i _to;
							__m512 _from;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm512_load_ps(&source[i]);
								_to = _mm512_cvtps_epi32(_from);

								_mm512_store_epi32(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX)
						{
							__m256i _to;
							__m256 _from;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm256_load_ps(&source[i]);
								_to = _mm256_cvtps_epi32(_from);

								_mm256_store_si256((__m256i*)& dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE2)
						{
							__m128 _from;
							__m128i _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_ps(&source[i]);
								_to = _mm_cvtps_epi32(_from);

								_mm_store_si128((__m128i*) &dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region float32-to-uint32
					if constexpr (std::is_same<FROM_T, float>() && std::is_same<TO_T, int64_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m512i _to;
							__m512 _from;

							for (; (i + 16uz) < end; i += 16uz)
							{
								_from = _mm512_load_ps(&source[i]);
								_to = _mm512_cvtps_epu32(_from);

								_mm512_store_epi32(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX)
						{
							__m256i _to;
							__m256 _from;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm256_load_ps(&source[i]);
								_to = _mm256_cvtps_epi32(_from);

								_mm256_store_si256((__m256i*) & dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE2)
						{
							__m128i _from;
							__m128d _to;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm_load_ps(&source[i]);
								_to = _mm_cvtps_epi32(_from);

								_mm_store_si128((__m128*) & dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region float64-to-int64
					if constexpr (std::is_same<FROM_T, double>() && std::is_same<TO_T, int64_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_DWQW)
						{
							__m512i _to;
							__m512d _from;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm512_load_pd(&source[i]);
								_to = _mm512_cvtpd_epi64(_from);

								_mm512_store_epi64(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m256i _to;
							__m256d _from;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm256_load_pd(& source[i]);
								_to = tpa::simd::_mm256_cvtpd_epi64(_from);

								_mm256_store_si256((__m256i*)& dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE2)
						{
							__m128i _to;
							__m128d _from;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_pd(&source[i]);
								_to = tpa::simd::_mm_cvtpd_epi64(_from);

								_mm_store_si128((__m128i*)& dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion
#pragma region float64-to-uint64
					if constexpr (std::is_same<FROM_T, double>() && std::is_same<TO_T, uint64_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_DWQW)
						{
							__m512i _to;
							__m512d _from;

							for (; (i + 8uz) < end; i += 8uz)
							{
								_from = _mm512_load_pd(&source[i]);
								_to = _mm512_cvtpd_epu64(_from);

								_mm512_store_epi64(&dest[i], _to);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m256i _to;
							__m256d _from;

							for (; (i + 4uz) < end; i += 4uz)
							{
								_from = _mm256_load_pd(&source[i]);
								_to = tpa::simd::_mm256_cvtpd_epu64(_from);

								_mm256_store_si256((__m256i*) &dest[i], _to);
							}//End for
						}//End if
						else if (tpa::has_SSE2)
						{
							__m128i _to;
							__m128d _from;

							for (; (i + 2uz) < end; i += 2uz)
							{
								_from = _mm_load_pd(&source[i]);
								_to = tpa::simd::_mm_cvtpd_epu64(_from);

								_mm_store_si128((__m128i*) &dest[i], _to);
							}//End for
						}//End if
#endif
					}//End if
#pragma endregion

#pragma region generic
					for (; i < end; ++i)
					{
						dest[i] = static_cast<TO_T>(source[i]);
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
			std::cerr << "Exception thrown in tpa::simd::static_convert: " << ex.what() << "\n";
			std::cerr << "tpa::simd::static_convert will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::static_convert(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::static_convert: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::static_convert: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::static_convert: unknown!\n";
		}//End catch
	}//End of static_convert()
};//End of namespace
