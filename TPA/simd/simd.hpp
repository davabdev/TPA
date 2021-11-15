#pragma once
/*
*	Matrix SIMD functions for TPA Library
*	By: David Aaron Braun
*	2021-04-12
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
	/// <para>Computes matrix math on 2 aligned containers storing the result in a 3rd aligned container.</para> 
	/// <para> Containers of different types are allowed but not recomended.</para>
	/// <para> Containers of different value types are NOT allowed</para>
	/// <para> Containers do not have to be a particular size</para>
	/// <para> If passing 2 containers of different sizes, values will only be calculated up to the container with the smallest size, the destination container must be at least this large.</para> 
	/// <para>Templated predicate takes 1 of these predicates: tpa::op</para>
	/// <para>---------------------------------------------------</para>
	///<para>tpa::op::ADD< / para>
	/// <para>tpa::op::SUBRACT</para>		
	/// <para>tpa::op::MULTIPLY</para>		
	/// <para>tpa::op::DIVIDE</para>
	/// <para>tpa::op::MODULO</para>
	/// <para>tpa::op::MIN</para>
	/// <para>tpa::op::MAX</para>
	/// <para>tpa::op::POW</para>
	/// <para>tpa::op::AVERAGE</para>
	/// </summary>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::op INSTR, typename CONTAINER_A, typename CONTAINER_B, typename CONTAINER_C>
	inline constexpr void calculate(
		const CONTAINER_A& source1,
		const CONTAINER_B& source2,
		CONTAINER_C& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A> &&
		tpa::util::contiguous_seqeunce<CONTAINER_B> &&
		tpa::util::contiguous_seqeunce<CONTAINER_C>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, CONTAINER_B::value_type>() &&
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>(),
				"Compile Error! The source and destination vectors must be of the same value type!");

			using T = CONTAINER_A::value_type;

			//Determin the smallest container
			smallest = tpa::util::min(source1.size(), source2.size());

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
#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+64) < end; i += 64)
								{
									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi8((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[64] = {};
										for (size_t x = 0; x != 64; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m512i_i8[x] * _Bi.m512i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm512_loadu_epi8((__m512i*)&d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi8(2);

										_DESTi = _mm512_add_epi8(_Ai, _Bi);
										_DESTi = _mm512_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[32] = {};
										for (size_t x = 0; x != 32; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m256i_i8[x] * _Bi.m256i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
									}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);
								}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi8(2);

										_DESTi = _mm256_add_epi8(_Ai, _Bi);
										_DESTi = _mm256_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[16] = {};
										for (size_t x = 0; x != 16; ++x)
										{
	#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m128i_i8[x] * _Bi.m128i_i8[x]);
	#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
	#endif
										}//End for
										_DESTi = _mm_load_si128((__m128i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_min_epi8(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_max_epi8(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi8(2);

										_DESTi = _mm_add_epi8(_Ai, _Bi);
										_DESTi = _mm_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif
						}//End if

#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+64) < end; i += 64)
								{
									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi8((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[64] = {};
										for (size_t x = 0; x != 64; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m512i_u8[x] * _Bi.m512i_u8[x]);
#else
											d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm512_loadu_epi8((__m512i*)&d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm512_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[32] = {};
										for (size_t x = 0; x != 32; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m256i_u8[x] * _Bi.m256i_u8[x]);
#else
											d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm256_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[16] = {};
										for (size_t x = 0; x != 16; ++x)
										{
	#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m128i_i8[x] * _Bi.m128i_i8[x]);
	#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
	#endif
										}//End for
										_DESTi = _mm_load_si128((__m128i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif				
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi16(2);

										_DESTi = _mm512_add_epi16(_Ai, _Bi);
										_DESTi = _mm512_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi16(2);

										_DESTi = _mm256_add_epi16(_Ai, _Bi);
										_DESTi = _mm256_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi16(2);

										_DESTi = _mm_add_epi16(_Ai, _Bi);
										_DESTi = _mm_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm512_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
									}();
								}//End else

								//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
							}//End for
						}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm256_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_min_epu16(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_max_epu16(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif 							
			}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm512_add_epi32(_Ai, _Bi);
										_DESTi = _mm512_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm256_add_epi32(_Ai, _Bi);
										_DESTi = _mm256_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = tpa::util::_mm_mul_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_min_epi32(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_max_epi32(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi32(2);

										_DESTi = _mm_add_epi32(_Ai, _Bi);
										_DESTi = _mm_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi32(2);

										_DESTi = _mm512_add_epi32(_Ai, _Bi);
										_DESTi = _mm512_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm256_add_epi32(_Ai, _Bi);
										_DESTi = _mm256_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm_mul_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_min_epu32(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_max_epu32(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi32(2);

										_DESTi = _mm_add_epi32(_Ai, _Bi);
										_DESTi = _mm_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
#ifdef __AVX512DQ__
										_DESTi = _mm512_mullo_epi64(_Ai, _Bi);
#else
										_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
#endif
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi64(2);

										_DESTi = _mm512_add_epi64(_Ai, _Bi);
										_DESTi = _mm512_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{										
										_DESTi = tpa::util::_mm256_mul_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi64x(2);

										_DESTi = _mm256_add_epi64(_Ai, _Bi);
										_DESTi = _mm256_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = tpa::util::_mm_mul_epi64(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi64x(2);

										_DESTi = _mm_add_epi64(_Ai, _Bi);
										_DESTi = _mm_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
#ifdef __AVX512DQ__
										_DESTi = _mm512_mullo_epi64(_Ai, _Bi);
#else
										_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
#endif
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi64(2);

										_DESTi = _mm512_add_epi64(_Ai, _Bi);
										_DESTi = _mm512_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = tpa::util::_mm256_mul_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::BIT_AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::BIT_OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::BIT_XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::BIT_AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi64x(2);

										_DESTi = _mm256_add_epi64(_Ai, _Bi);
										_DESTi = _mm256_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = tpa::util::_mm_mul_epi64(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epu64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi64x(2);

										_DESTi = _mm_add_epi64(_Ai, _Bi);
										_DESTi = _mm_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);
									_Bi = _mm512_load_ps(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if		
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512 _TWO = _mm512_set1_ps(2.0f);

										_DESTi = _mm512_add_ps(_Ai, _Bi);
										_DESTi = _mm512_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm512_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm512_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);
									_Bi = _mm256_load_ps(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if	
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256 _TWO = _mm256_set1_ps(2.0f);

										_DESTi = _mm256_add_ps(_Ai, _Bi);
										_DESTi = _mm256_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm256_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm256_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE)
							{
								__m128 _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_ps(&source1[i]);
									_Bi = _mm_load_ps(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm_mul_ps(_Ai, _Bi);						
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break;
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_ps(2.0f);

										_DESTi = _mm_add_ps(_Ai, _Bi);
										_DESTi = _mm_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if has_SSE
#endif						
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);
									_Bi = _mm512_load_pd(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512d _TWO = _mm512_set1_pd(2.0);

										_DESTi = _mm512_add_pd(_Ai, _Bi);
										_DESTi = _mm512_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm512_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm512_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _Bi, _DESTi;

								for (; (i+4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);
									_Bi = _mm256_load_pd(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256d _TWO = _mm256_set1_pd(2.0);

										_DESTi = _mm256_add_pd(_Ai, _Bi);
										_DESTi = _mm256_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm256_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm256_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE2)
							{
								__m128d _Ai, _Bi, _DESTi;

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_pd(&source1[i]);
									_Bi = _mm_load_pd(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break;
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_pd(2.0);

										_DESTi = _mm_add_pd(_Ai, _Bi);
										_DESTi = _mm_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif						
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Calc
							if constexpr (INSTR == tpa::op::ADD)
							{
								dest[i] = source1[i] + source2[i];
							}//End if
							else if constexpr (INSTR == tpa::op::SUBTRACT)
							{
								dest[i] = source1[i] - source2[i];
							}//End if
							else if constexpr (INSTR == tpa::op::MULTIPLY)
							{
								dest[i] = source1[i] * source2[i];
							}//End if
							else if constexpr (INSTR == tpa::op::DIVIDE)
							{
								dest[i] = source1[i] / source2[i];
							}//End if
							else if constexpr (INSTR == tpa::op::MODULO)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = static_cast<T>(std::fmod(source1[i], source2[i]));
								}//End if
								else
								{
									dest[i] = source1[i] % source2[i];
								}//End else
							}//end if							
							else if constexpr (INSTR == tpa::op::MIN)
							{
								dest[i] = tpa::util::min(source1[i], source2[i]);
							}//End if
							else if constexpr (INSTR == tpa::op::MAX)
							{
								dest[i] = tpa::util::max(source1[i], source2[i]);
							}//End if
							else if constexpr (INSTR == tpa::op::AVERAGE)
							{
								dest[i] = static_cast<T>((source1[i] + source2[i]) / 2);
							}//End if
							else if constexpr (INSTR == tpa::op::POWER)
							{
								dest[i] = tpa::util::pow(source1[i], source2[i]);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
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
			std::cerr << "Exception thrown in tpa::simd::calculate(): " << ex.what() << "\n";
			std::cerr << "tpa::simd::calculate will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: unknown!\n";
		}//End catch
	}//End of calculate()

	/// <summary>
	/// <para>Computes matrix math on 2 aligned containers storing the result in a 3rd aligned container.</para> 
	/// <para> Containers of different types are allowed but not recomended.</para>
	/// <para> Containers of different value types are NOT allowed</para>
	/// <para> Containers do not have to be a particular size</para>
	/// <para> If passing 2 containers of different sizes, values will only be calculated up to the container with the smallest size, the destination container must be at least this large.</para> 
	/// <para>Templated predicate takes 1 of these predicates: tpa::op</para>
	/// <para>---------------------------------------------------</para>
	///<para>tpa::op::ADD< / para>
	/// <para>tpa::op::SUBRACT</para>		
	/// <para>tpa::op::MULTIPLY</para>		
	/// <para>tpa::op::DIVIDE</para>
	/// <para>tpa::op::MODULO</para>
	/// <para>tpa::op::MIN</para>
	/// <para>tpa::op::MAX</para>
	/// <para>tpa::op::POW</para>
	/// <para>tpa::op::AVERAGE</para>
	/// </summary>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::op INSTR, typename CONTAINER_A, typename T, typename CONTAINER_C>
	inline constexpr void calculate_const(
		const CONTAINER_A& source1,
		const T val,
		CONTAINER_C& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_C>
	{
		uint32_t complete = 0;
		size_t smallest = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, T>() &&
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>(),
				"Compile Error! The source and destination vectors must be of the same value type!");
			smallest = source1.size();

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
				temp = tpa::tp->addTask([&source1, &val, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T _val = val;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi8(_val);

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[64] = {};
										for (size_t x = 0; x != 64; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m512i_i8[x] * _Bi.m512i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm512_loadu_epi8((__m512i*)&d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi8(2);

										_DESTi = _mm512_add_epi8(_Ai, _Bi);
										_DESTi = _mm512_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi8(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[32] = {};
										for (size_t x = 0; x != 32; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m256i_i8[x] * _Bi.m256i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi8(2);

										_DESTi = _mm256_add_epi8(_Ai, _Bi);
										_DESTi = _mm256_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif
						}//End if

#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi8(_val);

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[64] = {};
										for (size_t x = 0; x != 64; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m512i_u8[x] * _Bi.m512i_u8[x]);
#else
											d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm512_loadu_epi8((__m512i*)&d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm512_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi8(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[32] = {};
										for (size_t x = 0; x != 32; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m256i_u8[x] * _Bi.m256i_u8[x]);
#else
											d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm256_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX2
#endif				
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi16(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi16(2);

										_DESTi = _mm512_add_epi16(_Ai, _Bi);
										_DESTi = _mm512_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi16(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi16(2);

										_DESTi = _mm256_add_epi16(_Ai, _Bi);
										_DESTi = _mm256_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi16(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm512_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi16(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm256_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif 							
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi32(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm512_add_epi32(_Ai, _Bi);
										_DESTi = _mm512_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi32(_val);

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm256_add_epi32(_Ai, _Bi);
										_DESTi = _mm256_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi32(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi32(2);

										_DESTi = _mm512_add_epi32(_Ai, _Bi);
										_DESTi = _mm512_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi32(_val);

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm256_add_epi32(_Ai, _Bi);
										_DESTi = _mm256_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi64(_val);

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi64(2);

										_DESTi = _mm512_add_epi64(_Ai, _Bi);
										_DESTi = _mm512_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi64x(_val);

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint64_t d[4] = {};
										for (size_t x = 0; x != 4; ++x)
										{
#if _WIN32
											d[x] = _Ai.m256i_i64[x] * _Bi.m256i_i64[x];
#else
											d[x] = _Ai[x] * _Bi[x];
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);

									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi64x(2);

										_DESTi = _mm256_add_epi64(_Ai, _Bi);
										_DESTi = _mm256_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi64(_val);

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi64(2);

										_DESTi = _mm512_add_epi64(_Ai, _Bi);
										_DESTi = _mm512_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi64x(_val);

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint64_t d[4] = {};
										for (size_t x = 0; x != 4; ++x)
										{
#ifdef _WIN32
											d[x] = _Ai.m256i_i64[x] * _Bi.m256i_i64[x];
#else
											d[x] = _Ai[x] * _Bi[x];
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);

									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi64x(2);

										_DESTi = _mm256_add_epi64(_Ai, _Bi);
										_DESTi = _mm256_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _Ai, _DESTi;
								const __m512 _Bi = _mm512_set1_ps(_val);

								for (; i != end; i += 16)
								{

									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if		
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512 _TWO = _mm512_set1_ps(2.0f);

										_DESTi = _mm512_add_ps(_Ai, _Bi);
										_DESTi = _mm512_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm512_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm512_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _DESTi;
								const __m256 _Bi = _mm256_set1_ps(_val);

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if		
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256 _TWO = _mm256_set1_ps(2.0f);

										_DESTi = _mm256_add_ps(_Ai, _Bi);
										_DESTi = _mm256_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm256_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm256_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif						
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _Ai, _DESTi;
								const __m512d _Bi = _mm512_set1_pd(_val);

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512d _TWO = _mm512_set1_pd(2);

										_DESTi = _mm512_add_pd(_Ai, _Bi);
										_DESTi = _mm512_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm512_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm512_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _DESTi;
								const __m256d _Bi = _mm256_set1_pd(_val);

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256d _TWO = _mm256_set1_pd(2);

										_DESTi = _mm256_add_pd(_Ai, _Bi);
										_DESTi = _mm256_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm256_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm256_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif						
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Calc
							if constexpr (INSTR == tpa::op::ADD)
							{
								dest[i] = source1[i] + _val;
							}//End if
							else if constexpr (INSTR == tpa::op::SUBTRACT)
							{
								dest[i] = source1[i] - _val;
							}//End if
							else if constexpr (INSTR == tpa::op::MULTIPLY)
							{
								dest[i] = source1[i] * _val;
							}//End if
							else if constexpr (INSTR == tpa::op::DIVIDE)
							{
								dest[i] = source1[i] / _val;
							}//End if
							else if constexpr (INSTR == tpa::op::MODULO)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = static_cast<T>(std::fmod(source1[i], _val));
								}//End if
								else
								{
									dest[i] = source1[i] % _val;
								}//End else
							}//end if							
							else if constexpr (INSTR == tpa::op::MIN)
							{
								dest[i] = tpa::util::min(source1[i], _val);
							}//End if
							else if constexpr (INSTR == tpa::op::MAX)
							{
								dest[i] = tpa::util::max(source1[i], _val);
							}//End if
							else if constexpr (INSTR == tpa::op::AVERAGE)
							{
								dest[i] = static_cast<T>((source1[i] + _val) / 2);
							}//End if
							else if constexpr (INSTR == tpa::op::POWER)
							{
								dest[i] = tpa::util::pow(source1[i], _val);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
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
			std::cerr << "Exception thrown in tpa::simd::calculate(): " << ex.code()
				<< " " << ex.what() << "\n";
			std::cerr << "tpa::simd::calculate will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate(): " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: unknown!\n";
		}//End catch
	}//End of calculate()


	/// <summary>
	/// <para>Compares 2 containers and stores the results of the expression in a 3rd container.</para>
	/// <para>Example: if source1[0] > source2[0]...dest[0] = source1[0] otherwise writes zeros.</para>
	/// <para>Templated predicate takes 1 of these predicates: tpa::comp</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::comp::EQUAL_TO</para>			
	/// <para>tpa::comp::NOT_EQUAL_TO</para>		
	/// <para>tpa::comp::LESS_THAN</para>		
	/// <para>tpa::comp::LESS_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::GREATER_THAN</para>			
	/// <para>tpa::comp::GREATER_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::MIN</para>
	/// <para>tpa::comp::MAX</para>
	/// </summary>
	/// <typeparam name="CONTAINER"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::comp INSTR, typename CONTAINER_A, typename CONTAINER_B, typename CONTAINER_C>
	inline constexpr void compare(
		const CONTAINER_A& source1,
		const CONTAINER_B& source2,
		CONTAINER_C& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A> && 
		tpa::util::contiguous_seqeunce<CONTAINER_B> &&
		tpa::util::contiguous_seqeunce<CONTAINER_C>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, CONTAINER_B::value_type>() &&
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>(),
				"Compile Error! The source and destination containers must be of the same value type!");

			using T = CONTAINER_A::value_type;

			//Determin the smallest container
			smallest = tpa::util::min(source1.size(), source2.size());

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
#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask64 _mask;

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8(&source1[i]);
									_Bi = _mm512_loadu_epi8(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask64 _mask;

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8(&source1[i]);
									_Bi = _mm512_loadu_epi8(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask32 _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16(&source1[i]);
									_Bi = _mm512_loadu_epi16(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask32 _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16(&source1[i]);
									_Bi = _mm512_loadu_epi16(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32(&source1[i]);
									_Bi = _mm512_load_epi32(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64	
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32(&source1[i]);
									_Bi = _mm512_load_epi32(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64(&source1[i]);
									_Bi = _mm512_load_epi64(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64	
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64(&source1[i]);
									_Bi = _mm512_load_epi64(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64			
							if (tpa::hasAVX512)
							{
								__m512 _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);
									_Bi = _mm512_load_ps(&source2[i]);
									_dest = _mm512_setzero_ps();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GT_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GE_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LT_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LE_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_EQ_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_ps(_Ai, _Bi);
										_mm512_store_ps(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_ps(_Ai, _Bi);
										_mm512_store_ps(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _Bi, _mask;

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);
									_Bi = _mm256_load_ps(&source2[i]);

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GT_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GE_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LT_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LE_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_EQ_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_mask = _mm256_min_ps(_Ai, _Bi);
										_mm256_store_ps(&dest[i], _mask);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_mask = _mm256_max_ps(_Ai, _Bi);
										_mm256_store_ps(&dest[i], _mask);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else
								}//End for
							}//End if has AVX2
#endif							
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);
									_Bi = _mm512_load_pd(&source2[i]);
									_dest = _mm512_setzero_pd();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GT_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GE_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LT_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LE_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_EQ_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_pd(_Ai, _Bi);
										_mm512_store_pd(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_pd(_Ai, _Bi);
										_mm512_store_pd(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _Bi, _mask;

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);
									_Bi = _mm256_load_pd(&source2[i]);

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GT_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GE_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LT_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LE_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_EQ_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_mask = _mm256_min_pd(_Ai, _Bi);
										_mm256_store_pd(&dest[i], _mask);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_mask = _mm256_max_pd(_Ai, _Bi);
										_mm256_store_pd(&dest[i], _mask);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								if (source1[i] > source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								if (source1[i] >= source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								if (source1[i] < source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								if (source1[i] <= source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								if (source1[i] == source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								if (source1[i] != source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								dest[i] = tpa::util::min(source1[i], source2[i]);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								dest[i] = tpa::util::max(source1[i], source2[i]);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
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
			std::cerr << "Exception thrown in tpa::simd::compare(): " << ex.what() << "\n";
			std::cerr << "tpa::simd::compare will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare: unknown!\n";
		}//End catch
	}//End of compare()

	/// <summary>
	/// <para>Compares 1 container and 1 constant then stores the results of the expression in a 2nd container.</para>
	/// <para>Example: if source1[0] > const_val...dest[0] = source1[0] otherwise writes zeros.</para>
	/// <para>Templated predicate takes 1 of these predicates: tpa::comp</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::comp::EQUAL_TO</para>			
	/// <para>tpa::comp::NOT_EQUAL_TO</para>		
	/// <para>tpa::comp::LESS_THAN</para>		
	/// <para>tpa::comp::LESS_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::GREATER_THAN</para>			
	/// <para>tpa::comp::GREATER_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::MIN</para>
	/// <para>tpa::comp::MAX</para>
	/// </summary>
	/// <typeparam name="CONTAINER"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::comp INSTR, typename CONTAINER_A, typename T, typename CONTAINER_C>
	inline constexpr void compare_const(
		const CONTAINER_A& source1,
		const T val,
		CONTAINER_C& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_C>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, T>() &&
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>(),
				"Compile Error! The source and destination containers must be of the same value type!");

			smallest = source1.size();

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
				temp = tpa::tp->addTask([&source1, &val, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T _val = val;//Copy for thread-locality

#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask64 _mask;

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8(&source1[i]);
									_Bi = _mm512_set1_epi8(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi8(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask64 _mask;

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8(&source1[i]);
									_Bi = _mm512_set1_epi8(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi8(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask32 _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16(&source1[i]);
									_Bi = _mm512_set1_epi16(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi16(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask32 _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16(&source1[i]);
									_Bi = _mm512_set1_epi16(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi16(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32(&source1[i]);
									_Bi = _mm512_set1_epi32(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi32(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64	
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32(&source1[i]);
									_Bi = _mm512_set1_epi32(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi32(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64(&source1[i]);
									_Bi = _mm512_set1_epi64(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi64x(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64	
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64(&source1[i]);
									_Bi = _mm512_set1_epi64(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi64x(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64			
							if (tpa::hasAVX512)
							{
								__m512 _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);
									_Bi = _mm512_set1_ps(_val);
									_dest = _mm512_setzero_ps();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GT_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GE_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LT_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LE_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_EQ_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_ps(_Ai, _Bi);
										_mm512_store_ps(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_ps(_Ai, _Bi);
										_mm512_store_ps(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _Bi, _mask;

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);
									_Bi = _mm256_set1_ps(_val);

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GT_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GE_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LT_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LE_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_EQ_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_mask = _mm256_min_ps(_Ai, _Bi);
										_mm256_store_ps(&dest[i], _mask);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_mask = _mm256_max_ps(_Ai, _Bi);
										_mm256_store_ps(&dest[i], _mask);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else
								}//End for
							}//End if has AVX2
#endif							
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);
									_Bi = _mm512_set1_pd(_val);
									_dest = _mm512_setzero_pd();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GT_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GE_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LT_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LE_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_EQ_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_pd(_Ai, _Bi);
										_mm512_store_pd(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_pd(_Ai, _Bi);
										_mm512_store_pd(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _Bi, _mask;

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);
									_Bi = _mm256_set1_pd(_val);

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GT_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GE_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LT_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LE_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_EQ_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_mask = _mm256_min_pd(_Ai, _Bi);
										_mm256_store_pd(&dest[i], _mask);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_mask = _mm256_max_pd(_Ai, _Bi);
										_mm256_store_pd(&dest[i], _mask);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								if (source1[i] > _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								if (source1[i] >= _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								if (source1[i] < _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								if (source1[i] <= _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								if (source1[i] == _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								if (source1[i] != _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								dest[i] = tpa::util::min(source1[i], _val);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								dest[i] = tpa::util::max(source1[i], _val);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
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
			std::cerr << "Exception thrown in tpa::compare_const: " << ex.what() << "\n";
			std::cerr << "tpa::simd::compare_const will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const: " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const: unknown!\n";
		}//End catch
	}//End of compare_const()
#pragma endregion
#pragma region array
	
	/// <summary>
	/// <para>Computes matrix math on 2 std::arrays storing the result in a 3rd std::array .</para> 
	/// <para> Arrays of different types are NOT allowed</para>
	/// <para> Arrays of different value types are NOT allowed</para>
	/// <para> Arrays do not have to be a particular size</para>
	/// <para> If passing 2 arrays of different sizes, values will only be calculated up to the array with the smallest size, the destination array must be at least this large</para> 
	/// <para>Templated predicate takes 1 of these predicates: tpa::op</para>
	/// <para>---------------------------------------------------</para>
	///<para>tpa::op::ADD< / para>
	/// <para>tpa::op::SUBRACT</para>		
	/// <para>tpa::op::MULTIPLY</para>		
	/// <para>tpa::op::DIVIDE</para>
	/// <para>tpa::op::MODULO</para>
	/// <para>tpa::op::MIN</para>
	/// <para>tpa::op::MAX</para>
	/// <para>tpa::op::POW</para>
	/// <para>tpa::op::AVERAGE</para>
	/// </summary>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::op INSTR, size_t SIZE1, size_t SIZE2, size_t SIZE3, typename T>
		inline constexpr void calculate(
			const std::array<T,SIZE1>& source1,
			const std::array<T,SIZE2>& source2,
			std::array<T,SIZE3>& dest)
	{
		try
		{
			uint32_t complete = 0;
			size_t smallest = 0;

			static_assert(SIZE1 <= SIZE3 && SIZE2 <= SIZE3,
				"Compile Error! The destination array is not large enough!");

			//Determin the smallest container
			smallest = tpa::util::min(source1.size(), source2.size());

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
#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i + 64) < end; i += 64)
								{
									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi8((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[64] = {};
										for (size_t x = 0; x != 64; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m512i_i8[x] * _Bi.m512i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm512_loadu_epi8((__m512i*)&d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi8(2);

										_DESTi = _mm512_add_epi8(_Ai, _Bi);
										_DESTi = _mm512_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
									}//End for
								}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i + 32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[32] = {};
										for (size_t x = 0; x != 32; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m256i_i8[x] * _Bi.m256i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi8(_Ai, _Bi);
								}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi8(2);

										_DESTi = _mm256_add_epi8(_Ai, _Bi);
										_DESTi = _mm256_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[16] = {};
										for (size_t x = 0; x != 16; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m128i_i8[x] * _Bi.m128i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm_load_si128((__m128i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_min_epi8(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_max_epi8(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi8(2);

										_DESTi = _mm_add_epi8(_Ai, _Bi);
										_DESTi = _mm_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
										}//End for
									}//End if has_SSE2
#endif
								}//End if

#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i + 64) < end; i += 64)
								{
									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi8((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[64] = {};
										for (size_t x = 0; x != 64; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m512i_u8[x] * _Bi.m512i_u8[x]);
#else
											d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm512_loadu_epi8((__m512i*)&d);
										}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm512_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
									}//End for
								}//End if has AVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i + 32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[32] = {};
										for (size_t x = 0; x != 32; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m256i_u8[x] * _Bi.m256i_u8[x]);
#else
											d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm256_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[16] = {};
										for (size_t x = 0; x != 16; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m128i_i8[x] * _Bi.m128i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm_load_si128((__m128i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif				
							}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i + 32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi16(2);

										_DESTi = _mm512_add_epi16(_Ai, _Bi);
										_DESTi = _mm512_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi16(2);

										_DESTi = _mm256_add_epi16(_Ai, _Bi);
										_DESTi = _mm256_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi16(2);

										_DESTi = _mm_add_epi16(_Ai, _Bi);
										_DESTi = _mm_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i + 32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm512_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm256_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_min_epu16(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_max_epu16(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif 							
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm512_add_epi32(_Ai, _Bi);
										_DESTi = _mm512_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm256_add_epi32(_Ai, _Bi);
										_DESTi = _mm256_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = tpa::util::_mm_mul_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_min_epi32(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_max_epi32(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi32(2);

										_DESTi = _mm_add_epi32(_Ai, _Bi);
										_DESTi = _mm_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi32(2);

										_DESTi = _mm512_add_epi32(_Ai, _Bi);
										_DESTi = _mm512_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm256_add_epi32(_Ai, _Bi);
										_DESTi = _mm256_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
									}//End for
								}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm_mul_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_min_epu32(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = _mm_max_epu32(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi32(2);

										_DESTi = _mm_add_epi32(_Ai, _Bi);
										_DESTi = _mm_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
							}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
#ifdef __AVX512DQ__
										_DESTi = _mm512_mullo_epi64(_Ai, _Bi);
#else
										_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
#endif
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi64(2);

										_DESTi = _mm512_add_epi64(_Ai, _Bi);
										_DESTi = _mm512_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = tpa::util::_mm256_mul_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi64x(2);

										_DESTi = _mm256_add_epi64(_Ai, _Bi);
										_DESTi = _mm256_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = tpa::util::_mm_mul_epi64(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi64x(2);

										_DESTi = _mm_add_epi64(_Ai, _Bi);
										_DESTi = _mm_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
#ifdef __AVX512DQ__
										_DESTi = _mm512_mullo_epi64(_Ai, _Bi);
#else
										_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
#endif
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi64(2);

										_DESTi = _mm512_add_epi64(_Ai, _Bi);
										_DESTi = _mm512_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = tpa::util::_mm256_mul_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::BIT_AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::BIT_OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::BIT_XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::BIT_AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi64x(2);

										_DESTi = _mm256_add_epi64(_Ai, _Bi);
										_DESTi = _mm256_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										if (tpa::has_SSE41) [[likely]]
										{
											_DESTi = tpa::util::_mm_mul_epi64(_Ai, _Bi);
										}//End if
										else
										{
											break;
										}//End else
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm_rem_epu64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										break;
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_epi64x(2);

										_DESTi = _mm_add_epi64(_Ai, _Bi);
										_DESTi = _mm_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _Ai, _Bi, _DESTi;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);
									_Bi = _mm512_load_ps(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if		
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512 _TWO = _mm512_set1_ps(2.0f);

										_DESTi = _mm512_add_ps(_Ai, _Bi);
										_DESTi = _mm512_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm512_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm512_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);
									_Bi = _mm256_load_ps(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if	
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256 _TWO = _mm256_set1_ps(2.0f);

										_DESTi = _mm256_add_ps(_Ai, _Bi);
										_DESTi = _mm256_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm256_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm256_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE)
							{
								__m128 _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_ps(&source1[i]);
									_Bi = _mm_load_ps(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break;
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_ps(2.0f);

										_DESTi = _mm_add_ps(_Ai, _Bi);
										_DESTi = _mm_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if has_SSE
#endif						
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);
									_Bi = _mm512_load_pd(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512d _TWO = _mm512_set1_pd(2.0);

										_DESTi = _mm512_add_pd(_Ai, _Bi);
										_DESTi = _mm512_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm512_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm512_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);
									_Bi = _mm256_load_pd(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256d _TWO = _mm256_set1_pd(2.0);

										_DESTi = _mm256_add_pd(_Ai, _Bi);
										_DESTi = _mm256_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm256_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm256_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE2)
							{
								__m128d _Ai, _Bi, _DESTi;

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_pd(&source1[i]);
									_Bi = _mm_load_pd(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break;
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m128i _TWO = _mm_set1_pd(2.0);

										_DESTi = _mm_add_pd(_Ai, _Bi);
										_DESTi = _mm_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif						
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Calc
							if constexpr (INSTR == tpa::op::ADD)
							{
								dest[i] = source1[i] + source2[i];
							}//End if
							else if constexpr (INSTR == tpa::op::SUBTRACT)
							{
								dest[i] = source1[i] - source2[i];
							}//End if
							else if constexpr (INSTR == tpa::op::MULTIPLY)
							{
								dest[i] = source1[i] * source2[i];
							}//End if
							else if constexpr (INSTR == tpa::op::DIVIDE)
							{
								dest[i] = source1[i] / source2[i];
							}//End if
							else if constexpr (INSTR == tpa::op::MODULO)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = static_cast<T>(std::fmod(source1[i], source2[i]));
								}//End if
								else
								{
									dest[i] = source1[i] % source2[i];
								}//End else
							}//end if							
							else if constexpr (INSTR == tpa::op::MIN)
							{
								dest[i] = tpa::util::min(source1[i], source2[i]);
							}//End if
							else if constexpr (INSTR == tpa::op::MAX)
							{
								dest[i] = tpa::util::max(source1[i], source2[i]);
							}//End if
							else if constexpr (INSTR == tpa::op::AVERAGE)
							{
								dest[i] = static_cast<T>((source1[i] + source2[i]) / 2);
							}//End if
							else if constexpr (INSTR == tpa::op::POWER)
							{
								dest[i] = tpa::util::pow(source1[i], source2[i]);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
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
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: unknown!\n";
		}//End catch
	}//End of calculate()

	/// <summary>
	/// <para>Computes matrix math on 2 std::arrays storing the result in a 3rd std::array .</para> 
	/// <para> Arrays of different types are NOT allowed</para>
	/// <para> Arrays of different value types are NOT allowed</para>
	/// <para> Arrays do not have to be a particular size</para>
	/// <para> If passing 2 arrays of different sizes, values will only be calculated up to the array with the smallest size, the destination array must be at least this large</para> 
	/// <para>Templated predicate takes 1 of these predicates: tpa::op</para>
	/// <para>---------------------------------------------------</para>
	///<para>tpa::op::ADD< / para>
	/// <para>tpa::op::SUBRACT</para>		
	/// <para>tpa::op::MULTIPLY</para>		
	/// <para>tpa::op::DIVIDE</para>
	/// <para>tpa::op::MODULO</para>
	/// <para>tpa::op::MIN</para>
	/// <para>tpa::op::MAX</para>
	/// <para>tpa::op::POW</para>
	/// <para>tpa::op::AVERAGE</para>
	/// </summary>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::op INSTR, size_t SIZE1, size_t SIZE3, typename T>
	inline constexpr void calculate_const(
		const std::array<T, SIZE1>& source1,
		const T val,
		std::array<T, SIZE3>& dest)
	{
		try
		{
			uint32_t complete = 0;

			static_assert(SIZE1 <= SIZE3,
				"Compile Error! The destination array is not large enough!");

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, source1.size());

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &val, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T _val = val;//Copy for thread-locality

#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi8(_val);

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[64] = {};
										for (size_t x = 0; x != 64; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m512i_i8[x] * _Bi.m512i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm512_loadu_epi8((__m512i*)&d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi8(2);

										_DESTi = _mm512_add_epi8(_Ai, _Bi);
										_DESTi = _mm512_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi8(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[32] = {};
										for (size_t x = 0; x != 32; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m256i_i8[x] * _Bi.m256i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi8(2);

										_DESTi = _mm256_add_epi8(_Ai, _Bi);
										_DESTi = _mm256_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif
						}//End if

#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi8(_val);

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[64] = {};
										for (size_t x = 0; x != 64; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m512i_u8[x] * _Bi.m512i_u8[x]);
#else
											d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm512_loadu_epi8((__m512i*)&d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm512_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi8(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[32] = {};
										for (size_t x = 0; x != 32; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m256i_u8[x] * _Bi.m256i_u8[x]);
#else
											d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm256_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX2
#endif				
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi16(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi16(2);

										_DESTi = _mm512_add_epi16(_Ai, _Bi);
										_DESTi = _mm512_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi16(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi16(2);

										_DESTi = _mm256_add_epi16(_Ai, _Bi);
										_DESTi = _mm256_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi16(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm512_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi16(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm256_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif 							
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi32(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm512_add_epi32(_Ai, _Bi);
										_DESTi = _mm512_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi32(_val);

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm256_add_epi32(_Ai, _Bi);
										_DESTi = _mm256_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi32(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi32(2);

										_DESTi = _mm512_add_epi32(_Ai, _Bi);
										_DESTi = _mm512_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi32(_val);

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm256_add_epi32(_Ai, _Bi);
										_DESTi = _mm256_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi64(_val);

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi64(2);

										_DESTi = _mm512_add_epi64(_Ai, _Bi);
										_DESTi = _mm512_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi64x(_val);

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint64_t d[4] = {};
										for (size_t x = 0; x != 4; ++x)
										{
#if _WIN32
											d[x] = _Ai.m256i_i64[x] * _Bi.m256i_i64[x];
#else
											d[x] = _Ai[x] * _Bi[x];
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);

									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi64x(2);

										_DESTi = _mm256_add_epi64(_Ai, _Bi);
										_DESTi = _mm256_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi64(_val);

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi64(2);

										_DESTi = _mm512_add_epi64(_Ai, _Bi);
										_DESTi = _mm512_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi64x(_val);

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint64_t d[4] = {};
										for (size_t x = 0; x != 4; ++x)
										{
#ifdef _WIN32
											d[x] = _Ai.m256i_i64[x] * _Bi.m256i_i64[x];
#else
											d[x] = _Ai[x] * _Bi[x];
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);

									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi64x(2);

										_DESTi = _mm256_add_epi64(_Ai, _Bi);
										_DESTi = _mm256_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _Ai, _DESTi;
								const __m512 _Bi = _mm512_set1_ps(_val);

								for (; i != end; i += 16)
								{

									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if		
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512 _TWO = _mm512_set1_ps(2.0f);

										_DESTi = _mm512_add_ps(_Ai, _Bi);
										_DESTi = _mm512_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm512_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm512_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _DESTi;
								const __m256 _Bi = _mm256_set1_ps(_val);

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if		
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256 _TWO = _mm256_set1_ps(2.0f);

										_DESTi = _mm256_add_ps(_Ai, _Bi);
										_DESTi = _mm256_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm256_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm256_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif						
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _Ai, _DESTi;
								const __m512d _Bi = _mm512_set1_pd(_val);

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512d _TWO = _mm512_set1_pd(2);

										_DESTi = _mm512_add_pd(_Ai, _Bi);
										_DESTi = _mm512_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm512_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm512_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _DESTi;
								const __m256d _Bi = _mm256_set1_pd(_val);

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256d _TWO = _mm256_set1_pd(2);

										_DESTi = _mm256_add_pd(_Ai, _Bi);
										_DESTi = _mm256_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm256_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm256_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif						
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Calc
							if constexpr (INSTR == tpa::op::ADD)
							{
								dest[i] = source1[i] + _val;
							}//End if
							else if constexpr (INSTR == tpa::op::SUBTRACT)
							{
								dest[i] = source1[i] - _val;
							}//End if
							else if constexpr (INSTR == tpa::op::MULTIPLY)
							{
								dest[i] = source1[i] * _val;
							}//End if
							else if constexpr (INSTR == tpa::op::DIVIDE)
							{
								dest[i] = source1[i] / _val;
							}//End if
							else if constexpr (INSTR == tpa::op::MODULO)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = static_cast<T>(std::fmod(source1[i], _val));
								}//End if
								else
								{
									dest[i] = source1[i] % _val;
								}//End else
							}//end if							
							else if constexpr (INSTR == tpa::op::MIN)
							{
								dest[i] = tpa::util::min(source1[i], _val);
							}//End if
							else if constexpr (INSTR == tpa::op::MAX)
							{
								dest[i] = tpa::util::max(source1[i], _val);
							}//End if
							else if constexpr (INSTR == tpa::op::AVERAGE)
							{
								dest[i] = static_cast<T>((source1[i] + _val) / 2);
							}//End if
							else if constexpr (INSTR == tpa::op::POWER)
							{
								dest[i] = tpa::util::pow(source1[i], _val);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
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
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: unknown!\n";
		}//End catch
	}//End of calculate_const()

	/// <summary>
	/// <para>Compares 2 arrays and stores the results of the expression in a 3rd array.</para>
	/// <para>Example: if source1[0] > source2[0]...dest[0] = source1[0] otherwise writes zeros.</para>
	/// <para>Optionally you can remove zeros, false by default</para>
	/// <para>Templated predicate takes 1 of these predicates: tpa::comp</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::comp::EQUAL_TO</para>			
	/// <para>tpa::comp::NOT_EQUAL_TO</para>		
	/// <para>tpa::comp::LESS_THAN</para>		
	/// <para>tpa::comp::LESS_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::GREATER_THAN</para>			
	/// <para>tpa::comp::GREATER_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::MIN</para>
	/// <para>tpa::comp::MAX</para>
	/// </summary>
	/// <typeparam name="CONTAINER"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::comp INSTR, typename T, size_t SIZE1, size_t SIZE2, size_t SIZE3>
	inline constexpr void compare(
		const std::array<T,SIZE1>& source1,
		const std::array<T,SIZE2>& source2,
		std::array<T,SIZE3>& dest)
	{
		try
		{
			size_t smallest = 0;
			uint32_t complete = 0;

			static_assert(SIZE1 <= SIZE3 && SIZE2 <= SIZE3,
				"Compile Error! The destination array is not large enough!");

			//Determin the smallest container
			smallest = tpa::util::min(source1.size(), source2.size());

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
#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask64 _mask;

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8(&source1[i]);
									_Bi = _mm512_loadu_epi8(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask64 _mask;

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8(&source1[i]);
									_Bi = _mm512_loadu_epi8(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask32 _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16(&source1[i]);
									_Bi = _mm512_loadu_epi16(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask32 _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16(&source1[i]);
									_Bi = _mm512_loadu_epi16(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32(&source1[i]);
									_Bi = _mm512_load_epi32(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64	
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32(&source1[i]);
									_Bi = _mm512_load_epi32(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64(&source1[i]);
									_Bi = _mm512_load_epi64(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64	
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64(&source1[i]);
									_Bi = _mm512_load_epi64(&source2[i]);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64			
							if (tpa::hasAVX512)
							{
								__m512 _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);
									_Bi = _mm512_load_ps(&source2[i]);
									_dest = _mm512_setzero_ps();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GT_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GE_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LT_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LE_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_EQ_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_ps(_Ai, _Bi);
										_mm512_store_ps(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_ps(_Ai, _Bi);
										_mm512_store_ps(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _Bi, _mask;

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);
									_Bi = _mm256_load_ps(&source2[i]);

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GT_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GE_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LT_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LE_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_EQ_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_mask = _mm256_min_ps(_Ai, _Bi);
										_mm256_store_ps(&dest[i], _mask);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_mask = _mm256_max_ps(_Ai, _Bi);
										_mm256_store_ps(&dest[i], _mask);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else
								}//End for
							}//End if has AVX2
#endif							
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);
									_Bi = _mm512_load_pd(&source2[i]);
									_dest = _mm512_setzero_pd();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GT_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GE_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LT_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LE_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_EQ_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_pd(_Ai, _Bi);
										_mm512_store_pd(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_pd(_Ai, _Bi);
										_mm512_store_pd(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _Bi, _mask;

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);
									_Bi = _mm256_load_pd(&source2[i]);

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GT_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GE_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LT_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LE_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_EQ_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_mask = _mm256_min_pd(_Ai, _Bi);
										_mm256_store_pd(&dest[i], _mask);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_mask = _mm256_max_pd(_Ai, _Bi);
										_mm256_store_pd(&dest[i], _mask);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								if (source1[i] > source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								if (source1[i] >= source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								if (source1[i] < source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								if (source1[i] <= source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								if (source1[i] == source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								if (source1[i] != source2[i])
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								dest[i] = tpa::util::min(source1[i], source2[i]);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								dest[i] = tpa::util::max(source1[i], source2[i]);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
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
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare: unknown!\n";
		}//End catch
	}//End of compare()

	/// <summary>
	/// <para>Compares 2 arrays and stores the results of the expression in a 3rd array.</para>
	/// <para>Example: if source1[0] > source2[0]...dest[0] = source1[0] otherwise writes zeros.</para>
	/// <para>Optionally you can remove zeros, false by default</para>
	/// <para>Templated predicate takes 1 of these predicates: tpa::comp</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::comp::EQUAL_TO</para>			
	/// <para>tpa::comp::NOT_EQUAL_TO</para>		
	/// <para>tpa::comp::LESS_THAN</para>		
	/// <para>tpa::comp::LESS_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::GREATER_THAN</para>			
	/// <para>tpa::comp::GREATER_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::MIN</para>
	/// <para>tpa::comp::MAX</para>
	/// </summary>
	/// <typeparam name="CONTAINER"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::comp INSTR, typename T, size_t SIZE1, size_t SIZE3>
	inline constexpr void compare_const(
		const std::array<T, SIZE1>& source1,
		const T val,
		std::array<T, SIZE3>& dest)
	{
		try
		{
			size_t smallest = 0;
			uint32_t complete = 0;

			static_assert(SIZE1 <= SIZE3,
				"Compile Error! The destination array is not large enough!");

			//Determin the smallest container
			smallest = source1.size();

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &val, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T _val = val;//Copy for thread-locality

#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask64 _mask;

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8(&source1[i]);
									_Bi = _mm512_set1_epi8(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi8(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask64 _mask;

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8(&source1[i]);
									_Bi = _mm512_set1_epi8(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi8(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask32 _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16(&source1[i]);
									_Bi = _mm512_set1_epi16(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi16(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask32 _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16(&source1[i]);
									_Bi = _mm512_set1_epi16(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi16(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32(&source1[i]);
									_Bi = _mm512_set1_epi32(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi32(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64	
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32(&source1[i]);
									_Bi = _mm512_set1_epi32(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi32(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64(&source1[i]);
									_Bi = _mm512_set1_epi64(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi64x(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64	
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64(&source1[i]);
									_Bi = _mm512_set1_epi64(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi64x(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64			
							if (tpa::hasAVX512)
							{
								__m512 _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);
									_Bi = _mm512_set1_ps(_val);
									_dest = _mm512_setzero_ps();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GT_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GE_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LT_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LE_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_EQ_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_ps(_Ai, _Bi);
										_mm512_store_ps(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_ps(_Ai, _Bi);
										_mm512_store_ps(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _Bi, _mask;

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);
									_Bi = _mm256_set1_ps(_val);

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GT_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GE_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LT_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LE_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_EQ_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_mask = _mm256_min_ps(_Ai, _Bi);
										_mm256_store_ps(&dest[i], _mask);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_mask = _mm256_max_ps(_Ai, _Bi);
										_mm256_store_ps(&dest[i], _mask);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else
								}//End for
							}//End if has AVX2
#endif							
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);
									_Bi = _mm512_set1_pd(_val);
									_dest = _mm512_setzero_pd();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GT_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GE_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LT_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LE_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_EQ_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_pd(_Ai, _Bi);
										_mm512_store_pd(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_pd(_Ai, _Bi);
										_mm512_store_pd(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _Bi, _mask;

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);
									_Bi = _mm256_set1_pd(_val);

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GT_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GE_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LT_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LE_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_EQ_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_mask = _mm256_min_pd(_Ai, _Bi);
										_mm256_store_pd(&dest[i], _mask);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_mask = _mm256_max_pd(_Ai, _Bi);
										_mm256_store_pd(&dest[i], _mask);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								if (source1[i] > _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								if (source1[i] >= _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								if (source1[i] < _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								if (source1[i] <= _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								if (source1[i] == _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								if (source1[i] != _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								dest[i] = tpa::util::min(source1[i], _val);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								dest[i] = tpa::util::max(source1[i], _val);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
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
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const: unknown!\n";
		}//End catch
	}//End of compare_vector()
#pragma endregion
#pragma region vector

	/// <summary>
	/// <para>Computes matrix math on 2 std::vectors storing the result in a 3rd std::vector.</para> 
	/// <para> Vectors of different types are NOT allowed</para>
	/// <para> Vectors of different value types are NOT allowed</para>
	/// <para> Vectors do not have to be a particular size</para>
	/// <para> If passing 2 vectors of different sizes, values will only be calculated up to the vector with the smallest size, the destination vector should be at least this large, if not it will be resized.</para> 
	/// <para>Templated predicate takes 1 of these predicates: tpa::op</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::op::ADD</para>			
	/// <para>tpa::op::SUBRACT</para>		
	/// <para>tpa::op::MULTIPLY</para>		
	/// <para>tpa::op::DIVIDE</para>
	/// <para>tpa::op::MODULO</para>
	/// <para>tpa::op::MIN</para>
	/// <para>tpa::op::MAX</para>
	/// <para>tpa::op::POW</para>
	/// <para>tpa::op::AVERAGE</para>
	/// </summary>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::op INSTR, typename T>
	inline constexpr void calculate(
		const std::vector<T>& source1,
		const std::vector<T>& source2,
		std::vector<T>& dest)
	{
		try
		{
			uint32_t complete = 0;

			//Determin the smallest container
			size_t smallest = tpa::util::min(source1.size(), source2.size());

			//Make sure the results vector is adaquately sized			
			if (dest.size() < smallest)
			{
				dest.resize(smallest);
			}//End if
			
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

#pragma region byte
					if constexpr (std::is_same<T, int8_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; (i + 64) < end; i += 64)
							{
								//Set Values
								_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);
								_Bi = _mm512_loadu_epi8((__m512i*)&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm512_add_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm512_sub_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									int8_t d[64] = {};
									for (size_t x = 0; x != 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_Ai.m512i_i8[x] * _Bi.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
									}//End for
									_DESTi = _mm512_loadu_epi8((__m512i*)&d);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm512_div_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm512_rem_epi8(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm512_min_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm512_max_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m512i _TWO = _mm512_set1_epi8(2);

									_DESTi = _mm512_add_epi8(_Ai, _Bi);
									_DESTi = _mm512_div_epi8(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
									}();
								}//End else

								//Store Result
								_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _DESTi;

							for (; (i + 32) < end; i += 32)
							{
								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm256_sub_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									int8_t d[32] = {};
									for (size_t x = 0; x != 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_Ai.m256i_i8[x] * _Bi.m256i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
									}//End for
									_DESTi = _mm256_load_si256((__m256i*) & d);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm256_div_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm256_rem_epi8(_Ai, _Bi);
							}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm256_min_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm256_max_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m256i _TWO = _mm256_set1_epi8(2);

									_DESTi = _mm256_add_epi8(_Ai, _Bi);
									_DESTi = _mm256_div_epi8(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
						}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							__m128i _Ai, _Bi, _DESTi;

							for (; (i + 16) < end; i += 16)
							{
								//Set Values
								_Ai = _mm_load_si128((__m128i*) & source1[i]);
								_Bi = _mm_load_si128((__m128i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm_add_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm_sub_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									int8_t d[16] = {};
									for (size_t x = 0; x != 16; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_Ai.m128i_i8[x] * _Bi.m128i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
									}//End for
									_DESTi = _mm_load_si128((__m128i*) & d);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm_div_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm_rem_epi8(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									if (tpa::has_SSE41) [[likely]]
									{
										_DESTi = _mm_min_epi8(_Ai, _Bi);
									}//End if
									else
									{
										break;
									}//End else
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									if (tpa::has_SSE41) [[likely]]
									{
										_DESTi = _mm_max_epi8(_Ai, _Bi);
									}//End if
									else
									{
										break;
									}//End else
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m128i _TWO = _mm_set1_epi8(2);

									_DESTi = _mm_add_epi8(_Ai, _Bi);
									_DESTi = _mm_div_epi8(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & dest[i], _DESTi);
									}//End for
								}//End if has_SSE2
#endif
							}//End if

#pragma endregion
#pragma region unsigned byte
					else if constexpr (std::is_same<T, uint8_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; (i + 64) < end; i += 64)
							{
								//Set Values
								_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);
								_Bi = _mm512_loadu_epi8((__m512i*)&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm512_sub_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									uint8_t d[64] = {};
									for (size_t x = 0; x != 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_Ai.m512i_u8[x] * _Bi.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
									}//End for
									_DESTi = _mm512_loadu_epi8((__m512i*)&d);
									}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm512_div_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm512_rem_epu8(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm512_min_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm512_max_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									_DESTi = _mm512_avg_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
									}();
								}//End else

								//Store Result
								_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX512
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _DESTi;

							for (; (i + 32) < end; i += 32)
							{
								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm256_sub_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									uint8_t d[32] = {};
									for (size_t x = 0; x != 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_Ai.m256i_u8[x] * _Bi.m256i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
									}//End for
									_DESTi = _mm256_load_si256((__m256i*) & d);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm256_div_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm256_rem_epu8(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm256_min_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm256_max_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									_DESTi = _mm256_avg_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if has AVX2
						else if (tpa::has_SSE2)
						{
							__m128i _Ai, _Bi, _DESTi;

							for (; (i + 16) < end; i += 16)
							{
								//Set Values
								_Ai = _mm_load_si128((__m128i*) & source1[i]);
								_Bi = _mm_load_si128((__m128i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm_add_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm_sub_epi8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									uint8_t d[16] = {};
									for (size_t x = 0; x != 16; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_Ai.m128i_i8[x] * _Bi.m128i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
									}//End for
									_DESTi = _mm_load_si128((__m128i*) & d);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm_div_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm_rem_epu8(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm_min_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm_max_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									_DESTi = _mm_avg_epu8(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & dest[i], _DESTi);
							}//End for
						}//End if has_SSE2
#endif				
						}//End if
#pragma endregion
#pragma region short
					else if constexpr (std::is_same<T, int16_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; (i + 32) < end; i += 32)
							{
								//Set Values
								_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
								_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm512_add_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm512_sub_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm512_div_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm512_rem_epi16(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm512_min_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm512_max_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m512i _TWO = _mm512_set1_epi16(2);

									_DESTi = _mm512_add_epi16(_Ai, _Bi);
									_DESTi = _mm512_div_epi16(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
									}();
								}//End else

								//Store Result
								_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _DESTi;

							for (; (i + 16) < end; i += 16)
							{
								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm256_sub_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm256_div_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm256_rem_epi16(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm256_min_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm256_max_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m256i _TWO = _mm256_set1_epi16(2);

									_DESTi = _mm256_add_epi16(_Ai, _Bi);
									_DESTi = _mm256_div_epi16(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							__m128i _Ai, _Bi, _DESTi;

							for (; (i + 8) < end; i += 8)
							{
								//Set Values
								_Ai = _mm_load_si128((__m128i*) & source1[i]);
								_Bi = _mm_load_si128((__m128i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm_add_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm_sub_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm_mullo_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm_div_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm_rem_epi16(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm_min_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm_max_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m128i _TWO = _mm_set1_epi16(2);

									_DESTi = _mm_add_epi16(_Ai, _Bi);
									_DESTi = _mm_div_epi16(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & dest[i], _DESTi);
							}//End for
						}//End if has_SSE2
#endif							
					}//End if
#pragma endregion
#pragma region unsigned short
					else if constexpr (std::is_same<T, uint16_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; (i + 32) < end; i += 32)
							{
								//Set Values
								_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
								_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm512_add_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm512_sub_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm512_div_epu16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm512_rem_epu16(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm512_min_epu16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm512_max_epu16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									_DESTi = _mm512_avg_epu16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _DESTi;

							for (; (i + 16) < end; i += 16)
							{
								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm256_sub_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm256_div_epu16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm256_rem_epu16(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm256_min_epu16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm256_max_epu16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									_DESTi = _mm256_avg_epu16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							__m128i _Ai, _Bi, _DESTi;

							for (; (i + 8) < end; i += 8)
							{
								//Set Values
								_Ai = _mm_load_si128((__m128i*) & source1[i]);
								_Bi = _mm_load_si128((__m128i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm_add_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm_sub_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm_mullo_epi16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm_div_epu16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm_rem_epu16(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									if (tpa::has_SSE41) [[likely]]
									{
										_DESTi = _mm_min_epu16(_Ai, _Bi);
									}//End if
									else
									{
										break;
									}//End else
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									if (tpa::has_SSE41) [[likely]]
									{
										_DESTi = _mm_max_epu16(_Ai, _Bi);
									}//End if
									else
									{
										break;
									}//End else
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									_DESTi = _mm_avg_epu16(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & dest[i], _DESTi);
							}//End for
						}//End if has_SSE2
#endif 							
					}//End if
#pragma endregion
#pragma region int
					else if constexpr (std::is_same<T, int32_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; (i + 16) < end; i += 16)
							{
								//Set Values
								_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
								_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm512_add_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm512_sub_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm512_div_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm512_rem_epi32(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm512_min_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm512_max_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m256i _TWO = _mm256_set1_epi32(2);

									_DESTi = _mm512_add_epi32(_Ai, _Bi);
									_DESTi = _mm512_div_epi32(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
									}();
								}//End else

								//Store Result
								_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _DESTi;

							for (; (i + 8) < end; i += 8)
							{
								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm256_sub_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm256_div_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm256_rem_epi32(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm256_min_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm256_max_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m256i _TWO = _mm256_set1_epi32(2);

									_DESTi = _mm256_add_epi32(_Ai, _Bi);
									_DESTi = _mm256_div_epi32(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							__m128i _Ai, _Bi, _DESTi;

							for (; (i + 4) < end; i += 4)
							{
								//Set Values
								_Ai = _mm_load_si128((__m128i*) & source1[i]);
								_Bi = _mm_load_si128((__m128i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm_add_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm_sub_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = tpa::util::_mm_mul_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm_div_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm_rem_epi32(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									if (tpa::has_SSE41) [[likely]]
									{
										_DESTi = _mm_min_epi32(_Ai, _Bi);
									}//End if
									else
									{
										break;
									}//End else
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									if (tpa::has_SSE41) [[likely]]
									{
										_DESTi = _mm_max_epi32(_Ai, _Bi);
									}//End if
									else
									{
										break;
									}//End else
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m128i _TWO = _mm_set1_epi32(2);

									_DESTi = _mm_add_epi32(_Ai, _Bi);
									_DESTi = _mm_div_epi32(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & dest[i], _DESTi);
							}//End for
						}//End if has_SSE2
#endif							
					}//End if
#pragma endregion
#pragma region unsigned int
					else if constexpr (std::is_same<T, uint32_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; (i + 16) < end; i += 16)
							{
								//Set Values
								_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
								_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm512_add_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm512_sub_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm512_div_epu32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm512_rem_epu32(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm512_min_epu32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm512_max_epu32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m512i _TWO = _mm512_set1_epi32(2);

									_DESTi = _mm512_add_epi32(_Ai, _Bi);
									_DESTi = _mm512_div_epu32(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
									}();
								}//End else

								//Store Result
								_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
							}//End for
						}//End if
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _DESTi;

							for (; (i + 8) < end; i += 8)
							{
								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm256_sub_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm256_div_epu32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm256_rem_epu32(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm256_min_epu32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm256_max_epu32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m256i _TWO = _mm256_set1_epi32(2);

									_DESTi = _mm256_add_epi32(_Ai, _Bi);
									_DESTi = _mm256_div_epu32(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							__m128i _Ai, _Bi, _DESTi;

							for (; (i + 4) < end; i += 4)
							{
								//Set Values
								_Ai = _mm_load_si128((__m128i*) & source1[i]);
								_Bi = _mm_load_si128((__m128i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm_add_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm_sub_epi32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm_mul_epu32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm_div_epu32(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm_rem_epu32(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									if (tpa::has_SSE41) [[likely]]
									{
										_DESTi = _mm_min_epu32(_Ai, _Bi);
									}//End if
									else
									{
										break;
									}//End else
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									if (tpa::has_SSE41) [[likely]]
									{
										_DESTi = _mm_max_epu32(_Ai, _Bi);
									}//End if
									else
									{
										break;
									}//End else
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m128i _TWO = _mm_set1_epi32(2);

									_DESTi = _mm_add_epi32(_Ai, _Bi);
									_DESTi = _mm_div_epu32(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & dest[i], _DESTi);
							}//End for
						}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region long
					else if constexpr (std::is_same<T, int64_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; (i + 8) < end; i += 8)
							{
								//Set Values
								_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
								_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm512_add_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm512_sub_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
#ifdef __AVX512DQ__
									_DESTi = _mm512_mullo_epi64(_Ai, _Bi);
#else
									_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
#endif
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm512_div_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm512_rem_epi64(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm512_min_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm512_max_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m512i _TWO = _mm512_set1_epi64(2);

									_DESTi = _mm512_add_epi64(_Ai, _Bi);
									_DESTi = _mm512_div_epi64(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
									}();
								}//End else

								//Store Result
								_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _DESTi;

							for (; (i + 4) < end; i += 4)
							{
								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm256_sub_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = tpa::util::_mm256_mul_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm256_div_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm256_rem_epi64(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									break;
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									break;
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m256i _TWO = _mm256_set1_epi64x(2);

									_DESTi = _mm256_add_epi64(_Ai, _Bi);
									_DESTi = _mm256_div_epi64(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							__m128i _Ai, _Bi, _DESTi;

							for (; (i + 2) < end; i += 2)
							{
								//Set Values
								_Ai = _mm_load_si128((__m128i*) & source1[i]);
								_Bi = _mm_load_si128((__m128i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm_add_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm_sub_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									if (tpa::has_SSE41) [[likely]]
									{
										_DESTi = tpa::util::_mm_mul_epi64(_Ai, _Bi);
									}//End if
									else
									{
										break;
									}//End else
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm_div_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm_rem_epi64(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									break;
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									break;
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m128i _TWO = _mm_set1_epi64x(2);

									_DESTi = _mm_add_epi64(_Ai, _Bi);
									_DESTi = _mm_div_epi64(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & dest[i], _DESTi);
							}//End for
						}//End if has_SSE2
#endif							
					}//End if
#pragma endregion
#pragma region unsigned long
					else if constexpr (std::is_same<T, uint64_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; (i + 8) < end; i += 8)
							{
								//Set Values
								_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
								_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm512_add_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm512_sub_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
#ifdef __AVX512DQ__
									_DESTi = _mm512_mullo_epi64(_Ai, _Bi);
#else
									_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
#endif
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm512_div_epu64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm512_rem_epu64(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm512_min_epu64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm512_max_epu64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m512i _TWO = _mm512_set1_epi64(2);

									_DESTi = _mm512_add_epi64(_Ai, _Bi);
									_DESTi = _mm512_div_epu64(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
									}();
								}//End else

								//Store Result
								_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _DESTi;

							for (; (i + 4) < end; i += 4)
							{
								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm256_sub_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = tpa::util::_mm256_mul_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm256_div_epu64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm256_rem_epu64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::BIT_AND)
								{
									_DESTi = _mm256_and_si256(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::BIT_OR)
								{
									_DESTi = _mm256_or_si256(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::BIT_XOR)
								{
									_DESTi = _mm256_xor_si256(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::BIT_AND_NOT)
								{
									_DESTi = _mm256_andnot_si256(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm256_min_pd(
										_mm256_castsi256_pd(_Ai),
										_mm256_castsi256_pd(_Bi)
									);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm256_max_pd(
										_mm256_castsi256_pd(_Ai),
										_mm256_castsi256_pd(_Bi)
									);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m256i _TWO = _mm256_set1_epi64x(2);

									_DESTi = _mm256_add_epi64(_Ai, _Bi);
									_DESTi = _mm256_div_epu64(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							__m128i _Ai, _Bi, _DESTi;

							for (; (i + 2) < end; i += 2)
							{
								//Set Values
								_Ai = _mm_load_si128((__m128i*) & source1[i]);
								_Bi = _mm_load_si128((__m128i*) & source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm_add_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm_sub_epi64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									if (tpa::has_SSE41) [[likely]]
									{
										_DESTi = tpa::util::_mm_mul_epi64(_Ai, _Bi);
									}//End if
									else
									{
										break;
									}//End else
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm_div_epu64(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									_DESTi = _mm_rem_epu64(_Ai, _Bi);
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									break;
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									break;
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m128i _TWO = _mm_set1_epi64x(2);

									_DESTi = _mm_add_epi64(_Ai, _Bi);
									_DESTi = _mm_div_epu64(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									break;
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & dest[i], _DESTi);
							}//End for
						}//End if has_SSE2
#endif							
					}//End if
#pragma endregion
#pragma region float
					else if constexpr (std::is_same<T, float>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512 _Ai, _Bi, _DESTi;

							for (; (i + 16) < end; i += 16)
							{
								//Set Values
								_Ai = _mm512_load_ps(&source1[i]);
								_Bi = _mm512_load_ps(&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm512_add_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm512_sub_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm512_mul_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm512_div_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									break; //Use fmod
								}//End if		
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm512_min_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm512_max_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m512 _TWO = _mm512_set1_ps(2.0f);

									_DESTi = _mm512_add_ps(_Ai, _Bi);
									_DESTi = _mm512_div_ps(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									_DESTi = _mm512_pow_ps(_Ai, _Bi);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm512_store_ps(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							__m256 _Ai, _Bi, _DESTi;

							for (; (i + 8) < end; i += 8)
							{
								//Set Values
								_Ai = _mm256_load_ps(&source1[i]);
								_Bi = _mm256_load_ps(&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm256_sub_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm256_mul_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm256_div_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									break; //Use fmod
								}//End if	
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm256_min_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm256_max_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m256 _TWO = _mm256_set1_ps(2.0f);

									_DESTi = _mm256_add_ps(_Ai, _Bi);
									_DESTi = _mm256_div_ps(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									_DESTi = _mm256_pow_ps(_Ai, _Bi);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm256_store_ps(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE)
						{
							__m128 _Ai, _Bi, _DESTi;

							for (; (i + 4) < end; i += 4)
							{
								//Set Values
								_Ai = _mm_load_ps(&source1[i]);
								_Bi = _mm_load_ps(&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm_add_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm_sub_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm_mul_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm_div_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									break;
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm_min_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm_max_ps(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m128i _TWO = _mm_set1_ps(2.0f);

									_DESTi = _mm_add_ps(_Ai, _Bi);
									_DESTi = _mm_div_ps(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									_DESTi = _mm_pow_ps(_Ai, _Bi);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm_store_ps(&dest[i], _DESTi);
							}//End for
						}//End if has_SSE
#endif						
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512d _Ai, _Bi, _DESTi;

							for (; (i + 8) < end; i += 8)
							{
								//Set Values
								_Ai = _mm512_load_pd(&source1[i]);
								_Bi = _mm512_load_pd(&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm512_add_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm512_sub_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm512_mul_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm512_div_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									break; //Use fmod
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm512_min_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm512_max_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m512d _TWO = _mm512_set1_pd(2.0);

									_DESTi = _mm512_add_pd(_Ai, _Bi);
									_DESTi = _mm512_div_pd(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									_DESTi = _mm512_pow_pd(_Ai, _Bi);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
									}();
								}//End else

								//Store Result
								_mm512_store_pd(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							__m256d _Ai, _Bi, _DESTi;

							for (; (i + 4) < end; i += 4)
							{
								//Set Values
								_Ai = _mm256_load_pd(&source1[i]);
								_Bi = _mm256_load_pd(&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm256_add_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm256_sub_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm256_mul_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm256_div_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									break; //Use fmod
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm256_min_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm256_max_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m256d _TWO = _mm256_set1_pd(2.0);

									_DESTi = _mm256_add_pd(_Ai, _Bi);
									_DESTi = _mm256_div_pd(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									_DESTi = _mm256_pow_pd(_Ai, _Bi);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
									}();
								}//End else

								//Store Result
								_mm256_store_pd(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX
						else if (tpa::has_SSE2)
						{
							__m128d _Ai, _Bi, _DESTi;

							for (; (i + 2) < end; i += 2)
							{
								//Set Values
								_Ai = _mm_load_pd(&source1[i]);
								_Bi = _mm_load_pd(&source2[i]);

								//Calc
								if constexpr (INSTR == tpa::op::ADD)
								{
									_DESTi = _mm_add_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::SUBTRACT)
								{
									_DESTi = _mm_sub_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MULTIPLY)
								{
									_DESTi = _mm_mul_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::DIVIDE)
								{
									_DESTi = _mm_div_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MODULO)
								{
									break;
								}//End if									
								else if constexpr (INSTR == tpa::op::MIN)
								{
									_DESTi = _mm_min_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::MAX)
								{
									_DESTi = _mm_max_pd(_Ai, _Bi);
								}//End if
								else if constexpr (INSTR == tpa::op::AVERAGE)
								{
									const __m128i _TWO = _mm_set1_pd(2.0);

									_DESTi = _mm_add_pd(_Ai, _Bi);
									_DESTi = _mm_div_pd(_DESTi, _TWO);
								}//End if
								else if constexpr (INSTR == tpa::op::POWER)
								{
									_DESTi = _mm_pow_pd(_Ai, _Bi);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
									}();
								}//End else

								//Store Result
								_mm_store_pd(&dest[i], _DESTi);
							}//End for
						}//End if has_SSE2
#endif						
					}//End if
#pragma endregion
#pragma region generic
					for (; i != end; ++i)
					{
						//Calc
						if constexpr (INSTR == tpa::op::ADD)
						{
							dest[i] = source1[i] + source2[i];
						}//End if
						else if constexpr (INSTR == tpa::op::SUBTRACT)
						{
							dest[i] = source1[i] - source2[i];
						}//End if
						else if constexpr (INSTR == tpa::op::MULTIPLY)
						{
							dest[i] = source1[i] * source2[i];
						}//End if
						else if constexpr (INSTR == tpa::op::DIVIDE)
						{
							dest[i] = source1[i] / source2[i];
						}//End if
						else if constexpr (INSTR == tpa::op::MODULO)
						{
							if constexpr (std::is_floating_point<T>())
							{
								dest[i] = static_cast<T>(std::fmod(source1[i], source2[i]));
							}//End if
							else
							{
								dest[i] = source1[i] % source2[i];
							}//End else
						}//end if							
						else if constexpr (INSTR == tpa::op::MIN)
						{
							dest[i] = tpa::util::min(source1[i], source2[i]);
						}//End if
						else if constexpr (INSTR == tpa::op::MAX)
						{
							dest[i] = tpa::util::max(source1[i], source2[i]);
						}//End if
						else if constexpr (INSTR == tpa::op::AVERAGE)
						{
							dest[i] = static_cast<T>((source1[i] + source2[i]) / 2);
						}//End if
						else if constexpr (INSTR == tpa::op::POWER)
						{
							dest[i] = tpa::util::pow(source1[i], source2[i]);
						}//End if
						else
						{
							[] <bool flag = false>()
							{
								static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
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
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: unknown!\n";
		}//End catch
	}//End of calculate()

	/// <summary>
	/// <para>Computes matrix math on 1 std::vector and a constant storing the result in an std::vector.</para> 
	/// <para> Vectors of different types are NOT allowed</para>
	/// <para> Vectors of different value types are NOT allowed</para>
	/// <para> Vectors do not have to be a particular size</para>
	/// <para>Templated predicate takes 1 of these predicates: tpa::op</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::op::ADD</para>			
	/// <para>tpa::op::SUBRACT</para>		
	/// <para>tpa::op::MULTIPLY</para>		
	/// <para>tpa::op::DIVIDE</para>
	/// <para>tpa::op::MODULO</para>
	/// <para>tpa::op::MIN</para>
	/// <para>tpa::op::MAX</para>
	/// <para>tpa::op::POW</para>
	/// <para>tpa::op::AVERAGE</para>
	/// </summary>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::op INSTR, typename T>
	inline constexpr void calculate_const(
		const std::vector<T>& source1,
		const T val,
		std::vector<T>& dest)
	{
		try
		{
			uint32_t complete = 0;

			//Make sure the results vector is adaquately sized			
			if (dest.size() < source1.size())
			{
				dest.resize(source1.size());
			}//End if

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, source1.size());

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &val, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T _val = val; //Copy for thread-locality

#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi8(_val);

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[64] = {};
										for (size_t x = 0; x != 64; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m512i_i8[x] * _Bi.m512i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm512_loadu_epi8((__m512i*)&d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi8(2);

										_DESTi = _mm512_add_epi8(_Ai, _Bi);
										_DESTi = _mm512_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi8(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										int8_t d[32] = {};
										for (size_t x = 0; x != 32; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<int8_t>(_Ai.m256i_i8[x] * _Bi.m256i_i8[x]);
#else
											d[x] = static_cast<int8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi8(2);

										_DESTi = _mm256_add_epi8(_Ai, _Bi);
										_DESTi = _mm256_div_epi8(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
									}//End for
								}//End if hasAVX2
#endif
							}//End if

#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi8(_val);

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[64] = {};
										for (size_t x = 0; x != 64; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m512i_u8[x] * _Bi.m512i_u8[x]);
#else
											d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm512_loadu_epi8((__m512i*)&d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm512_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
									}();
								}//End else

								//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if has AVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi8(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint8_t d[32] = {};
										for (size_t x = 0; x != 32; ++x)
										{
#ifdef _WIN32
											d[x] = static_cast<uint8_t>(_Ai.m256i_u8[x] * _Bi.m256i_u8[x]);
#else
											d[x] = static_cast<uint8_t>(_Ai[x] * _Bi[x]);
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu8(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm256_avg_epu8(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX2
#endif				
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi16(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi16(2);

										_DESTi = _mm512_add_epi16(_Ai, _Bi);
										_DESTi = _mm512_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai,  _DESTi;
								const __m256i _Bi = _mm256_set1_epi16(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi16(2);

										_DESTi = _mm256_add_epi16(_Ai, _Bi);
										_DESTi = _mm256_div_epi16(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi16(_val);

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm512_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi16(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu16(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										_DESTi = _mm256_avg_epu16(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif 							
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi32(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm512_add_epi32(_Ai, _Bi);
										_DESTi = _mm512_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi32(_val);

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);						

									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm256_add_epi32(_Ai, _Bi);
										_DESTi = _mm256_div_epi32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi32(_val);

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi32(2);

										_DESTi = _mm512_add_epi32(_Ai, _Bi);
										_DESTi = _mm512_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi32(_val);

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mullo_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu32(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_epu32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi32(2);

										_DESTi = _mm256_add_epi32(_Ai, _Bi);
										_DESTi = _mm256_div_epu32(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calc_vector<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi64(_val);

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi64(2);

										_DESTi = _mm512_add_epi64(_Ai, _Bi);
										_DESTi = _mm512_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi64x(_val);

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint64_t d[4] = {};
										for (size_t x = 0; x != 4; ++x)
										{
#if _WIN32
											d[x] = _Ai.m256i_i64[x] * _Bi.m256i_i64[x];
#else
											d[x] = _Ai[x] * _Bi[x];
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);

									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epi64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi64x(2);

										_DESTi = _mm256_add_epi64(_Ai, _Bi);
										_DESTi = _mm256_div_epi64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi64(_val);

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mullox_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm512_rem_epu64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512i _TWO = _mm512_set1_epi64(2);

										_DESTi = _mm512_add_epi64(_Ai, _Bi);
										_DESTi = _mm512_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi64x(_val);

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										uint64_t d[4] = {};
										for (size_t x = 0; x != 4; ++x)
										{
#ifdef _WIN32
											d[x] = _Ai.m256i_i64[x] * _Bi.m256i_i64[x];
#else
											d[x] = _Ai[x] * _Bi[x];
#endif
										}//End for
										_DESTi = _mm256_load_si256((__m256i*) & d);

									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_epu64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										_DESTi = _mm256_rem_epu64(_Ai, _Bi);
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(
											_mm256_castsi256_pd(_Ai),
											_mm256_castsi256_pd(_Bi)
										);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256i _TWO = _mm256_set1_epi64x(2);

										_DESTi = _mm256_add_epi64(_Ai, _Bi);
										_DESTi = _mm256_div_epu64(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										break;
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
								}//End if hasAVX2
#endif							
							}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _Ai, _DESTi;
								const __m512 _Bi = _mm512_set1_ps(_val);

								for (; i != end; i += 16)
								{

									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if		
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512 _TWO = _mm512_set1_ps(2.0f);

										_DESTi = _mm512_add_ps(_Ai, _Bi);
										_DESTi = _mm512_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm512_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm512_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _DESTi;
								const __m256 _Bi = _mm256_set1_ps(_val);

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mul_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if		
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256 _TWO = _mm256_set1_ps(2.0f);

										_DESTi = _mm256_add_ps(_Ai, _Bi);
										_DESTi = _mm256_div_ps(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm256_pow_ps(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm256_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif						
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _Ai, _DESTi;
								const __m512d _Bi = _mm512_set1_pd(_val);

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm512_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm512_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm512_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm512_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm512_min_pd(_Ai, _Bi);
								}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm512_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m512d _TWO = _mm512_set1_pd(2);

										_DESTi = _mm512_add_pd(_Ai, _Bi);
										_DESTi = _mm512_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm512_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm512_store_pd(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _DESTi;
								const __m256d _Bi = _mm256_set1_pd(_val);

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);
									
									//Calc
									if constexpr (INSTR == tpa::op::ADD)
									{
										_DESTi = _mm256_add_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::SUBTRACT)
									{
										_DESTi = _mm256_sub_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MULTIPLY)
									{
										_DESTi = _mm256_mul_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::DIVIDE)
									{
										_DESTi = _mm256_div_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MODULO)
									{
										break; //Use fmod
									}//End if									
									else if constexpr (INSTR == tpa::op::MIN)
									{
										_DESTi = _mm256_min_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::MAX)
									{
										_DESTi = _mm256_max_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::op::AVERAGE)
									{
										const __m256d _TWO = _mm256_set1_pd(2);

										_DESTi = _mm256_add_pd(_Ai, _Bi);
										_DESTi = _mm256_div_pd(_DESTi, _TWO);
									}//End if
									else if constexpr (INSTR == tpa::op::POWER)
									{
										_DESTi = _mm256_pow_pd(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm256_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif						
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Calc
							if constexpr (INSTR == tpa::op::ADD)
							{
								dest[i] = source1[i] + _val;
							}//End if
							else if constexpr (INSTR == tpa::op::SUBTRACT)
							{
								dest[i] = source1[i] - _val;
							}//End if
							else if constexpr (INSTR == tpa::op::MULTIPLY)
							{
								dest[i] = source1[i] * _val;
							}//End if
							else if constexpr (INSTR == tpa::op::DIVIDE)
							{
								dest[i] = source1[i] / _val;
							}//End if
							else if constexpr (INSTR == tpa::op::MODULO)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = static_cast<T>(std::fmod(source1[i], _val));
								}//End if
								else
								{
									dest[i] = source1[i] % _val;
								}//End else
							}//end if							
							else if constexpr (INSTR == tpa::op::MIN)
							{
								dest[i] = tpa::util::min(source1[i], _val);
							}//End if
							else if constexpr (INSTR == tpa::op::MAX)
							{
								dest[i] = tpa::util::max(source1[i], _val);
							}//End if
							else if constexpr (INSTR == tpa::op::AVERAGE)
							{
								dest[i] = static_cast<T>((source1[i] + _val) / 2);
							}//End if
							else if constexpr (INSTR == tpa::op::POWER)
							{
								dest[i] = tpa::util::pow(source1[i], _val);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::calculate<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
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
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::calculate: unknown!\n";
		}//End catch
	}//End of calculate_const()

	/// <summary>
	/// <para>Compares 2 std::vectors and stores the results of the expression in a 3rd std::vector.</para>
	/// <para>Example: if source1[0] > source2[0]...dest[0] = source1[0] otherwise unchanged.</para>
	/// <para>Optionally you can remove zeros, false by default</para>
	/// <para>Templated predicate takes 1 of these predicates: tpa::comp</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::comp::EQUAL_TO</para>			
	/// <para>tpa::comp::NOT_EQUAL_TO</para>		
	/// <para>tpa::comp::LESS_THAN</para>		
	/// <para>tpa::comp::LESS_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::GREATER_THAN</para>			
	/// <para>tpa::comp::GREATER_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::MIN</para>
	/// <para>tpa::comp::MAX</para>
	/// </summary>
	/// <typeparam name="CONTAINER"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::comp INSTR, typename T>
	inline constexpr void compare(
		const std::vector<T>& source1,
		const std::vector<T>& source2,
		std::vector<T>& dest,
		const bool removeZeros = false)
	{
		try
		{
			size_t smallest = 0;
			uint32_t complete = 0;

			//Determin the smallest container
			smallest = tpa::util::min(source1.size(), source2.size());

			if (dest.size() < smallest)
			{
				dest.resize(smallest);
			}//End if

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
#pragma region byte
					if constexpr (std::is_same<T, int8_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _Bi, _dest;
							__mmask64 _mask;

							for (; i != end; i += 64)
							{
								if ((i + 64) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm512_loadu_epi8(&source1[i]);
								_Bi = _mm512_loadu_epi8(&source2[i]);
								_dest = _mm512_setzero_si512();
								_mask = 0;

								//Compare
								if constexpr (INSTR == tpa::comp::GREATER_THAN)
								{
									_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GT);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
								{
									_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GE);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN)
								{
									_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LT);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
								{
									_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LE);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::EQUAL)
								{
									_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
								{
									_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_NE);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::MIN)
								{
									_dest = _mm512_min_epi8(_Ai, _Bi);
									_mm512_storeu_epi8(&dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::MAX)
								{
									_dest = _mm512_max_epi8(_Ai, _Bi);
									_mm512_storeu_epi8(&dest[i], _dest);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
									}();
								}//End else
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _dest, _mask;

							for (; i != end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);
								_dest = _mm256_setzero_si256();
								_mask = _mm256_setzero_si256();

								//Compare
								if constexpr (INSTR == tpa::comp::GREATER_THAN)
								{
									_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
									_dest = _mm256_and_si256(_Ai, _mask);

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
								{
									_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
									_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

									_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN)
								{
									_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
									_dest = _mm256_and_si256(_Ai, _mask);

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
								{
									_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
									_dest = _mm256_andnot_si256(_mask, _Ai);

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::EQUAL)
								{
									_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
									_dest = _mm256_and_si256(_Ai, _mask);

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
								{
									_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
									_dest = _mm256_andnot_si256(_mask, _Ai);

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::MIN)
								{
									_dest = _mm256_min_epi8(_Ai, _Bi);
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::MAX)
								{
									_dest = _mm256_max_epi8(_Ai, _Bi);
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
									}();
								}//End else
							}//End for
						}//End if AVX2
	#endif
					}//End if
#pragma endregion
#pragma region unsigned byte
					else if constexpr (std::is_same<T, uint8_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _Bi, _dest;
							__mmask64 _mask;

							for (; i != end; i += 64)
							{
								if ((i + 64) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm512_loadu_epi8(&source1[i]);
								_Bi = _mm512_loadu_epi8(&source2[i]);
								_dest = _mm512_setzero_si512();
								_mask = 0;

								//Compare
								if constexpr (INSTR == tpa::comp::GREATER_THAN)
								{
									_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GT);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
								{
									_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GE);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN)
								{
									_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LT);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
								{
									_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LE);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::EQUAL)
								{
									_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
								{
									_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_NE);
									_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::MIN)
								{
									_dest = _mm512_min_epu8(_Ai, _Bi);
									_mm512_storeu_epi8(&dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::MAX)
								{
									_dest = _mm512_max_epu8(_Ai, _Bi);
									_mm512_storeu_epi8(&dest[i], _dest);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
									}();
								}//End else
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _dest, _mask;

							for (; i != end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);
								_dest = _mm256_setzero_si256();
								_mask = _mm256_setzero_si256();

								//Compare
								if constexpr (INSTR == tpa::comp::GREATER_THAN)
								{
									_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
									_dest = _mm256_and_si256(_Ai, _mask);

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
								{
									_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
									_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

									_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN)
								{
									_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
									_dest = _mm256_and_si256(_Ai, _mask);

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
								{
									_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
									_dest = _mm256_andnot_si256(_mask, _Ai);

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::EQUAL)
								{
									_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
									_dest = _mm256_and_si256(_Ai, _mask);

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
								{
									_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
									_dest = _mm256_andnot_si256(_mask, _Ai);

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::MIN)
								{
									_dest = _mm256_min_epu8(_Ai, _Bi);
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::MAX)
								{
									_dest = _mm256_max_epu8(_Ai, _Bi);
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
									}();
								}//End else
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region short
					else if constexpr (std::is_same<T, int16_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _Bi, _dest;
							__mmask32 _mask;

							for (; i != end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm512_loadu_epi16(&source1[i]);
								_Bi = _mm512_loadu_epi16(&source2[i]);
								_dest = _mm512_setzero_si512();
								_mask = 0;

								//Compare
								if constexpr (INSTR == tpa::comp::GREATER_THAN)
								{
									_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GT);
									_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
								{
									_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GE);
									_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN)
								{
									_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LT);
									_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
								{
									_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LE);
									_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::EQUAL)
								{
									_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
									_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
								{
									_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_NE);
									_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::MIN)
								{
									_dest = _mm512_min_epi16(_Ai, _Bi);
									_mm512_storeu_epi16(&dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::MAX)
								{
									_dest = _mm512_max_epi16(_Ai, _Bi);
									_mm512_storeu_epi16(&dest[i], _dest);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
									}();
								}//End else
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _dest, _mask;

							for (; i != end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source1[i]);
								_Bi = _mm256_load_si256((__m256i*) & source2[i]);
								_dest = _mm256_setzero_si256();
								_mask = _mm256_setzero_si256();

								//Compare
								if constexpr (INSTR == tpa::comp::GREATER_THAN)
								{
									_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
									_dest = _mm256_and_si256(_Ai, _mask);

									_mm256_store_si256((__m256i*)& dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
								{
									_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
									_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

									_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

									_mm256_store_si256((__m256i*)& dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN)
								{
									_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
									_dest = _mm256_and_si256(_Ai, _mask);

									_mm256_store_si256((__m256i*)& dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
								{
									_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
									_dest = _mm256_andnot_si256(_mask, _Ai);

									_mm256_store_si256((__m256i*)& dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::EQUAL)
								{
									_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
									_dest = _mm256_and_si256(_Ai, _mask);

									_mm256_store_si256((__m256i*)& dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
								{
									_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
									_dest = _mm256_andnot_si256(_mask, _Ai);

									_mm256_store_si256((__m256i*)& dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::MIN)
								{
									_dest = _mm256_min_epi16(_Ai, _Bi);
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::MAX)
								{
									_dest = _mm256_max_epi16(_Ai, _Bi);
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
									}();
								}//End else
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned short
					else if constexpr (std::is_same<T, uint16_t>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512_ByteWord)
					{
						__m512i _Ai, _Bi, _dest;
						__mmask32 _mask;

						for (; i != end; i += 32)
						{
							if ((i + 32) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_loadu_epi16(&source1[i]);
							_Bi = _mm512_loadu_epi16(&source2[i]);
							_dest = _mm512_setzero_si512();
							_mask = 0;

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GT);
								_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GE);
								_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LT);
								_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LE);
								_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
								_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_NE);
								_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm512_min_epu16(_Ai, _Bi);
								_mm512_storeu_epi16(&dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm512_max_epu16(_Ai, _Bi);
								_mm512_storeu_epi16(&dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if hasAVX512
					else if (tpa::hasAVX2)
					{
						__m256i _Ai, _Bi, _dest, _mask;

						for (; i != end; i += 16)
						{
							if ((i + 16) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_si256((__m256i*) & source1[i]);
							_Bi = _mm256_load_si256((__m256i*) & source2[i]);
							_dest = _mm256_setzero_si256();
							_mask = _mm256_setzero_si256();

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
								_dest = _mm256_and_si256(_Ai, _mask);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
								_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

								_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
								_dest = _mm256_and_si256(_Ai, _mask);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
								_dest = _mm256_andnot_si256(_mask, _Ai);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
								_dest = _mm256_and_si256(_Ai, _mask);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
								_dest = _mm256_andnot_si256(_mask, _Ai);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm256_min_epu16(_Ai, _Bi);
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm256_max_epu16(_Ai, _Bi);
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region int
					else if constexpr (std::is_same<T, int32_t>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512i _Ai, _Bi, _dest;
						__mmask16 _mask;

						for (; i != end; i += 16)
						{
							if ((i + 16) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_load_epi32(&source1[i]);
							_Bi = _mm512_load_epi32(&source2[i]);
							_dest = _mm512_setzero_si512();
							_mask = 0;

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GT);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GE);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LT);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LE);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_NE);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm512_min_epi32(_Ai, _Bi);
								_mm512_store_epi32(&dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm512_max_epi32(_Ai, _Bi);
								_mm512_store_epi32(&dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if hasAVX512
					else if (tpa::hasAVX2)
					{
						__m256i _Ai, _Bi, _dest, _mask;

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_si256((__m256i*) & source1[i]);
							_Bi = _mm256_load_si256((__m256i*) & source2[i]);
							_dest = _mm256_setzero_si256();
							_mask = _mm256_setzero_si256();

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i],_mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
								_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

								_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
								_dest = _mm256_andnot_si256(_mask, _Ai);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
								_dest = _mm256_andnot_si256(_mask, _Ai);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm256_min_epi32(_Ai, _Bi);
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm256_max_epi32(_Ai, _Bi);
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if AVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned int
					else if constexpr (std::is_same<T, uint32_t>() == true)
					{
#ifdef _M_AMD64	
					if (tpa::hasAVX512)
					{
						__m512i _Ai, _Bi, _dest;
						__mmask16 _mask;

						for (; i != end; i += 16)
						{
							if ((i + 16) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_load_epi32(&source1[i]);
							_Bi = _mm512_load_epi32(&source2[i]);
							_dest = _mm512_setzero_si512();
							_mask = 0;

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GT);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GE);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LT);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LE);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_NE);
								_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm512_min_epu32(_Ai, _Bi);
								_mm512_store_epi32(&dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm512_max_epu32(_Ai, _Bi);
								_mm512_store_epi32(&dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if hasAVX512
					else if (tpa::hasAVX2)
					{
						__m256i _Ai, _Bi, _dest, _mask;

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_si256((__m256i*) & source1[i]);
							_Bi = _mm256_load_si256((__m256i*) & source2[i]);
							_dest = _mm256_setzero_si256();
							_mask = _mm256_setzero_si256();

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
								_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

								_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
								_dest = _mm256_andnot_si256(_mask, _Ai);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
								_dest = _mm256_andnot_si256(_mask, _Ai);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm256_min_epi32(_Ai, _Bi);
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm256_max_epi32(_Ai, _Bi);
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if AVX2
#endif							
					}//End if
#pragma endregion
#pragma region long
					else if constexpr (std::is_same<T, int64_t>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512i _Ai, _Bi, _dest;
						__mmask8 _mask;

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_load_epi64(&source1[i]);
							_Bi = _mm512_load_epi64(&source2[i]);
							_dest = _mm512_setzero_si512();
							_mask = 0;

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GT);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GE);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LT);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LE);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_NE);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm512_min_epi64(_Ai, _Bi);
								_mm512_store_epi64(&dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm512_max_epi64(_Ai, _Bi);
								_mm512_store_epi64(&dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if hasAVX512
					else if (tpa::hasAVX2)
					{
						__m256i _Ai, _Bi, _dest, _mask;

						for (; i != end; i += 4)
						{
							if ((i + 4) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_si256((__m256i*) & source1[i]);
							_Bi = _mm256_load_si256((__m256i*) & source2[i]);
							_dest = _mm256_setzero_si256();
							_mask = _mm256_setzero_si256();

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
								_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

								_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
								_dest = _mm256_andnot_si256(_mask, _Ai);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
								_dest = _mm256_andnot_si256(_mask, _Ai);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm256_min_epi64(_Ai, _Bi);
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm256_max_epi64(_Ai, _Bi);
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if AVX2
#endif							
					}//End if
#pragma endregion
#pragma region unsigned long
					else if constexpr (std::is_same<T, uint64_t>() == true)
					{
#ifdef _M_AMD64	
					if (tpa::hasAVX512)
					{
						__m512i _Ai, _Bi, _dest;
						__mmask8 _mask;

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_load_epi64(&source1[i]);
							_Bi = _mm512_load_epi64(&source2[i]);
							_dest = _mm512_setzero_si512();
							_mask = 0;

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GT);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GE);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LT);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LE);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_NE);
								_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm512_min_epu64(_Ai, _Bi);
								_mm512_store_epi64(&dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm512_max_epu64(_Ai, _Bi);
								_mm512_store_epi64(&dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if hasAVX512
					else if (tpa::hasAVX2)
					{
						__m256i _Ai, _Bi, _dest, _mask;

						for (; i != end; i += 4)
						{
							if ((i + 4) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_si256((__m256i*) & source1[i]);
							_Bi = _mm256_load_si256((__m256i*) & source2[i]);
							_dest = _mm256_setzero_si256();
							_mask = _mm256_setzero_si256();

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
								_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

								_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
								_dest = _mm256_andnot_si256(_mask, _Ai);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
								_dest = _mm256_andnot_si256(_mask, _Ai);

								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm256_min_epu64(_Ai, _Bi);
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm256_max_epu64(_Ai, _Bi);
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if AVX2
#endif
					}//End if
#pragma endregion
#pragma region float
					else if constexpr (std::is_same<T, float>() == true)
					{
#ifdef _M_AMD64			
						if (tpa::hasAVX512)
						{
							__m512 _Ai, _Bi, _dest;
							__mmask16 _mask;

							for (; i != end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm512_load_ps(&source1[i]);
								_Bi = _mm512_load_ps(&source2[i]);
								_dest = _mm512_setzero_ps();
								_mask = 0;

								//Compare
								if constexpr (INSTR == tpa::comp::GREATER_THAN)
								{
									_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GT_OQ);
									_mm512_mask_store_ps(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
								{
									_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GE_OQ);
									_mm512_mask_store_ps(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN)
								{
									_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LT_OQ);
									_mm512_mask_store_ps(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
								{
									_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LE_OQ);
									_mm512_mask_store_ps(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::EQUAL)
								{
									_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_EQ_OQ);
									_mm512_mask_store_ps(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
								{
									_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_NEQ_OQ);
									_mm512_mask_store_ps(&dest[i], _mask, _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::MIN)
								{
									_dest = _mm512_min_ps(_Ai, _Bi);
									_mm512_store_ps(&dest[i], _dest);
								}//End if
								else if constexpr (INSTR == tpa::comp::MAX)
								{
									_dest = _mm512_max_ps(_Ai, _Bi);
									_mm512_store_ps(&dest[i], _dest);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
									}();
								}//End else
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							__m256 _Ai, _Bi, _mask;

							for (; i != end; i += 8)
							{

								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm256_load_ps(&source1[i]);
								_Bi = _mm256_load_ps(&source2[i]);

								//Compare
								if constexpr (INSTR == tpa::comp::GREATER_THAN)
								{
									_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GT_OQ);
									_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
								{
									_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GE_OQ);
									_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN)
								{
									_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LT_OQ);
									_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
								{
									_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LE_OQ);
									_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::EQUAL)
								{
									_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_EQ_OQ);
									_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
								{
									_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_NEQ_OQ);
									_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
								}//End if
								else if constexpr (INSTR == tpa::comp::MIN)
								{
									_mask = _mm256_min_ps(_Ai, _Bi);
									_mm256_store_ps(&dest[i], _mask);
								}//End if
								else if constexpr (INSTR == tpa::comp::MAX)
								{
									_mask = _mm256_max_ps(_Ai, _Bi);
									_mm256_store_ps(&dest[i], _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else
							}//End for
						}//End if has AVX2
#endif							
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512d _Ai, _Bi, _dest;
						__mmask8 _mask;

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_load_pd(&source1[i]);
							_Bi = _mm512_load_pd(&source2[i]);
							_dest = _mm512_setzero_pd();
							_mask = 0;

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GT_OQ);
								_mm512_mask_store_pd(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GE_OQ);
								_mm512_mask_store_pd(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LT_OQ);
								_mm512_mask_store_pd(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LE_OQ);
								_mm512_mask_store_pd(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_EQ_OQ);
								_mm512_mask_store_pd(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_NEQ_OQ);
								_mm512_mask_store_pd(&dest[i], _mask, _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_dest = _mm512_min_pd(_Ai, _Bi);
								_mm512_store_pd(&dest[i], _dest);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_dest = _mm512_max_pd(_Ai, _Bi);
								_mm512_store_pd(&dest[i], _dest);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
								}();
							}//End else
						}//End for
					}//End if hasAVX512
					else if (tpa::hasAVX)
					{
						__m256d _Ai, _Bi, _mask;

						for (; i != end; i += 4)
						{

							if ((i + 4) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_pd(&source1[i]);
							_Bi = _mm256_load_pd(&source2[i]);

							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GT_OQ);
								_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GE_OQ);
								_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LT_OQ);
								_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LE_OQ);
								_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_EQ_OQ);
								_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_NEQ_OQ);
								_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								_mask = _mm256_min_pd(_Ai, _Bi);
								_mm256_store_pd(&dest[i], _mask);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								_mask = _mm256_max_pd(_Ai, _Bi);
								_mm256_store_pd(&dest[i], _mask);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
								}();
							}//End else
						}//End for
					}//End if hasAVX2
#endif							
				}//End if
#pragma endregion
#pragma region generic
					for (; i != end; ++i)
					{
						//Compare
						if constexpr (INSTR == tpa::comp::GREATER_THAN)
						{
							if (source1[i] > source2[i])
							{
								dest[i] = source1[i];
							}//End if
						}//End if
						else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
						{
							if (source1[i] >= source2[i])
							{
								dest[i] = source1[i];
							}//End if
						}//End if
						else if constexpr (INSTR == tpa::comp::LESS_THAN)
						{
							if (source1[i] < source2[i])
							{
								dest[i] = source1[i];
							}//End if
						}//End if
						else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
						{
							if (source1[i] <= source2[i])
							{
								dest[i] = source1[i];
							}//End if
						}//End if
						else if constexpr (INSTR == tpa::comp::EQUAL)
						{
							if (source1[i] == source2[i])
							{
								dest[i] = source1[i];
							}//End if
						}//End if
						else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
						{
							if (source1[i] != source2[i])
							{
								dest[i] = source1[i];
							}//End if
						}//End if
						else if constexpr (INSTR == tpa::comp::MIN)
						{
							dest[i] = tpa::util::min(source1[i], source2[i]);
						}//End if
						else if constexpr (INSTR == tpa::comp::MAX)
						{
							dest[i] = tpa::util::max(source1[i], source2[i]);
						}//End if
						else
						{
							[] <bool flag = false>()
							{
								static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
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

			if (removeZeros == true)
			{
				//Remove all zeros from dest
				dest.erase(std::remove_if(std::execution::par_unseq, dest.begin(), dest.end(), [](T x) {return x == static_cast<T>(0); }));
			}//End if

		}//End try
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare: unknown!\n";
		}//End catch
	}//End of compare()

	/// <summary>
	/// <para>Compares 1 std::vector and a constant then stores the results of the expression in an std::vector.</para>
	/// <para>Example: if source1[0] > const_val...dest[0] = source1[0] otherwise unchanged.</para>
	/// <para>Optionally you can remove zeros, false by default</para>
	/// <para>Templated predicate takes 1 of these predicates: tpa::comp</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::comp::EQUAL_TO</para>			
	/// <para>tpa::comp::NOT_EQUAL_TO</para>		
	/// <para>tpa::comp::LESS_THAN</para>		
	/// <para>tpa::comp::LESS_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::GREATER_THAN</para>			
	/// <para>tpa::comp::GREATER_THAN_OR_EQUAL_TO</para>
	/// <para>tpa::comp::MIN</para>
	/// <para>tpa::comp::MAX</para>
	/// </summary>
	/// <typeparam name="CONTAINER"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::comp INSTR, typename T>
	inline constexpr void compare_const(
		const std::vector<T>& source1,
		const T val,
		std::vector<T>& dest,
		const bool removeZeros = false)
	{
		try
		{
			uint32_t complete = 0;

			if (dest.size() < source1.size())
			{
				dest.resize(source1.size());
			}//End if

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, source1.size());

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &val, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T _val = val;//Copy for thread-locality
#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask64 _mask;

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8(&source1[i]);
									_Bi = _mm512_set1_epi8(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi8_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi8(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask64 _mask;

								for (; i != end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi8(&source1[i]);
									_Bi = _mm512_set1_epi8(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu8_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi8(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu8(_Ai, _Bi);
										_mm512_storeu_epi8(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi8(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi8(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi8(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi8(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu8(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask32 _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16(&source1[i]);
									_Bi = _mm512_set1_epi16(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi16_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi16(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask32 _mask;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16(&source1[i]);
									_Bi = _mm512_set1_epi16(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu16_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_storeu_epi16(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu16(_Ai, _Bi);
										_mm512_storeu_epi16(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi16(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi16(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi16(_Bi, _Ai);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_and_si256(_Ai, _mask);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi16(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu16(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32(&source1[i]);
									_Bi = _mm512_set1_epi32(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi32_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi32(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64	
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32(&source1[i]);
									_Bi = _mm512_set1_epi32(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu32_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu32(_Ai, _Bi);
										_mm512_store_epi32(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi32(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi32(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi32(_Bi, _Ai);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);											_mm256_maskstore_epi32(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi32(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi32(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64(&source1[i]);
									_Bi = _mm512_set1_epi64(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epi64_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epi64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epi64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi64x(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epi64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epi64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64	
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64(&source1[i]);
									_Bi = _mm512_set1_epi64(_val);
									_dest = _mm512_setzero_si512();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_GE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LT);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_LE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_EQ);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_epu64_mask(_Ai, _Bi, _MM_CMPINT_NE);
										_mm512_mask_store_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_epu64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_epu64(_Ai, _Bi);
										_mm512_store_epi64(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _dest, _mask;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_set1_epi64x(_val);
									_dest = _mm256_setzero_si256();
									_mask = _mm256_setzero_si256();

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_cmpeq_epi64(_Ai, _Bi);

										_dest = _mm256_and_si256(_Ai, _mm256_or_si256(_mask, _dest));

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmpgt_epi64(_Bi, _Ai);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmpgt_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);											_mm256_maskstore_epi64(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmpeq_epi64(_Ai, _Bi);
										_dest = _mm256_andnot_si256(_mask, _Ai);

										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm256_min_epu64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm256_max_epu64(_Ai, _Bi);
										_mm256_store_si256((__m256i*) & dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if AVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64			
							if (tpa::hasAVX512)
							{
								__m512 _Ai, _Bi, _dest;
								__mmask16 _mask;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);
									_Bi = _mm512_set1_ps(_val);
									_dest = _mm512_setzero_ps();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GT_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_GE_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LT_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_LE_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_EQ_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_ps_mask(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm512_mask_store_ps(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_ps(_Ai, _Bi);
										_mm512_store_ps(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_ps(_Ai, _Bi);
										_mm512_store_ps(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _Bi, _mask;

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);
									_Bi = _mm256_set1_ps(_val);

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GT_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_GE_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LT_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_LE_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_EQ_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmp_ps(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm256_maskstore_ps(&dest[i], _mm256_castps_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_mask = _mm256_min_ps(_Ai, _Bi);
										_mm256_store_ps(&dest[i], _mask);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_mask = _mm256_max_ps(_Ai, _Bi);
										_mm256_store_ps(&dest[i], _mask);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else
								}//End for
							}//End if has AVX2
#endif							
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _Ai, _Bi, _dest;
								__mmask8 _mask;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);
									_Bi = _mm512_set1_pd(_val);
									_dest = _mm512_setzero_pd();
									_mask = 0;

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GT_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_GE_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LT_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_LE_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_EQ_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm512_cmp_pd_mask(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm512_mask_store_pd(&dest[i], _mask, _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_dest = _mm512_min_pd(_Ai, _Bi);
										_mm512_store_pd(&dest[i], _dest);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_dest = _mm512_max_pd(_Ai, _Bi);
										_mm512_store_pd(&dest[i], _dest);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _Bi, _mask;

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);
									_Bi = _mm256_set1_pd(_val);

									//Compare
									if constexpr (INSTR == tpa::comp::GREATER_THAN)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GT_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_GE_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LT_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_LE_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_EQ_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
									{
										_mask = _mm256_cmp_pd(_Ai, _Bi, _CMP_NEQ_OQ);
										_mm256_maskstore_pd(&dest[i], _mm256_castpd_si256(_mask), _Ai);
									}//End if
									else if constexpr (INSTR == tpa::comp::MIN)
									{
										_mask = _mm256_min_pd(_Ai, _Bi);
										_mm256_store_pd(&dest[i], _mask);
									}//End if
									else if constexpr (INSTR == tpa::comp::MAX)
									{
										_mask = _mm256_max_pd(_Ai, _Bi);
										_mm256_store_pd(&dest[i], _mask);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Compare
							if constexpr (INSTR == tpa::comp::GREATER_THAN)
							{
								if (source1[i] > _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::GREATER_THAN_OR_EQUAL)
							{
								if (source1[i] >= _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN)
							{
								if (source1[i] < _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::LESS_THAN_OR_EQUAL)
							{
								if (source1[i] <= _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::EQUAL)
							{
								if (source1[i] == _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::NOT_EQUAL)
							{
								if (source1[i] != _val)
								{
									dest[i] = source1[i];
								}//End if
							}//End if
							else if constexpr (INSTR == tpa::comp::MIN)
							{
								dest[i] = tpa::util::min(source1[i], _val);
							}//End if
							else if constexpr (INSTR == tpa::comp::MAX)
							{
								dest[i] = tpa::util::max(source1[i], _val);
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::compare<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
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

			if (removeZeros == true)
			{
				//Remove all zeros from dest
				dest.erase(std::remove_if(std::execution::par_unseq, dest.begin(), dest.end(), [](T x) {return x == static_cast<T>(0); }));
			}//End if

		}//End try
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::compare_const: unknown!\n";
		}//End catch
	}//End of compare_const()
#pragma endregion
}//End of namespace