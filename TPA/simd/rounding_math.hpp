#pragma once
/*
*	Matrix SIMD functions for TPA Library
*	By: David Aaron Braun
*	2021-08-10
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

#include <fenv.h>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
	#include <float.h>
#if defined(_M_FP_PRECISE) || defined(_M_FP_STRICT) 
	#pragma fenv_access (on)
#endif
#else
	#pragma STDC FENV_ACCESS ON
#endif

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
#undef floor
#undef ceil
#undef round

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
	/// <para>Computes the Absolute Value (Distance from 0) on elements in 'source' and stores the results in 'dest'</para>
	/// <para>Uses Multi-threading and SIMD (where available).</para>
	/// <para>It is recommended to use containers with the same value type as this is required for SIMD.</para>
	/// <para>Both Standard and Non-Standard Floating Point Typers are supported.</para>
	/// <para>Floating-Point Exceptions are suppressed by default</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	/// <param name="suppress_exceptions"> true by default</param>
	template<typename CONTAINER_A, typename CONTAINER_B>
	inline constexpr void abs(
		const CONTAINER_A& source,
		CONTAINER_B& dest,
		const bool suppress_exceptions = true)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			using T = CONTAINER_A::value_type;
			using RES = CONTAINER_B::value_type;

			//Set scaler exception mode
			if (suppress_exceptions)
				tpa::exceptions::FPExceptionDisabler d;

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

#pragma region byte
					if constexpr (std::is_same<T, int8_t>() && std::is_same<RES, int8_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _num;

							for (; (i+64) < end; i += 64)
							{
								_num = _mm512_loadu_epi8(&source[i]);

								_num = _mm512_abs_epi8(_num);

								_mm512_storeu_epi8(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;

							for (; (i+32) < end; i += 32)
							{
								_num = _mm256_load_si256((__m256i*) & source[i]);

								_num = _mm256_abs_epi8(_num);

								_mm256_store_si256((__m256i*) & dest[i], _num);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSSE3)
						{
							__m128i _num;

							for (; (i + 16) < end; i += 16)
							{
								_num = _mm_load_si128(&source[i]);

								_num = _mm_abs_epi8(_num);

								_mm_store_si128((__m128i*) &dest[i], _num);
							}//End for
						}//End if has SSSE3
#endif
					}//End if
#pragma endregion
#pragma region unsigned byte
					else if constexpr (std::is_same<T, uint8_t>() && std::is_same<RES, uint8_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _num;

							for (; (i+64) < end; i += 64)
							{
								_num = _mm512_loadu_epi8(&source[i]);

								_num = _mm512_abs_epi8(_num);

								_mm512_storeu_epi8(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;

							for (; (i+32) < end; i += 32)
							{
								_num = _mm256_load_si256((__m256i*) & source[i]);

								_num = _mm256_abs_epi8(_num);

								_mm256_store_si256((__m256i*) & dest[i], _num);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSSE3)
						{
							__m128i _num;

							for (; (i + 16) < end; i += 16)
							{
								_num = _mm_load_si128(&source[i]);

								_num = _mm_abs_epi8(_num);

								_mm_store_si128((__m128i*) & dest[i], _num);
							}//End for
						}//End if has SSSE3
#endif
					}//End if
#pragma endregion
#pragma region short
					else if constexpr (std::is_same<T, int16_t>() && std::is_same<RES, int16_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _num;

							for (; (i+32) < end; i += 32)
							{
								_num = _mm512_loadu_epi16(&source[i]);

								_num = _mm512_abs_epi16(_num);

								_mm512_storeu_epi16(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;

							for (; (i+16) < end; i += 16)
							{
								_num = _mm256_load_si256((__m256i*) & source[i]);

								_num = _mm256_abs_epi16(_num);

								_mm256_store_si256((__m256i*) & dest[i], _num);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSSE3)
						{
							__m128i _num;

							for (; (i + 8) < end; i += 8)
							{
								_num = _mm_load_si128(&source[i]);

								_num = _mm_abs_epi16(_num);

								_mm_store_si128((__m128i*) & dest[i], _num);
							}//End for
						}//End if has SSSE3
#endif
					}//End if
#pragma endregion
#pragma region unsigned short
					else if constexpr (std::is_same<T, uint16_t>() && std::is_same<RES, uint16_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _num;

							for (; (i+32) < end; i += 32)
							{
								_num = _mm512_loadu_epi16(&source[i]);

								_num = _mm512_abs_epi16(_num);

								_mm512_storeu_epi16(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;

							for (; (i+16) < end; i += 16)
							{
								_num = _mm256_load_si256((__m256i*) & source[i]);

								_num = _mm256_abs_epi16(_num);

								_mm256_store_si256((__m256i*) & dest[i], _num);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSSE3)
						{
							__m128i _num;

							for (; (i + 8) < end; i += 8)
							{
								_num = _mm_load_si128(&source[i]);

								_num = _mm_abs_epi16(_num);

								_mm_store_si128((__m128i*) & dest[i], _num);
							}//End for
						}//End if has SSSE3
#endif
					}//End if
#pragma endregion
#pragma region int
					else if constexpr (std::is_same<T, int32_t>() && std::is_same<RES, int32_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _num;

							for (; (i+16) < end; i += 16)
							{
								_num = _mm512_load_epi32(&source[i]);

								_num = _mm512_abs_epi32(_num);

								_mm512_store_epi32(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;

							for (; (i+8) < end; i += 8)
							{
								_num = _mm256_load_si256((__m256i*)&source[i]);

								_num = _mm256_abs_epi32(_num);

								_mm256_store_si256((__m256i*)&dest[i], _num);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSSE3)
						{
							__m128i _num;

							for (; (i + 4) < end; i += 4)
							{
								_num = _mm_load_si128(&source[i]);

								_num = _mm_abs_epi32(_num);

								_mm_store_si128((__m128i*) & dest[i], _num);
							}//End for
						}//End if has SSSE3
#endif
					}//End if
#pragma endregion
#pragma region unsigned int
					else if constexpr (std::is_same<T, uint32_t>() && std::is_same<RES, uint32_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _num;

							for (; (i+16) < end; i += 16)
							{
								_num = _mm512_load_epi32(&source[i]);

								_num = _mm512_abs_epi32(_num);

								_mm512_store_epi32(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;

							for (; (i+8) < end; i += 8)
							{
								_num = _mm256_load_si256((__m256i*) &source[i]);

								_num = _mm256_abs_epi32(_num);

								_mm256_store_si256((__m256i*)&dest[i], _num);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSSE3)
						{
							__m128i _num;

							for (; (i + 4) < end; i += 4)
							{
								_num = _mm_load_si128(&source[i]);

								_num = _mm_abs_epi32(_num);

								_mm_store_si128((__m128i*) & dest[i], _num);
							}//End for
						}//End if has SSSE3
#endif
					}//End if
#pragma endregion
#pragma region long
					else if constexpr (std::is_same<T, int64_t>() && std::is_same<RES, int64_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _num;

							for (; (i+8) < end; i += 8)
							{
								_num = _mm512_load_epi64(&source[i]);

								_num = _mm512_abs_epi64(_num);

								_mm512_store_epi64(&dest[i], _num);
							}//End for
						}//End if hasAVX512
#endif
					}//End if
#pragma endregion
#pragma region unsigned long
					else if constexpr (std::is_same<T, uint64_t>() && std::is_same<RES, uint64_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _num;

							for (; (i+8) < end; i += 8)
							{
								_num = _mm512_load_epi64(&source[i]);

								_num = _mm512_abs_epi64(_num);

								_mm512_store_epi64(&dest[i], _num);
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
							__m512 _num;

							for (; (i+16) < end; i += 16)
							{
								_num = _mm512_load_ps(&source[i]);

								_num = _mm512_abs_ps(_num);

								_mm512_store_ps(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							__m256 _num;

							for (; (i + 8) < end; i += 8)
							{
								_num = _mm256_load_ps(&source[i]);

								_num = tpa::util::_mm256_abs_ps(_num);

								_mm256_store_ps(&dest[i], _num);
							}//End for
						}//End if hasAVX
						else if (tpa::has_SSE)
						{
							__m128 _num;

							for (; (i + 4) < end; i += 4)
							{
								_num = _mm_load_ps(&source[i]);

								_num = tpa::util::_mm_abs_ps(_num);

								_mm_store_ps(&dest[i], _num);
							}//End for
						}//End if has_SSE
#endif
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>() && std::is_same<RES, double>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512d _num;

							for (; (i+8) < end; i += 8)
							{
								_num = _mm512_load_pd(&source[i]);

								_num = _mm512_abs_pd(_num);

								_mm512_store_pd(&dest[i], _num);								
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							__m256d _num;

							for (; (i + 4) < end; i += 4)
							{
								_num = _mm256_load_pd(&source[i]);

								_num = tpa::util::_mm256_abs_pd(_num);

								_mm256_store_pd(&dest[i], _num);
							}//End for
						}//End if hasAVX
						else if (tpa::has_SSE2)
						{
							__m128d _num;

							for (; (i + 2) < end; i += 2)
							{
								_num = _mm_load_pd(&source[i]);

								_num = tpa::util::_mm_abs_pd(_num);

								_mm_store_pd(&dest[i], _num);
							}//End for
						}//End if has_SSE2
#endif
					}//End if
#pragma endregion
#pragma region generic
					for (; i != end; ++i)
					{
						dest[i] = static_cast<RES>(tpa::util::abs(source[i]));
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
			std::cerr << "Exception thrown in tpa::simd::abs: " << ex.what() << "\n";
			std::cerr << "tpa::simd::abs will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::abs(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::abs: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::abs: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::abs: unknown!\n";
		}//End catch
	}//End of abs()

	/// <summary>
	/// <para>Rounds elements in 'source' down to nearest integer and stores the results in 'dest'</para>
	/// <para>Uses Multi-threading and SIMD (where available).</para>
	/// <para>It is recommended to use containers with the same value type as this is required for SIMD.</para>
	/// <para>Using same-width types also allows for SIMD in SOME cases. i.e. float->int32_t or double->int64_t</para>
	/// <para>Floating-Point Exceptions are suppressed by default.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	/// <param name="suppress_exceptions"> true by default</param>
	template<typename CONTAINER_A, typename CONTAINER_B>
	inline constexpr void floor(
		const CONTAINER_A& source,
		CONTAINER_B& dest,
		const bool suppress_exceptions = true)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			using T = CONTAINER_A::value_type;
			using RES = CONTAINER_B::value_type;

			//Set scaler exception mode
			if (suppress_exceptions)
				tpa::exceptions::FPExceptionDisabler d;

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
#pragma region float
			 if constexpr (std::is_same<T, float>() && std::is_same<RES, float>())
			{
#ifdef _M_AMD64
				if (tpa::hasAVX512)
				{
					__m512 _num;

					for (; (i+16) < end; i += 16)
					{
						_num = _mm512_load_ps(&source[i]);

						_num = _mm512_floor_ps(_num);

						_mm512_store_ps(&dest[i], _num);
					}//End for
				}//End if hasAVX512
				else if (tpa::hasAVX)
				{
					__m256 _num;

					for (; (i+8) < end; i += 8)
					{
						_num = _mm256_load_ps(&source[i]);

						_num = _mm256_floor_ps(_num);

						_mm256_store_ps(&dest[i], _num);
					}//End for
				}//End if hasAVX
				else if (tpa::has_SSE41)
				{
					__m128 _num;

					for (; (i + 4) < end; i += 4)
					{
						_num = _mm_load_ps(&source[i]);

						_num = _mm_floor_ps(_num);

						_mm_store_ps(&dest[i], _num);
					}//End for
				}//End if has_SSE4.1
#endif
			 }//End if
#pragma endregion
#pragma region float-to-int32
			 if constexpr (std::is_same<T, float>() && std::is_same<RES, int32_t>())
			 {
#ifdef _M_AMD64
				 if (tpa::hasAVX512)
				 {
					 __m512 _num;
					 __m512i _dest;

					 for (; (i + 16) < end; i += 16)
					 {
						 _num = _mm512_load_ps(&source[i]);

						 _dest = _mm512_cvt_roundps_epi32(_num, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

						 _mm512_store_epi32(&dest[i], _dest);
					 }//End for
				 }//End if hasAVX512
				 else if (tpa::hasAVX)
				 {
					 __m256 _num;
					 __m256i _dest;

					 for (; (i + 8) < end; i += 8)
					 {
						 _num = _mm256_load_ps(&source[i]);

						 _dest = _mm256_cvtps_epi32(_mm256_floor_ps(_num));

						 _mm256_store_si256((__m256i*)&dest[i], _dest);
					 }//End for
				 }//End if hasAVX
				 else if (tpa::has_SSE41)
				 {
					 __m128 _num;
					 __m128i _dest;

					 for (; (i + 4) < end; i += 4)
					 {
						 _num = _mm_load_ps(&source[i]);

						 _dest = _mm_cvtps_epi32(_mm_floor_ps(_num));

						 _mm_store_si128((__m128i*)&dest[i], _dest);
					 }//End for
				 }//End if has_SSE4.1
#endif
			 }//End if
#pragma endregion
#pragma region float-to-uint32
			 if constexpr (std::is_same<T, float>() && std::is_same<RES, uint32_t>())
			 {
#ifdef _M_AMD64
				 if (tpa::hasAVX512)
				 {
					 __m512d _num;
					 __m512i _dest;

					 for (; (i + 16) < end; i += 16)
					 {
						 _num = _mm512_load_ps(&source[i]);

						 _dest = _mm512_cvt_roundps_epu32(_num, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

						 _mm512_store_epi32(&dest[i], _dest);
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
							__m512d _num;

							for (; (i+8) < end; i += 8)
							{
								_num = _mm512_load_pd(&source[i]);

								_num = _mm512_floor_pd(_num);

								_mm512_store_pd(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							__m256d _num;

							for (; (i+4) < end; i += 4)
							{
								_num = _mm256_load_pd(&source[i]);

								_num = _mm256_floor_pd(_num);

								_mm256_store_pd(&dest[i], _num);
							}//End for
						}//End if hasAVX
						else if (tpa::has_SSE41)
						{
							__m128d _num;

							for (; (i + 2) < end; i += 2)
							{
								_num = _mm_load_pd(&source[i]);

								_num = _mm_floor_pd(_num);

								_mm_store_pd(&dest[i], _num);
							}//End for
						}//End if has_SSE4.1
#endif
					}//End if
#pragma endregion
#pragma region double-to-int64
			 if constexpr (std::is_same<T, double>() && std::is_same<RES, int64_t>())
			 {
#ifdef _M_AMD64
				 if (tpa::hasAVX512_DWQW)
				 {
					 __m512d _num;
					 __m512i _dest;

					 for (; (i + 8) < end; i += 8)
					 {
						 _num = _mm512_load_pd(&source[i]);

						 _dest = _mm512_cvt_roundpd_epi64(_num, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

						 _mm512_store_epi64(&dest[i], _dest);
					 }//End for
				 }//End if hasAVX512	
#endif
			 }//End if
#pragma endregion
#pragma region double-to-uint64
			 if constexpr (std::is_same<T, double>() && std::is_same<RES, uint64_t>())
			 {
#ifdef _M_AMD64
				 if (tpa::hasAVX512_DWQW)
				 {
					 __m512d _num;
					 __m512i _dest;

					 for (; (i + 8) < end; i += 8)
					 {
						 _num = _mm512_load_pd(&source[i]);

						 _dest = _mm512_cvt_roundpd_epu64(_num, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

						 _mm512_store_epi64(&dest[i], _dest);
					 }//End for
				 }//End if hasAVX512	
#endif
			 }//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							if constexpr (std::is_integral<T>())
							{
								dest[i] = static_cast<RES>(source[i]);
							}//End if
							else
							{
								dest[i] = static_cast<RES>(std::floor(source[i]));
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
			std::cerr << "Exception thrown in tpa::simd::floor: " << ex.what() << "\n";
			std::cerr << "tpa::simd::floor will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::floor(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::floor: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::floor: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::floor: unknown!\n";
		}//End catch
	}//End of floor()

	/// <summary>
	/// <para>Rounds elements in 'source' up to nearest integer and stores the results in 'dest'</para>
	/// <para>Uses Multi-threading and SIMD (where available).</para>
	/// <para>It is recommended to use containers with the same value type as this is required for SIMD.</para>
	/// <para>Using same-width types also allows for SIMD in SOME cases. i.e. float->int32_t or double->int64_t</para>
	/// <para>Floating-Point Exceptions are suppressed by default.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	/// <param name="suppress_exceptions"> true by default</param>
	template<typename CONTAINER_A, typename CONTAINER_B>
	inline constexpr void ceil(
		const CONTAINER_A& source,
		CONTAINER_B& dest,
		const bool suppress_exceptions = true)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			using T = CONTAINER_A::value_type;
			using RES = CONTAINER_B::value_type;

			//Set scaler exception mode
			if (suppress_exceptions)
				tpa::exceptions::FPExceptionDisabler d;

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
#pragma region float
						if constexpr (std::is_same<T, float>() && std::is_same<RES, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _num;

								for (; (i+16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_num = _mm512_ceil_ps(_num);

									_mm512_store_ps(&dest[i], _num);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _num;

								for (; (i+8) < end; i += 8)
								{
									_num = _mm256_load_ps(&source[i]);

									_num = _mm256_ceil_ps(_num);

									_mm256_store_ps(&dest[i], _num);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE41)
							{
								__m128 _num;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm_load_ps(&source[i]);

									_num = _mm_ceil_ps(_num);

									_mm_store_ps(&dest[i], _num);
								}//End for
							}//End if has_SSE4.1
#endif
						}//End if
#pragma endregion
#pragma region float-to-int32
						if constexpr (std::is_same<T, float>() && std::is_same<RES, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _num;
								__m512i _dest;

								for (; (i + 16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_dest = _mm512_cvt_roundps_epi32(_num, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

									_mm512_store_epi32(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _num;
								__m256i _dest;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm256_load_ps(&source[i]);

									_dest = _mm256_cvtps_epi32(_mm256_ceil_ps(_num));

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE41)
							{
								__m128 _num;
								__m128i _dest;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm_load_ps(&source[i]);

									_dest = _mm_cvtps_epi32(_mm_ceil_ps(_num));

									_mm_store_si128((__m128i*) & dest[i], _dest);
								}//End for
							}//End if has_SSE4.1
#endif
						}//End if
#pragma endregion
#pragma region float-to-uint32
						if constexpr (std::is_same<T, float>() && std::is_same<RES, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _num;
								__m512i _dest;

								for (; (i + 16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_dest = _mm512_cvt_roundps_epu32(_num, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

									_mm512_store_epi32(&dest[i], _dest);
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
								__m512d _num;

								for (; (i+8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_num = _mm512_ceil_pd(_num);

									_mm512_store_pd(&dest[i], _num);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _num;

								for (; (i+4) < end; i += 4)
								{
									_num = _mm256_load_pd(&source[i]);

									_num = _mm256_ceil_pd(_num);

									_mm256_store_pd(&dest[i], _num);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE41)
							{
								__m128d _num;

								for (; (i + 2) < end; i += 2)
								{
									_num = _mm_load_pd(&source[i]);

									_num = _mm_ceil_pd(_num);

									_mm_store_pd(&dest[i], _num);
								}//End for
							}//End if has_SSE4.1
#endif
						}//End if
#pragma endregion
#pragma region double-to-int64
						if constexpr (std::is_same<T, double>() && std::is_same<RES, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_DWQW)
							{
								__m512d _num;
								__m512i _dest;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_dest = _mm512_cvt_roundpd_epi64(_num, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

									_mm512_store_epi64(&dest[i], _dest);
								}//End for
							}//End if hasAVX512	
#endif
						}//End if
#pragma endregion
#pragma region double-to-uint64
						if constexpr (std::is_same<T, double>() && std::is_same<RES, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_DWQW)
							{
								__m512d _num;
								__m512i _dest;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_dest = _mm512_cvt_roundpd_epu64(_num, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

									_mm512_store_epi64(&dest[i], _dest);
								}//End for
							}//End if hasAVX512	
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							if constexpr (std::is_integral<T>())
							{
								dest[i] = static_cast<RES>(source[i]);
							}//End if
							else
							{
								dest[i] = static_cast<RES>(std::ceil(source[i]));
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
			std::cerr << "Exception thrown in tpa::simd::ceil: " << ex.what() << "\n";
			std::cerr << "tpa::simd::ceil will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::ceil(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::ceil: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::ceil: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::ceil: unknown!\n";
		}//End catch
	}//End of ceil()

	/// <summary>
	/// <para>Rounds floating-point numbers stored in 'source' according to the selected rounding mode </para>
	/// <para>and stores the results in 'dest'</para>
	/// <para>Floating-point exceptions are suppressed by default.</para>
	/// <para>Non-standard floating-point types may be truncated or produce otherwise incorrect results</para>
	/// <para>Uses Multi-threading and SIMD (where available).</para>
	/// <para>It is reccomended to use containers with identical value_type as this is required for SIMD.</para>
	/// <para>Using same-width types also allows for SIMD in SOME cases. i.e. float->int32_t or double->int64_t</para>
	/// </summary>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	/// <param name="suppress_exceptions"> true by default </param>
	template<tpa::rnd MODE, typename CONTAINER_A, typename CONTAINER_B>
	inline constexpr void round(
		const CONTAINER_A& source,
		CONTAINER_B& dest,
		const bool suppress_exceptions = true)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			using T = CONTAINER_A::value_type;
			using RES = CONTAINER_B::value_type;

			//Set scaler rounding mode
			switch (MODE) {
				case tpa::rnd::DOWN:
					fesetround(FE_DOWNWARD);
					break;
				case tpa::rnd::UP:
					fesetround(FE_UPWARD);
					break;
				case tpa::rnd::NEAREST_INT:
					fesetround(FE_TONEAREST);
					break;
				case tpa::rnd::TRUNCATE_TO_ZERO:
					fesetround(FE_TOWARDZERO);
					break;
				default:
					fesetround(fegetround());
					break;
			}//End switch

			//Set scaler exception mode
			if (suppress_exceptions)
				tpa::exceptions::FPExceptionDisabler d;

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
				temp = tpa::tp->addTask([&source, &dest, &suppress_exceptions, &sec]()
					{
						const bool no_fp_ex = suppress_exceptions;//Floating-point exceptions are suppressed
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;
#pragma region float
					if constexpr (std::is_same<T, float>() && std::is_same<RES, float>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							if (no_fp_ex)
							{
								__m512 _num;

								for (; (i+16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_num = _mm512_roundscale_round_ps(_num, (int8_t)MODE, _MM_FROUND_NO_EXC);

									_mm512_store_ps(&dest[i], _num);
								}//End for
							}//End if no_fp_ex
							else
							{
								__m512 _num;

								for (; (i+16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_num = _mm512_roundscale_ps(_num, (int8_t)MODE);

									_mm512_store_ps(&dest[i], _num);
								}//End for
							}//End if
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							if (no_fp_ex)
							{
								__m256 _num;

								for (; (i+8) < end; i += 8)
								{
									_num = _mm256_load_ps(&source[i]);

									_num = _mm256_round_ps(_num, (int8_t)MODE | _MM_FROUND_NO_EXC);

									_mm256_store_ps(&dest[i], _num);
								}//End for
							}//End if no_fp_ex
							else
							{
								__m256 _num;

								for (; (i+8) < end; i += 8)
								{
									_num = _mm256_load_ps(&source[i]);

									_num = _mm256_round_ps(_num, (int8_t)MODE);

									_mm256_store_ps(&dest[i], _num);
								}//End for
							}//End else
						}//End if hasAVX
						else if (tpa::has_SSE41)
						{
							if (no_fp_ex)
							{
								__m128 _num;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm_load_ps(&source[i]);

									_num = _mm_round_ps(_num, (int8_t)MODE | _MM_FROUND_NO_EXC);

									_mm_store_ps(&dest[i], _num);
								}//End for
							}//End if no_fp_ex
							else
							{
								__m128 _num;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm_load_ps(&source[i]);

									_num = _mm_round_ps(_num, (int8_t)MODE);

									_mm_store_ps(&dest[i], _num);
								}//End for
							}//End else
						}//End if has_SSE4.1
#endif
					}//End if
#pragma endregion
#pragma region float-to-int32
					if constexpr (std::is_same<T, float>() && std::is_same<RES, int32_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							if (no_fp_ex)
							{
								__m512 _num;
								__m512i _dest;

								for (; (i + 16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_dest = _mm512_cvt_roundps_epi32(_num, (int8_t)MODE | _MM_FROUND_NO_EXC);

									_mm512_store_epi32(&dest[i], _dest);
								}//End for
							}//End if no except
							else
							{
								__m512 _num;
								__m512i _dest;

								for (; (i + 16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_dest = _mm512_cvt_roundps_epi32(_num, (int8_t)MODE);

									_mm512_store_epi32(&dest[i], _dest);
								}//End for
							}//End else
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							if (no_fp_ex)
							{
								__m256 _num;
								__m256i _dest;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm256_load_ps(&source[i]);

									_dest = _mm256_cvtps_epi32(_mm256_round_ps(_num, (int8_t)MODE | _MM_FROUND_NO_EXC));

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if no except
							else
							{
								__m256 _num;
								__m256i _dest;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm256_load_ps(&source[i]);

									_dest = _mm256_cvtps_epi32(_mm256_round_ps(_num, (int8_t)MODE));

									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End else
						}//End if hasAVX
						else if (tpa::has_SSE41)
						{
							if (no_fp_ex)
							{
								__m128 _num;
								__m128i _dest;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm_load_ps(&source[i]);

									_dest = _mm_cvtps_epi32(_mm_round_ps(_num, (int8_t)MODE | _MM_FROUND_NO_EXC));

									_mm_store_si128((__m128i*) & dest[i], _dest);
								}//End for
							}//End if no except
							else
							{
								__m128 _num;
								__m128i _dest;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm_load_ps(&source[i]);

									_dest = _mm_cvtps_epi32(_mm_round_ps(_num, (int8_t)MODE));

									_mm_store_si128((__m128i*) & dest[i], _dest);
								}//End for
							}//End else
						}//End if has_SSE4.1
#endif
					}//End if
#pragma endregion
#pragma region float-to-uint32
					if constexpr (std::is_same<T, float>() && std::is_same<RES, uint32_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							if (no_fp_ex)
							{
								__m512d _num;
								__m512i _dest;

								for (; (i + 16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_dest = _mm512_cvt_roundps_epu32(_num, (int8_t)MODE | _MM_FROUND_NO_EXC);

									_mm512_store_epi32(&dest[i], _dest);
								}//End for
							}//End if no except
							else
							{
								__m512d _num;
								__m512i _dest;

								for (; (i + 16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_dest = _mm512_cvt_roundps_epu32(_num, (int8_t)MODE);

									_mm512_store_epi32(&dest[i], _dest);
								}//End for
							}//End else
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
							if (no_fp_ex)
							{
								__m512d _num;

								for (; (i+8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_num = _mm512_roundscale_round_pd(_num, (int8_t)MODE, _MM_FROUND_NO_EXC);

									_mm512_store_pd(&dest[i], _num);
								}//End for
							}//End if no_fp_ex
							else
							{
								__m512d _num;

								for (; (i+8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_num = _mm512_roundscale_pd(_num, (int8_t)MODE);

									_mm512_store_pd(&dest[i], _num);
								}//End for
							}//End if
						}//End if hasAVX512
						else if (tpa::hasAVX)
						{
							if (no_fp_ex)
							{
								__m256d _num;

								for (; (i+4) < end; i += 4)
								{
									_num = _mm256_load_pd(&source[i]);

									_num = _mm256_round_pd(_num, (int8_t)MODE | _MM_FROUND_NO_EXC);

									_mm256_store_pd(&dest[i], _num);
								}//End for
							}//End if no_fp_ex
							else
							{
								__m256d _num;

								for (; (i+4) < end; i += 4)
								{
									_num = _mm256_load_pd(&source[i]);

									_num = _mm256_round_pd(_num, (int8_t)MODE);

									_mm256_store_pd(&dest[i], _num);
								}//End for
							}//End else
						}//End if hasAVX
						else if (tpa::has_SSE41)
						{
							if (no_fp_ex)
							{
								__m128d _num;

								for (; (i + 2) < end; i += 2)
								{
									_num = _mm_load_pd(&source[i]);

									_num = _mm_round_pd(_num, (int8_t)MODE | _MM_FROUND_NO_EXC);

									_mm_store_pd(&dest[i], _num);
								}//End for
							}//End if no_fp_ex
							else
							{
								__m128d _num;

								for (; (i + 2) < end; i += 2)
								{
									_num = _mm_load_pd(&source[i]);

									_num = _mm_round_pd(_num, (int8_t)MODE);

									_mm_store_pd(&dest[i], _num);
								}//End for
							}//End else
						}//End if has_SSE4.1
#endif
					}//End if
#pragma endregion
#pragma region double-to-int64
					if constexpr (std::is_same<T, double>() && std::is_same<RES, int64_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_DWQW)
						{
							if (no_fp_ex)
							{
								__m512d _num;
								__m512i _dest;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_dest = _mm512_cvt_roundpd_epi64(_num, (int8_t)MODE | _MM_FROUND_NO_EXC);

									_mm512_store_epi64(&dest[i], _dest);
								}//End for
							}//End if no except
							else
							{
								__m512d _num;
								__m512i _dest;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_dest = _mm512_cvt_roundpd_epi64(_num, (int8_t)MODE);

									_mm512_store_epi64(&dest[i], _dest);
								}//End for
							}//End else
						}//End if hasAVX512	
#endif
					}//End if
#pragma endregion
#pragma region double-to-uint64
					if constexpr (std::is_same<T, double>() && std::is_same<RES, uint64_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_DWQW)
						{
							if (no_fp_ex)
							{
								__m512d _num;
								__m512i _dest;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_dest = _mm512_cvt_roundpd_epu64(_num, (int8_t)MODE | _MM_FROUND_NO_EXC);

									_mm512_store_epi64(&dest[i], _dest);
								}//End for
							}//End if no except
							else
							{
								__m512d _num;
								__m512i _dest;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_dest = _mm512_cvt_roundpd_epu64(_num, (int8_t)MODE);

									_mm512_store_epi64(&dest[i], _dest);
								}//End for
							}//End else
						}//End if hasAVX512	
#endif
					}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							if constexpr (std::is_integral<T>())
							{
								dest[i] = static_cast<RES>(source[i]);
							}//End if
							else
							{
								if constexpr (MODE == tpa::rnd::DOWN)
								{
									dest[i] = static_cast<RES>(std::floor(source[i]));
								}//End if
								else if constexpr (MODE == tpa::rnd::UP)
								{
									dest[i] = static_cast<RES>(std::ceil(source[i]));
								}//End if
								else if constexpr (MODE == tpa::rnd::NEAREST_INT)
								{
									dest[i] = static_cast<RES>(std::round(source[i]));
								}//End if
								else if constexpr (MODE == tpa::rnd::TRUNCATE_TO_ZERO)
								{
									dest[i] = static_cast<RES>(std::trunc(source[i]));
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::round<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
									}();
								}//End else
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
			std::cerr << "Exception thrown in tpa::simd::round: " << ex.what() << "\n";
			std::cerr << "tpa::simd::round will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::round(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::round: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::round: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::round: unknown!\n";
		}//End catch
	}//End of round()

	/// <summary>
	/// <para>Rounds elements in 'source' towards the nearest multiple specified and stores the results in 'dest'</para>
	/// <para>Uses Multi-threading and SIMD (where available).</para>
	/// <para>It is reccomended to use containers with the same value type as this is required for SIMD.</para>
	/// <para>Using same-width value_types also allows for SIMD in SOME cases. i.e. float->int32_t or double->int64_t</para>
	/// <para>Negative Numbers may only round up.</para>
	/// <para>The multiple 'mult' must be an integral type or causes rounding errors.</para>
	/// <para>Floating-Point Exceptions are suppressed by default.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="NUM"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	/// <param name="mult"></param>
	/// <param name="suppress_exceptions"> true by default</param>
	template<typename CONTAINER_A, typename CONTAINER_B, typename NUM>
	inline constexpr void round_nearest(
		const CONTAINER_A& source,
		CONTAINER_B& dest,
		const NUM mult,
		const bool suppress_exceptions = true)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			using T = CONTAINER_A::value_type;
			using RES = CONTAINER_B::value_type;

			static_assert(!std::is_floating_point<NUM>(), "You cannot spcify a floating-point number as a multiple");

			//Set Scaler FP Rounding Mode
			fesetround(FE_TONEAREST);

			//Set scaler exception mode
			if (suppress_exceptions)
				tpa::exceptions::FPExceptionDisabler d;

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
				temp = tpa::tp->addTask([&source, &dest, &mult, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;
#pragma region byte
					if constexpr (std::is_same<T, int8_t>() && std::is_same<RES, int8_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _num;
							const __m512i _mult = _mm512_set1_epi8(static_cast<int8_t>(mult));
							const __m512i _half = _mm512_set1_epi8(static_cast<int8_t>(mult / 2));

							for (; (i+64) < end; i += 64)
							{
								_num = _mm512_loadu_epi8(&source[i]);

								_num = _mm512_add_epi8(_num, _half);
								_num = _mm512_div_epi8(_num, _mult);
								
								int8_t d[64] = {};
								for (size_t x = 0; x != 64; ++x)
								{
#ifdef _WIN32
									d[x] = static_cast<int8_t>(_num.m512i_i8[x] * _mult.m512i_i8[x]);
#else
									d[x] = static_cast<int8_t>(_num[x] * _mult[x]);
#endif
								}//End for
								_num = _mm512_loadu_epi8(&d);

								_mm512_storeu_epi8(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;
							const __m256i _mult = _mm256_set1_epi8(static_cast<int8_t>(mult));
							const __m256i _half = _mm256_set1_epi8(static_cast<int8_t>(mult / 2));

							for (; (i+32) < end; i += 32)
							{
								_num = _mm256_load_si256((__m256i*) & source[i]);

								_num = _mm256_add_epi8(_num, _half);
								_num = _mm256_div_epi8(_num, _mult);
								
								int8_t d[32] = {};
								for (size_t x = 0; x != 32; ++x)
								{
#ifdef _WIN32
									d[x] = static_cast<int8_t>(_num.m256i_i8[x] * _mult.m256i_i8[x]);
#else
									d[x] = static_cast<int8_t>(_num[x] * _mult[x]);
#endif
								}//End for
								_num = _mm256_load_si256((__m256i*) &d);

								_mm256_store_si256((__m256i*) & dest[i], _num);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned byte
					else if constexpr (std::is_same<T, uint8_t>() && std::is_same<RES, uint8_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _num;
							const __m512i _mult = _mm512_set1_epi8(static_cast<uint8_t>(mult));
							const __m512i _half = _mm512_set1_epi8(static_cast<uint8_t>(mult / 2));

							for (; (i+64) < end; i += 64)
							{
								_num = _mm512_loadu_epi8(&source[i]);

								_num = _mm512_add_epi8(_num, _half);
								_num = _mm512_div_epu8(_num, _mult);

								uint8_t d[64] = {};
								for (size_t x = 0; x != 64; ++x)
								{
#ifdef _WIN32
									d[x] = static_cast<uint8_t>(_num.m512i_u8[x] * _mult.m512i_u8[x]);
#else
									d[x] = static_cast<uint8_t>(_num[x] * _mult[x]);
#endif
								}//End for
								_num = _mm512_loadu_epi8(&d);

								_mm512_storeu_epi8(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;
							const __m256i _mult = _mm256_set1_epi8(static_cast<uint8_t>(mult));
							const __m256i _half = _mm256_set1_epi8(static_cast<uint8_t>(mult / 2));

							for (; (i+32) < end; i += 32)
							{
								_num = _mm256_load_si256((__m256i*) & source[i]);

								_num = _mm256_add_epi8(_num, _half);
								_num = _mm256_div_epu8(_num, _mult);

								uint8_t d[32] = {};
								for (size_t x = 0; x != 32; ++x)
								{
#ifdef _WIN32
									d[x] = static_cast<uint8_t>(_num.m256i_u8[x] * _mult.m256i_u8[x]);
#else
									d[x] = static_cast<uint8_t>(_num[x] * _mult[x]);
#endif
								}//End for
								_num = _mm256_load_si256((__m256i*) & d);

								_mm256_store_si256((__m256i*) & dest[i], _num);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region short
					else if constexpr (std::is_same<T, int16_t>() && std::is_same<RES, int16_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _num;
							const __m512i _mult = _mm512_set1_epi16(static_cast<int16_t>(mult));
							const __m512i _half = _mm512_set1_epi16(static_cast<int16_t>(mult / 2));

							for (; (i+32) < end; i += 32)
							{
								_num = _mm512_loadu_epi16(&source[i]);

								_num = _mm512_add_epi16(_num, _half);
								_num = _mm512_div_epi16(_num, _mult);
								_num = _mm512_mullo_epi16(_num, _mult);

								_mm512_storeu_epi16(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;
							const __m256i _mult = _mm256_set1_epi16(static_cast<int16_t>(mult));
							const __m256i _half = _mm256_set1_epi16(static_cast<int16_t>(mult / 2));

							for (; (i+16) < end; i += 16)
							{
								_num = _mm256_load_si256((__m256i*) & source[i]);

								_num = _mm256_add_epi16(_num, _half);
								_num = _mm256_div_epi16(_num, _mult);
								_num = _mm256_mullo_epi16(_num, _mult);

								_mm256_store_si256((__m256i*) & dest[i], _num);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned short
					else if constexpr (std::is_same<T, uint16_t>() && std::is_same<RES, uint16_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _num;
							const __m512i _mult = _mm512_set1_epi16(static_cast<uint16_t>(mult));
							const __m512i _half = _mm512_set1_epi16(static_cast<uint16_t>(mult / 2));

							for (; (i+32) < end; i += 32)
							{
								_num = _mm512_loadu_epi16(&source[i]);

								_num = _mm512_add_epi16(_num, _half);
								_num = _mm512_div_epu16(_num, _mult);
								_num = _mm512_mullo_epi16(_num, _mult);

								_mm512_storeu_epi16(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;
							const __m256i _mult = _mm256_set1_epi16(static_cast<uint16_t>(mult));
							const __m256i _half = _mm256_set1_epi16(static_cast<uint16_t>(mult / 2));

							for (; (i+16) < end; i += 16)
							{
								_num = _mm256_load_si256((__m256i*) & source[i]);

								_num = _mm256_add_epi16(_num, _half);
								_num = _mm256_div_epu16(_num, _mult);
								_num = _mm256_mullo_epi16(_num, _mult);

								_mm256_store_si256((__m256i*) & dest[i], _num);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region int
					else if constexpr (std::is_same<T, int32_t>() && std::is_same<RES, int32_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _num;
							const __m512i _mult = _mm512_set1_epi32(static_cast<int32_t>(mult));
							const __m512i _half = _mm512_set1_epi32(static_cast<int32_t>(mult / 2));

							for (; (i+16) < end; i += 16)
							{
								_num = _mm512_load_epi32(&source[i]);

								_num = _mm512_add_epi32(_num, _half);
								_num = _mm512_div_epi32(_num, _mult);
								_num = _mm512_mullo_epi32(_num, _mult);

								_mm512_store_epi32(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;
							const __m256i _mult = _mm256_set1_epi32(static_cast<int32_t>(mult));
							const __m256i _half = _mm256_set1_epi32(static_cast<int32_t>(mult / 2));

							for (; (i+8) < end; i += 8)
							{
								_num = _mm256_load_si256((__m256i*)&source[i]);
								
								_num = _mm256_add_epi32(_num, _half);
								_num = _mm256_div_epi32(_num, _mult);
								_num = _mm256_mullo_epi32(_num, _mult);

								_mm256_store_si256((__m256i*)&dest[i], _num);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned int
					else if constexpr (std::is_same<T, uint32_t>() && std::is_same<RES, uint32_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _num;
							const __m512i _mult = _mm512_set1_epi32(static_cast<uint32_t>(mult));
							const __m512i _half = _mm512_set1_epi32(static_cast<uint32_t>(mult / 2));

							for (; (i+16) < end; i += 16)
							{
								_num = _mm512_load_epi32(&source[i]);

								_num = _mm512_add_epi32(_num, _half);
								_num = _mm512_div_epu32(_num, _mult);
								_num = _mm512_mullo_epi32(_num, _mult);

								_mm512_store_epi32(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;
							const __m256i _mult = _mm256_set1_epi32(static_cast<uint32_t>(mult));
							const __m256i _half = _mm256_set1_epi32(static_cast<uint32_t>(mult / 2));

							for (; (i+8) < end; i += 8)
							{
								_num = _mm256_load_si256((__m256i*) & source[i]);

								_num = _mm256_add_epi32(_num, _half);
								_num = _mm256_div_epu32(_num, _mult);
								_num = _mm256_mullo_epi32(_num, _mult);

								_mm256_store_si256((__m256i*) & dest[i], _num);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region long
					else if constexpr (std::is_same<T, int64_t>() && std::is_same<RES, int64_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _num;
							const __m512i _mult = _mm512_set1_epi64(static_cast<int64_t>(mult));
							const __m512i _half = _mm512_set1_epi64(static_cast<int64_t>(mult / 2));

							for (; (i+8) < end; i += 8)
							{
								_num = _mm512_load_epi64(&source[i]);

								_num = _mm512_add_epi64(_num, _half);
								_num = _mm512_div_epi64(_num, _mult);
								_num = _mm512_mullox_epi64(_num, _mult);

								_mm512_store_epi64(&dest[i], _num);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _num;
							const __m256i _mult = _mm256_set1_epi64x(static_cast<int64_t>(mult));
							const __m256i _half = _mm256_set1_epi64x(static_cast<int64_t>(mult / 2));

							for (; (i+4) < end; i += 4)
							{
								_num = _mm256_load_si256((__m256i*) & source[i]);

								_num = _mm256_add_epi64(_num, _half);
								_num = _mm256_div_epi64(_num, _mult);								
								_num = tpa::util::_mm256_mul_epi64(_num, _mult);

								_mm256_store_si256((__m256i*) & dest[i], _num);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned long
					else if constexpr (std::is_same<T, uint64_t>() && std::is_same<RES, uint64_t>())
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512i _num;
						const __m512i _mult = _mm512_set1_epi64(static_cast<uint64_t>(mult));
						const __m512i _half = _mm512_set1_epi64(static_cast<uint64_t>(mult / 2));

						for (; (i+8) < end; i += 8)
						{
							_num = _mm512_load_epi64(&source[i]);

							_num = _mm512_add_epi64(_num, _half);
							_num = _mm512_div_epu64(_num, _mult);
							_num = _mm512_mullox_epi64(_num, _mult);

							_mm512_store_epi64(&dest[i], _num);
						}//End for
					}//End if hasAVX512
					else if (tpa::hasAVX2)
					{
						__m256i _num;
						const __m256i _mult = _mm256_set1_epi64x(static_cast<uint64_t>(mult));
						const __m256i _half = _mm256_set1_epi64x(static_cast<uint64_t>(mult / 2));

						for (; (i+4) < end; i += 4)
						{
							_num = _mm256_load_si256((__m256i*) & source[i]);

							_num = _mm256_add_epi64(_num, _half);
							_num = _mm256_div_epu64(_num, _mult);
							_num = tpa::util::_mm256_mul_epi64(_num, _mult);

							_mm256_store_si256((__m256i*) & dest[i], _num);
						}//End for
					}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region float
					else if constexpr (std::is_same<T, float>() && std::is_same<RES, float>())
					{
						fesetround(FE_DOWNWARD);
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512 _num;
						const __m512 _mult = _mm512_set1_ps(static_cast<int32_t>(mult));
						const __m512 _half = _mm512_set1_ps(std::rint(static_cast<float>(mult / 2)));

						for (; (i+16) < end; i += 16)
						{
							_num = _mm512_load_ps(&source[i]);
							_num = _mm512_roundscale_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm512_add_ps(_num, _half);
							_num = _mm512_div_ps(_num, _mult);
							_num = _mm512_roundscale_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm512_mul_ps(_num, _mult);

							_mm512_store_ps(&dest[i], _num);
						}//End for
					}//End if hasAVX512
					else if (tpa::hasAVX)
					{
						__m256 _num;
						const __m256 _mult = _mm256_set1_ps(static_cast<int32_t>(mult));
						const __m256 _half = _mm256_set1_ps(std::rint(static_cast<float>(mult / 2)));

						for (; (i+8) < end; i += 8)
						{
							_num = _mm256_load_ps(&source[i]);
							_num = _mm256_round_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm256_add_ps(_num, _half);
							_num = _mm256_div_ps(_num, _mult);
							_num = _mm256_round_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm256_mul_ps(_num, _mult);
														
							_mm256_store_ps(&dest[i], _num);
						}//End for
					}//End if hasAVX
					else if (tpa::has_SSE41)
					{
						__m128 _num;
						const __m128 _mult = _mm_set1_ps(static_cast<int32_t>(mult));
						const __m128 _half = _mm_set1_ps(std::rint(static_cast<float>(mult / 2)));

						for (; (i + 4) < end; i += 4)
						{
							_num = _mm_load_ps(&source[i]);
							_num = _mm_round_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm_add_ps(_num, _half);
							_num = _mm_div_ps(_num, _mult);
							_num = _mm_round_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm_mul_ps(_num, _mult);

							_mm_store_ps(&dest[i], _num);
						}//End for
					}//End if has_SSE4.1
#endif
					}//End if
#pragma endregion
#pragma region float-int32
					else if constexpr (std::is_same<T, float>() && std::is_same<RES, int32_t>())
					{
					fesetround(FE_DOWNWARD);
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512 _num;
						__m512i _res;
						const __m512 _mult = _mm512_set1_ps(static_cast<int32_t>(mult));
						const __m512 _half = _mm512_set1_ps(std::rint(static_cast<float>(mult / 2)));

						for (; (i + 16) < end; i += 16)
						{
							_num = _mm512_load_ps(&source[i]);
							_num = _mm512_roundscale_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm512_add_ps(_num, _half);
							_num = _mm512_div_ps(_num, _mult);
							_num = _mm512_roundscale_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm512_mul_ps(_num, _mult);

							_res = _mm512_cvtps_epi32(_num);

							_mm512_store_epi32(&dest[i], _res);
						}//End for
					}//End if hasAVX512
					else if (tpa::hasAVX)
					{
						__m256 _num;
						__m256i _res;
						const __m256 _mult = _mm256_set1_ps(static_cast<int32_t>(mult));
						const __m256 _half = _mm256_set1_ps(std::rint(static_cast<float>(mult / 2)));

						for (; (i + 8) < end; i += 8)
						{
							_num = _mm256_load_ps(&source[i]);
							_num = _mm256_round_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm256_add_ps(_num, _half);
							_num = _mm256_div_ps(_num, _mult);
							_num = _mm256_round_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm256_mul_ps(_num, _mult);

							_res = _mm256_cvtps_epi32(_num);

							_mm256_store_si256((__m256i*)&dest[i], _res);
						}//End for
					}//End if hasAVX
					else if (tpa::has_SSE41)
					{
						__m128 _num;
						__m128i _res;
						const __m128 _mult = _mm_set1_ps(static_cast<int32_t>(mult));
						const __m128 _half = _mm_set1_ps(std::rint(static_cast<float>(mult / 2)));

						for (; (i + 4) < end; i += 4)
						{
							_num = _mm_load_ps(&source[i]);
							_num = _mm_round_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm_add_ps(_num, _half);
							_num = _mm_div_ps(_num, _mult);
							_num = _mm_round_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm_mul_ps(_num, _mult);

							_res = _mm_cvtps_epi32(_num);

							_mm_store_si128((__m128i*)&dest[i], _res);
						}//End for
					}//End if has_SSE4.1
#endif
					}//End if
#pragma endregion
#pragma region float-uint32
					else if constexpr (std::is_same<T, float>() && std::is_same<RES, uint32_t>())
					{
					fesetround(FE_DOWNWARD);
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512 _num;
						__m512i _res;
						const __m512 _mult = _mm512_set1_ps(static_cast<int32_t>(mult));
						const __m512 _half = _mm512_set1_ps(std::rint(static_cast<float>(mult / 2)));

						for (; (i + 16) < end; i += 16)
						{
							_num = _mm512_load_ps(&source[i]);
							_num = _mm512_roundscale_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm512_add_ps(_num, _half);
							_num = _mm512_div_ps(_num, _mult);
							_num = _mm512_roundscale_ps(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm512_mul_ps(_num, _mult);

							_res = _mm512_cvtps_epu32(_num);

							_mm512_store_epi32(&dest[i], _res);
						}//End for
					}//End if hasAVX512					
#endif
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>() && std::is_same<RES, double>())
					{
					fesetround(FE_DOWNWARD);
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512d _num;
						const __m512d _mult = _mm512_set1_pd(static_cast<int64_t>(mult));
						const __m512d _half = _mm512_set1_pd(std::rint(static_cast<double>(mult / 2)));

						for (; (i + 8) < end; i += 8)
						{
							_num = _mm512_load_pd(&source[i]);
							_num = _mm512_roundscale_pd(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm512_add_pd(_num, _half);
							_num = _mm512_div_pd(_num, _mult);
							_num = _mm512_roundscale_pd(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm512_mul_pd(_num, _mult);

							_mm512_store_pd(&dest[i], _num);
						}//End for
					}//End if hasAVX512
					else if (tpa::hasAVX)
					{
						__m256d _num;
						const __m256d _mult = _mm256_set1_pd(static_cast<int64_t>(mult));
						const __m256d _half = _mm256_set1_pd(std::rint(static_cast<double>(mult / 2)));

						for (; (i + 4) < end; i += 4)
						{
							_num = _mm256_load_pd(&source[i]);
							_num = _mm256_round_pd(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm256_add_pd(_num, _half);
							_num = _mm256_div_pd(_num, _mult);
							_num = _mm256_round_pd(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm256_mul_pd(_num, _mult);

							_mm256_store_pd(&dest[i], _num);
						}//End for
					}//End if hasAVX
					else if (tpa::has_SSE41)
					{
						__m128d _num;
						const __m128d _mult = _mm_set1_pd(static_cast<int64_t>(mult));
						const __m128d _half = _mm_set1_pd(std::rint(static_cast<double>(mult / 2)));

						for (; (i + 2) < end; i += 2)
						{
							_num = _mm_load_pd(&source[i]);
							_num = _mm_round_pd(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm_add_pd(_num, _half);
							_num = _mm_div_pd(_num, _mult);
							_num = _mm_round_pd(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm_mul_pd(_num, _mult);

							_mm_store_pd(&dest[i], _num);
						}//End for
					}//End if has_SSE4.1
#endif
					}//End if
#pragma endregion
#pragma region double-int64
					else if constexpr (std::is_same<T, double>() && std::is_same<RES, int64_t>())
					{
						fesetround(FE_DOWNWARD);
#ifdef _M_AMD64
						if (tpa::hasAVX512_DWQW)
						{
							__m512d _num;
							__m512i _res;
							const __m512d _mult = _mm512_set1_pd(static_cast<int64_t>(mult));
							const __m512d _half = _mm512_set1_pd(std::rint(static_cast<double>(mult / 2)));

							for (; (i + 8) < end; i += 8)
							{
								_num = _mm512_load_pd(&source[i]);
								_num = _mm512_roundscale_pd(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

								_num = _mm512_add_pd(_num, _half);
								_num = _mm512_div_pd(_num, _mult);
								_num = _mm512_roundscale_pd(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
								_num = _mm512_mul_pd(_num, _mult);

								_res = _mm512_cvtpd_epi64(_num);

								_mm512_store_epi64(&dest[i], _res);
							}//End for
						}//End if hasAVX512	
#endif
					}//End if
#pragma endregion
#pragma region double-uint64
					else if constexpr (std::is_same<T, double>() && std::is_same<RES, uint64_t>())
					{
					fesetround(FE_DOWNWARD);
#ifdef _M_AMD64
					if (tpa::hasAVX512_DWQW)
					{
						__m512d _num;
						__m512i _res;
						const __m512d _mult = _mm512_set1_pd(static_cast<int64_t>(mult));
						const __m512d _half = _mm512_set1_pd(std::rint(static_cast<double>(mult / 2)));

						for (; (i + 8) < end; i += 8)
						{
							_num = _mm512_load_pd(&source[i]);
							_num = _mm512_roundscale_pd(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);

							_num = _mm512_add_pd(_num, _half);
							_num = _mm512_div_pd(_num, _mult);
							_num = _mm512_roundscale_pd(_num, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
							_num = _mm512_mul_pd(_num, _mult);

							_res = _mm512_cvtpd_epu64(_num);

							_mm512_store_epu64(&dest[i], _res);
						}//End for
					}//End if hasAVX512	
#endif
					}//End if
#pragma endregion
#pragma region generic						
						for (; i != end; ++i)
						{
							dest[i] = static_cast<RES>(tpa::util::round_to_nearest(source[i], mult));
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
			std::cerr << "Exception thrown in tpa::simd::round_nearest: " << ex.what() << "\n";
			std::cerr << "tpa::simd::round_nearest will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::round_nearest(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::round_nearest: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::round_nearest: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::round_nearest: unknown!\n";
		}//End catch
	}//End of round_nearest()
#pragma endregion
}//End of namespace