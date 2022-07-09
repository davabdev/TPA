#pragma once
/*
*	Fused Multiply Add functions for TPA Library
*	By: David Aaron Braun
*	2021-06-27
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include <type_traits>
#include <utility>
#include <mutex>
#include <future>
#include <iostream>
#include <functional>

#include <array>
#include <vector>

#include "../_util.hpp"
#include "../ThreadPool.hpp"
#include "../excepts.hpp"
#include "../size_t_lit.hpp"
#include "../tpa_macros.hpp"
#include "../predicates.hpp"
#include "../tpa_concepts.hpp"
#include "../InstructionSet.hpp"
#include "simd.hpp"

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>SIMD Matrix Math Functions.</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa{
#pragma region generic

	/// <summary>
	/// <para>Performes Fused Multiply Add (( a * b) + z ) on 3 containers storing the results in a 4th container.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <typeparam name="C_DEST"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="arr_b"></param>
	/// <param name="arr_c"></param>
	/// <param name="dest"></param>
	template<class CONTAINER_A, class CONTAINER_B, class CONTAINER_C, class C_DEST>
	inline constexpr void fma(
		const CONTAINER_A& arr_a,
		const CONTAINER_B& arr_b,
		const CONTAINER_C& arr_c,
		C_DEST& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A> &&
		tpa::util::contiguous_seqeunce<CONTAINER_B> &&
		tpa::util::contiguous_seqeunce<CONTAINER_C> &&
		tpa::util::contiguous_seqeunce<C_DEST>
	{
		size_t smallest = 0;
		uint32_t complete = 0;

		try
		{

			static_assert(
			std::is_same<CONTAINER_A::value_type, CONTAINER_B::value_type>() &&
			std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>() &&
			std::is_same<CONTAINER_A::value_type, C_DEST::value_type>(),
			"Compile Error! All the source and destination containers must be of the same value_type!");

			using T = CONTAINER_A::value_type;

			//Determin the smallest container
			smallest = tpa::util::min(arr_a.size(), arr_b.size());
			smallest = tpa::util::min(smallest, arr_c.size());

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
				temp = tpa::tp->addTask([&arr_a, &arr_b, &arr_c, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region byte
					if constexpr (std::is_same<T, int8_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _a, _b, _c, _dest;

							for (; i < end; i += 64)
							{
								if ((i + 64) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
								_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);
								_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

								//Perform FMA (a * b) + c
								int8_t d[64] = {};
								for (size_t x = 0; x < 64; ++x)
								{
#ifdef _MSC_VER
									d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
									d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm512_loadu_epi8((__m256i*) & d);

								_dest = _mm512_add_epi8(_dest, _c);

								//Store Result
								_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _b, _c, _dest;

							for (; i < end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);
								_b = _mm256_load_si256((__m256i*) & arr_b[i]);
								_c = _mm256_load_si256((__m256i*) & arr_c[i]);

								//Perform FMA (a * b) + c	
								int8_t d[32] = {};
								for (size_t x = 0; x < 32; ++x)
								{
#ifdef _MSC_VER
									d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
									d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm256_load_si256((__m256i*) & d);

								_dest = _mm256_add_epi8(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned byte
					else if constexpr (std::is_same<T, uint8_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _a, _b, _c, _dest;

							for (; i < end; i += 64)
							{
								if ((i + 64) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
								_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);
								_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

								//Perform FMA (a * b) + c
								uint8_t d[64] = {};
								for (size_t x = 0; x < 64; ++x)
								{
#ifdef _MSC_VER
									d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
									d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm512_loadu_epi8((__m256i*) & d);

								_dest = _mm512_add_epi8(_dest, _c);

								//Store Result
								_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _b, _c, _dest;

							for (; i < end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);
								_b = _mm256_load_si256((__m256i*) & arr_b[i]);
								_c = _mm256_load_si256((__m256i*) & arr_c[i]);

								//Perform FMA (a * b) + c	
								uint8_t d[32] = {};
								for (size_t x = 0; x < 32; ++x)
								{
#ifdef _MSC_VER
									d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
									d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm256_load_si256((__m256i*) & d);

								_dest = _mm256_add_epi8(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region short
					else if constexpr (std::is_same<T, int16_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _a, _b, _c, _dest;

							for (; i < end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
								_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);
								_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_mullo_epi16(_a, _b);
								_dest = _mm512_add_epi16(_dest, _c);

								//Store Result
								_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _b, _c, _dest;

							for (; i < end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);
								_b = _mm256_load_si256((__m256i*) & arr_b[i]);
								_c = _mm256_load_si256((__m256i*) & arr_c[i]);

								//Perform FMA (a * b) + c	
								_dest = _mm256_mullo_epi16(_a, _b);
								_dest = _mm256_add_epi16(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned short
					else if constexpr (std::is_same<T, uint16_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _a, _b, _c, _dest;

							for (; i < end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
								_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);
								_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_mullo_epi16(_a, _b);
								_dest = _mm512_add_epi16(_dest, _c);

								//Store Result
								_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _b, _c, _dest;

							for (; i < end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);
								_b = _mm256_load_si256((__m256i*) & arr_b[i]);
								_c = _mm256_load_si256((__m256i*) & arr_c[i]);

								//Perform FMA (a * b) + c	
								_dest = _mm256_mullo_epi16(_a, _b);
								_dest = _mm256_add_epi16(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region int
					else if constexpr (std::is_same<T, int32_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m512i _a, _b, _c, _dest;

							for (; i < end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
								_b = _mm512_load_epi32((__m512i*)&arr_b[i]);
								_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_mullo_epi32(_a, _b);
								_dest = _mm512_add_epi32(_dest, _c);

								//Store Result
								_mm512_store_epi32((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _b, _c, _dest;

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);
								_b = _mm256_load_si256((__m256i*) & arr_b[i]);
								_c = _mm256_load_si256((__m256i*) & arr_c[i]);

								//Perform FMA (a * b) + c	
								_dest = _mm256_mullo_epi32(_a, _b);
								_dest = _mm256_add_epi32(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned int
					else if constexpr (std::is_same<T, uint32_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m512i _a, _b, _c, _dest;

							for (; i < end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
								_b = _mm512_load_epi32((__m512i*)&arr_b[i]);
								_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_mullo_epi32(_a, _b);
								_dest = _mm512_add_epi32(_dest, _c);

								//Store Result
								_mm512_store_epi32((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _b, _c, _dest;

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);
								_b = _mm256_load_si256((__m256i*) & arr_b[i]);
								_c = _mm256_load_si256((__m256i*) & arr_c[i]);

								//Perform FMA (a * b) + c	
								_dest = _mm256_mullo_epi32(_a,_b);
								_dest = _mm256_add_epi32(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region long
					else if constexpr (std::is_same<T, int64_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m512i _a, _b, _c, _dest;

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
								_b = _mm512_load_epi64((__m512i*)&arr_b[i]);
								_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

								//Perform FMA (a * b) + c								
								_dest = _mm512_mullox_epi64(_a, _b);
								_dest = _mm512_add_epi64(_dest, _c);

								//Store Result
								_mm512_store_epi64((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _b, _c, _dest;

							for (; i < end; i += 4)
							{
								if ((i + 4) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);
								_b = _mm256_load_si256((__m256i*) & arr_b[i]);
								_c = _mm256_load_si256((__m256i*) & arr_c[i]);

								//Perform FMA (a * b) + c								
								int64_t d[4] = {};
								for (size_t x = 0; x < 4; ++x)
								{
#ifdef _WIN32
									d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
									d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif

								}//End for
								_dest = _mm256_load_si256((__m256i*) & d);
								
								_dest = _mm256_add_epi64(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned long
					else if constexpr (std::is_same<T, uint64_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m512i _a, _b, _c, _dest;

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
								_b = _mm512_load_epi64((__m512i*)&arr_b[i]);
								_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_mullox_epi64(_a, _b);
								_dest = _mm512_add_epi64(_dest, _c);

								//Store Result
								_mm512_store_epi64((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _b, _c, _dest;

							for (; i < end; i += 4)
							{
								if ((i + 4) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) &arr_a[i]);
								_b = _mm256_load_si256((__m256i*) &arr_b[i]);
								_c = _mm256_load_si256((__m256i*) &arr_c[i]);

								//Perform FMA (a * b) + c
								uint64_t d[4] = {};
								for (size_t x = 0; x < 4; ++x)
								{
#ifdef _WIN32
									d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
									d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm256_load_si256((__m256i*) &d);

								_dest = _mm256_add_epi64(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) &dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region float
					else if constexpr (std::is_same<T, float>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m512 _a, _b, _c, _dest;

							for (; i < end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_ps(&arr_a[i]);
								_b = _mm512_load_ps(&arr_b[i]);
								_c = _mm512_load_ps(&arr_c[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_fmadd_ps(_a, _b, _c);

								//Store Result
								_mm512_store_ps(&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasFMA && tpa::hasAVX)
						{
							__m256 _a, _b, _c, _dest;

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_ps(&arr_a[i]);
								_b = _mm256_load_ps(&arr_b[i]);
								_c = _mm256_load_ps(&arr_c[i]);

								//Perform FMA (a * b) + c
								_dest = _mm256_fmadd_ps(_a, _b, _c);

								//Store Result
								_mm256_store_ps(&dest[i], _dest);
							}//End for
						}//End if hasFMA
#endif
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							__m512d _a, _b, _c, _dest;

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_pd(&arr_a[i]);
								_b = _mm512_load_pd(&arr_b[i]);
								_c = _mm512_load_pd(&arr_c[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_fmadd_pd(_a, _b, _c);

								//Store Result
								_mm512_store_pd(&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasFMA && tpa::hasAVX)
						{
							__m256d _a, _b, _c, _dest;

							for (; i < end; i += 4)
							{
								if ((i + 4) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_pd(&arr_a[i]);
								_b = _mm256_load_pd(&arr_b[i]);
								_c = _mm256_load_pd(&arr_c[i]);

								//Perform FMA (a * b) + c
								_dest = _mm256_fmadd_pd(_a, _b, _c);

								//Store Result
								_mm256_store_pd(&dest[i], _dest);
							}//End for
						}//End if hasFMA
#endif
					}//End if
#pragma endregion
#pragma region generic
					for (; i < end; ++i)
					{
						dest[i] = static_cast<T>((arr_a[i] * arr_b[i]) + arr_c[i]);
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
			std::cerr << "Exception thrown in tpa::fma(): " << ex.what() << "\n";
			std::cerr << "tpa::fma will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma

	/// <summary>
	/// <para>Performes Fused Multiply Add (( {CONTAINER} a * {CONTAINER} b) + {CONSTANT VALUE} z ) on 2 containers and a constant value. storing the results in a 4th container.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="T"></typeparam>
	/// <typeparam name="C_DEST"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="arr_b"></param>
	/// <param name="const_val"></param>
	/// <param name="dest"></param>
	template<class CONTAINER_A, class CONTAINER_B, class T = CONTAINER_A::value_type, class C_DEST>
	inline constexpr void fma_const_add(
		const CONTAINER_A& arr_a,
		const CONTAINER_B& arr_b,
		const T cnst_val,
		C_DEST& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A> &&
		tpa::util::contiguous_seqeunce<CONTAINER_B> &&
		tpa::util::contiguous_seqeunce<C_DEST>
	{
		size_t smallest = 0;
		uint32_t complete = 0;

		try
		{

			static_assert(
				std::is_same<CONTAINER_A::value_type, CONTAINER_B::value_type>() &&
				std::is_same<CONTAINER_A::value_type, T>() &&
				std::is_same<CONTAINER_A::value_type, C_DEST::value_type>(),
				"Compile Error! All the source and destination containers and the constant val must be of the same value_type!");

			//Determin the smallest container
			smallest = tpa::util::min(arr_a.size(), arr_b.size());

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
				temp = tpa::tp->addTask([&arr_a, &arr_b, &cnst_val, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T const_val = cnst_val;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									int8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									int8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									uint8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									uint8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi32((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi32((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi64((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c								
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi64(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c								
									int64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
										d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi64((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi64(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c
									uint64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
										d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _a, _b, _dest;
								const __m512 _c = _mm512_set1_ps(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_ps(&arr_a[i]);
									_b = _mm512_load_ps(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256 _a, _b, _dest;
								const __m256 _c = _mm256_set1_ps(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_ps(&arr_a[i]);
									_b = _mm256_load_ps(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _a, _b, _dest;
								const __m512d _c = _mm512_set1_pd(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_pd(&arr_a[i]);
									_b = _mm512_load_pd(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256d _a, _b, _dest;
								const __m512d _c = _mm512_set1_pd(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_pd(&arr_a[i]);
									_b = _mm256_load_pd(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<T>((arr_a[i] * arr_b[i]) + const_val);
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
			std::cerr << "Exception thrown in tpa::fma_const_add(): " << ex.what() << "\n";
			std::cerr << "tpa::fma_const_add will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_add(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_add: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_add: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma_const_add: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma_const_add

	/// <summary>
	/// <para>Performes Fused Multiply Add (( {CONTAINER} a * {CONSTANT_VALUE} b) + {CONTAINER} z ) on 2 containers and 1 constant value storing the results in a 4th container.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="T"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <typeparam name="C_DEST"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="const_val"></param>
	/// <param name="arr_c"></param>
	/// <param name="dest"></param>
	template<class CONTAINER_A, class T = CONTAINER_A::value_type, class CONTAINER_C, class C_DEST>
	inline constexpr void fma_const_multiply(
		const CONTAINER_A& arr_a,
		const T cnst_val,
		const CONTAINER_C& arr_c,
		C_DEST& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A> &&
		tpa::util::contiguous_seqeunce<CONTAINER_C> &&
		tpa::util::contiguous_seqeunce<C_DEST>
	{
		size_t smallest = 0;
		uint32_t complete = 0;

		try
		{

			static_assert(
				std::is_same<CONTAINER_A::value_type, T>() &&
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>() &&
				std::is_same<CONTAINER_A::value_type, C_DEST::value_type>(),
				"Compile Error! All the source and destination containers and the constant value must be of the same value_type!");

			//Determin the smallest container
			smallest = tpa::util::min(arr_a.size(), arr_c.size());

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
				temp = tpa::tp->addTask([&arr_a, &cnst_val, &arr_c, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T const_val = cnst_val;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									int8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									int8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									uint8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									uint8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c								
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi64x(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c								
									int64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
										d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi64x(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c
									uint64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
										d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _a, _c, _dest;
								const __m512 _b = _mm512_set1_ps(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_ps(&arr_a[i]);
									_c = _mm512_load_ps(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256 _a, _c, _dest;
								const __m256 _b = _mm256_set1_ps(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_ps(&arr_a[i]);
									_c = _mm256_load_ps(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _a, _c, _dest;
								const __m512 _b = _mm512_set1_pd(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_pd(&arr_a[i]);
									_c = _mm512_load_pd(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256d _a, _c, _dest;
								const __m256 _b = _mm256_set1_pd(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_pd(&arr_a[i]);
									_c = _mm256_load_pd(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<T>((arr_a[i] * const_val) + arr_c[i]);
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
			std::cerr << "Exception thrown in tpa::fma_const_multiply(): " << ex.what() << "\n";
			std::cerr << "tpa::fma will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma_const_multiply: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma_const_multiply

	/// <summary>
	/// <para>Performes Fused Multiply Add (( {CONTAINER} a * {CONSTANT_VALUE} b) + {CONSTANT_VALUE} z ) on 2 containers and 1 constant value storing the results in a 4th container.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="T"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <typeparam name="C_DEST"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="const_val"></param>
	/// <param name="const_val_c"></param>
	/// <param name="dest"></param>
	template<class CONTAINER_A, class T = CONTAINER_A::value_type, class C_DEST>
	inline constexpr void fma_const_multiply_add(
		const CONTAINER_A& arr_a,
		const T cnst_val,
		const T cnst_val_c,
		C_DEST& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A> &&
		tpa::util::contiguous_seqeunce<C_DEST>
	{
		size_t smallest = 0;
		uint32_t complete = 0;

		try
		{

			static_assert(
				std::is_same<CONTAINER_A::value_type, T>() &&
				std::is_same<CONTAINER_A::value_type, C_DEST::value_type>(),
				"Compile Error! All the source and destination containers and the constant value must be of the same value_type!");

			//Determin the smallest container
			smallest = tpa::util::min(arr_a.size(), dest.size());

		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&arr_a, &cnst_val, &cnst_val_c, &dest, &sec]()
				{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T const_val = cnst_val;
						const T const_val_c = cnst_val_c;

#pragma region byte
					if constexpr (std::is_same<T, int8_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _a, _dest;
							const __m512i _b = _mm512_set1_epi8(const_val);
							const __m512i _c = _mm512_set1_epi8(const_val_c);

							for (; i < end; i += 64)
							{
								if ((i + 64) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);

								//Perform FMA (a * b) + c
								int8_t d[64] = {};
								for (size_t x = 0; x < 64; ++x)
								{
#ifdef _WIN32
									d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
									d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm512_loadu_epi8((__m256i*) & d);

								_dest = _mm512_add_epi8(_dest, _c);

								//Store Result
								_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _dest;
							const __m256i _b = _mm256_set1_epi8(const_val);
							const __m256i _c = _mm256_set1_epi8(const_val_c);

							for (; i < end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);

								//Perform FMA (a * b) + c	
								int8_t d[32] = {};
								for (size_t x = 0; x < 32; ++x)
								{
#ifdef _WIN32
									d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
									d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm256_load_si256((__m256i*) & d);

								_dest = _mm256_add_epi8(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned byte
					else if constexpr (std::is_same<T, uint8_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _a, _dest;
							const __m512i _b = _mm512_set1_epi8(const_val);
							const __m512i _c = _mm512_set1_epi8(const_val_c);

							for (; i < end; i += 64)
							{
								if ((i + 64) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);

								//Perform FMA (a * b) + c
								uint8_t d[64] = {};
								for (size_t x = 0; x < 64; ++x)
								{
#ifdef _WIN32
									d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
									d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm512_loadu_epi8((__m256i*) & d);

								_dest = _mm512_add_epi8(_dest, _c);

								//Store Result
								_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _dest;
							const __m256i _b = _mm256_set1_epi8(const_val);
							const __m256i _c = _mm256_set1_epi8(const_val_c);

							for (; i < end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);

								//Perform FMA (a * b) + c	
								uint8_t d[32] = {};
								for (size_t x = 0; x < 32; ++x)
								{
#ifdef _WIN32
									d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
									d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm256_load_si256((__m256i*) & d);

								_dest = _mm256_add_epi8(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region short
					else if constexpr (std::is_same<T, int16_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _a, _dest;
							const __m512i _b = _mm512_set1_epi16(const_val);
							const __m512i _c = _mm512_set1_epi16(const_val_c);

							for (; i < end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_mullo_epi16(_a, _b);
								_dest = _mm512_add_epi16(_dest, _c);

								//Store Result
								_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _dest;
							const __m256i _b = _mm256_set1_epi16(const_val);
							const __m256i _c = _mm256_set1_epi16(const_val_c);

							for (; i < end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);

								//Perform FMA (a * b) + c	
								_dest = _mm256_mullo_epi16(_a, _b);
								_dest = _mm256_add_epi16(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned short
					else if constexpr (std::is_same<T, uint16_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _a, _dest;
							const __m512i _b = _mm512_set1_epi16(const_val);
							const __m512i _c = _mm512_set1_epi16(const_val_c);

							for (; i < end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_mullo_epi16(_a, _b);
								_dest = _mm512_add_epi16(_dest, _c);

								//Store Result
								_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _dest;
							const __m256i _b = _mm256_set1_epi16(const_val);
							const __m256i _c = _mm256_set1_epi16(const_val_c);

							for (; i < end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);

								//Perform FMA (a * b) + c	
								_dest = _mm256_mullo_epi16(_a, _b);
								_dest = _mm256_add_epi16(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region int
					else if constexpr (std::is_same<T, int32_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _a, _dest;
							const __m512i _b = _mm512_set1_epi32(const_val);
							const __m512i _c = _mm512_set1_epi32(const_val_c);

							for (; i < end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_epi32((__m512i*)&arr_a[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_mullo_epi32(_a, _b);
								_dest = _mm512_add_epi32(_dest, _c);

								//Store Result
								_mm512_store_epi32((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _dest;
							const __m256i _b = _mm256_set1_epi32(const_val);
							const __m256i _c = _mm256_set1_epi32(const_val_c);

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);

								//Perform FMA (a * b) + c	
								_dest = _mm256_mullo_epi32(_a, _b);
								_dest = _mm256_add_epi32(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned int
					else if constexpr (std::is_same<T, uint32_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _a, _dest;
							const __m512i _b = _mm512_set1_epi32(const_val);
							const __m512i _c = _mm512_set1_epi32(const_val_c);

							for (; i < end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_epi32((__m512i*)&arr_a[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_mullo_epi32(_a, _b);
								_dest = _mm512_add_epi32(_dest, _c);

								//Store Result
								_mm512_store_epi32((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _dest;
							const __m256i _b = _mm256_set1_epi32(const_val);
							const __m256i _c = _mm256_set1_epi32(const_val_c);

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);

								//Perform FMA (a * b) + c	
								_dest = _mm256_mullo_epi32(_a, _b);
								_dest = _mm256_add_epi32(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region long
					else if constexpr (std::is_same<T, int64_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _a, _dest;
							const __m512i _b = _mm512_set1_epi64(const_val);
							const __m512i _c = _mm512_set1_epi64(const_val_c);

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_epi64((__m512i*)&arr_a[i]);

								//Perform FMA (a * b) + c								
								_dest = _mm512_mullox_epi64(_a, _b);
								_dest = _mm512_add_epi64(_dest, _c);

								//Store Result
								_mm512_store_epi64((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasFMA && tpa::hasAVX2)
						{
							__m256i _a, _dest;
							const __m256i _b = _mm256_set1_epi64x(const_val);
							const __m256i _c = _mm256_set1_epi64x(const_val_c);

							for (; i < end; i += 4)
							{
								if ((i + 4) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);

								//Perform FMA (a * b) + c								
								int64_t d[4] = {};
								for (size_t x = 0; x < 4; ++x)
								{
#ifdef _WIN32
									d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
									d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm256_load_si256((__m256i*) & d);

								_dest = _mm256_add_epi64(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasAVX2
#endif
					}//End if
#pragma endregion
#pragma region unsigned long
					else if constexpr (std::is_same<T, uint64_t>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512i _a, _dest;
							const __m512i _b = _mm512_set1_epi64(const_val);
							const __m512i _c = _mm512_set1_epi64(const_val_c);

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_epi64((__m512i*)&arr_a[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_mullox_epi64(_a, _b);
								_dest = _mm512_add_epi64(_dest, _c);

								//Store Result
								_mm512_store_epi64((__m512i*)&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _a, _dest;
							const __m256i _b = _mm256_set1_epi64x(const_val);
							const __m256i _c = _mm256_set1_epi64x(const_val_c);

							for (; i < end; i += 4)
							{
								if ((i + 4) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_si256((__m256i*) & arr_a[i]);

								//Perform FMA (a * b) + c
								uint64_t d[4] = {};
								for (size_t x = 0; x < 4; ++x)
								{
#ifdef _WIN32
									d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
									d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
								}//End for
								_dest = _mm256_load_si256((__m256i*) & d);

								_dest = _mm256_add_epi64(_dest, _c);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _dest);
							}//End for
						}//End if hasFMA
#endif
					}//End if
#pragma endregion
#pragma region float
					else if constexpr (std::is_same<T, float>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512 _a, _dest;
							const __m512 _b = _mm512_set1_ps(const_val);
							const __m512 _c = _mm512_set1_ps(const_val_c);

							for (; i < end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_ps(&arr_a[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_fmadd_ps(_a, _b, _c);

								//Store Result
								_mm512_store_ps(&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasFMA && tpa::hasAVX)
						{
							__m256 _a, _dest;
							const __m256 _b = _mm256_set1_ps(const_val);
							const __m256 _c = _mm256_set1_ps(const_val_c);

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_ps(&arr_a[i]);

								//Perform FMA (a * b) + c
								_dest = _mm256_fmadd_ps(_a, _b, _c);

								//Store Result
								_mm256_store_ps(&dest[i], _dest);
							}//End for
						}//End if hasFMA
#endif
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							__m512d _a, _dest;
							const __m512d _b = _mm512_set1_pd(const_val);
							const __m512d _c = _mm512_set1_pd(const_val_c);

							for (; i < end; i += 8)
							{
								if ((i + 8) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm512_load_pd(&arr_a[i]);

								//Perform FMA (a * b) + c
								_dest = _mm512_fmadd_pd(_a, _b, _c);

								//Store Result
								_mm512_store_pd(&dest[i], _dest);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasFMA && tpa::hasAVX)
						{
							__m256d _a, _dest;
							const __m256d _b = _mm256_set1_pd(const_val);
							const __m256d _c = _mm256_set1_pd(const_val_c);

							for (; i < end; i += 4)
							{
								if ((i + 4) > end) [[unlikely]]
								{
									break;
								}//End if

								//Load
								_a = _mm256_load_pd(&arr_a[i]);

								//Perform FMA (a * b) + c
								_dest = _mm256_fmadd_pd(_a, _b, _c);

								//Store Result
								_mm256_store_pd(&dest[i], _dest);
							}//End for
						}//End if hasFMA
#endif
					}//End if
#pragma endregion
#pragma region generic
					for (; i < end; ++i)
					{
						dest[i] = static_cast<T>((arr_a[i] * const_val) + const_val_c);
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
			std::cerr << "Exception thrown in tpa::fma_const_multiply_add(): " << ex.what() << "\n";
			std::cerr << "tpa::fma_const_multiply_add will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in fma_const_multiply_add::fma(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply_add: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply_add: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma_const_multiply_add: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma_const_multiply_add

#pragma endregion
#pragma region array

	/// <summary>
	/// <para>Performes Fused Multiply Add (( a * b) + z ) on 3 containers storing the results in a 4th container.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <typeparam name="C_DEST"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="arr_b"></param>
	/// <param name="arr_c"></param>
	/// <param name="dest"></param>
	template<typename T, size_t SIZE_A, size_t SIZE_B, size_t SIZE_C, size_t SIZE_DEST>
	inline constexpr void fma(
		const std::array<T,SIZE_A>& arr_a,
		const std::array<T,SIZE_B>& arr_b,
		const std::array<T,SIZE_C>& arr_c,
		std::array<T,SIZE_DEST>& dest)
	{
		size_t smallest = 0;
		uint32_t complete = 0;

		try
		{

			static_assert(SIZE_A <= SIZE_DEST && SIZE_B <= SIZE_DEST && 
				SIZE_C <= SIZE_DEST,
				"Compile Error! The destination array is not large enough!");

			//Determin the smallest container
			smallest = tpa::util::min(arr_a.size(), arr_b.size());
			smallest = tpa::util::min(smallest, arr_c.size());

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&arr_a, &arr_b, &arr_c, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);
									_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									int8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									int8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);
									_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									uint8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									uint8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);
									_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);
									_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi32((__m512i*)&arr_b[i]);
									_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi32((__m512i*)&arr_b[i]);
									_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi64((__m512i*)&arr_b[i]);
									_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c								
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c								
									int64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
										d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif

									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi64((__m512i*)&arr_b[i]);
									_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c
									uint64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
										d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _a, _b, _c, _dest;

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_ps(&arr_a[i]);
									_b = _mm512_load_ps(&arr_b[i]);
									_c = _mm512_load_ps(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256 _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_ps(&arr_a[i]);
									_b = _mm256_load_ps(&arr_b[i]);
									_c = _mm256_load_ps(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_pd(&arr_a[i]);
									_b = _mm512_load_pd(&arr_b[i]);
									_c = _mm512_load_pd(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256d _a, _b, _c, _dest;

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_pd(&arr_a[i]);
									_b = _mm256_load_pd(&arr_b[i]);
									_c = _mm256_load_pd(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<T>((arr_a[i] * arr_b[i]) + arr_c[i]);
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
			std::cerr << "Exception thrown in tpa::fma(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma

	/// <summary>
	/// <para>Performes Fused Multiply Add (( {CONTAINER} a * {CONTAINER} b) + {CONSTANT VALUE} z ) on 2 arrays and a constant value. storing the results in a 4th array.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="arr_b"></param>
	/// <param name="const_val"></param>
	/// <param name="dest"></param>
	template<class T, size_t SIZE_A, size_t SIZE_B, size_t SIZE_DEST>
	inline constexpr void fma_const_add(
		const std::array<T,SIZE_A>& arr_a,
		const std::array<T,SIZE_B>& arr_b,
		const T cnst_val,
		std::array<T, SIZE_DEST>& dest)
	{
		size_t smallest = 0;
		uint32_t complete = 0;

		try
		{

			static_assert(SIZE_A <= SIZE_DEST && SIZE_B <= SIZE_DEST,
				"Compile Error! The destination array is not large enough!");

			//Determin the smallest container
			smallest = tpa::util::min(arr_a.size(), arr_b.size());

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&arr_a, &arr_b, &cnst_val, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T const_val = cnst_val;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									int8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									int8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									uint8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									uint8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi32((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi32((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi64((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c								
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi64(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c								
									int64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
										d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi64((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi64(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c
									uint64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
										d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _a, _b, _dest;
								const __m512 _c = _mm512_set1_ps(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_ps(&arr_a[i]);
									_b = _mm512_load_ps(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256 _a, _b, _dest;
								const __m256 _c = _mm256_set1_ps(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_ps(&arr_a[i]);
									_b = _mm256_load_ps(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _a, _b, _dest;
								const __m512d _c = _mm512_set1_pd(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_pd(&arr_a[i]);
									_b = _mm512_load_pd(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256d _a, _b, _dest;
								const __m512d _c = _mm512_set1_pd(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_pd(&arr_a[i]);
									_b = _mm256_load_pd(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<T>((arr_a[i] * arr_b[i]) + const_val);
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
			std::cerr << "Exception thrown in tpa::fma_const_add(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_add: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_add: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma_const_add: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma_const_add

	/// <summary>
	/// <para>Performes Fused Multiply Add (( {CONTAINER} a * {CONSTANT_VALUE} b) + {CONTAINER} z ) on 2 arrays and 1 constant value storing the results in a 4th array.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="T"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <typeparam name="C_DEST"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="const_val"></param>
	/// <param name="arr_c"></param>
	/// <param name="dest"></param>
	template<class T, size_t SIZE_A, size_t SIZE_C, size_t SIZE_DEST>
	inline constexpr void fma_const_multiply(
		const std::array<T,SIZE_A>& arr_a,
		const T cnst_val,
		const std::array<T,SIZE_C>& arr_c,
		std::array<T, SIZE_DEST>& dest)
	{
		uint32_t complete = 0;

		try
		{
			static_assert(SIZE_A <= SIZE_DEST && SIZE_C <= SIZE_DEST,
				"Compile Error! The destination array is not large enough!");

			//Determin the smallest container
			size_t smallest = tpa::util::min(arr_a.size(), arr_c.size());

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&arr_a, &cnst_val, &arr_c, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T const_val = cnst_val;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									int8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									int8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									uint8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									uint8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c								
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi64x(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c								
									int64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
										d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi64x(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c
									uint64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
										d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _a, _c, _dest;
								const __m512 _b = _mm512_set1_ps(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_ps(&arr_a[i]);
									_c = _mm512_load_ps(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256 _a, _c, _dest;
								const __m256 _b = _mm256_set1_ps(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_ps(&arr_a[i]);
									_c = _mm256_load_ps(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _a, _c, _dest;
								const __m512 _b = _mm512_set1_pd(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_pd(&arr_a[i]);
									_c = _mm512_load_pd(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256d _a, _c, _dest;
								const __m256 _b = _mm256_set1_pd(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_pd(&arr_a[i]);
									_c = _mm256_load_pd(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<T>((arr_a[i] * const_val) + arr_c[i]);
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
			std::cerr << "Exception thrown in tpa::fma_const_multiply(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma_const_multiply: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma_const_multiply

	/// <summary>
	/// <para>Performes Fused Multiply Add (( {CONTAINER} a * {CONSTANT_VALUE} b) + {CONSTANT_VALUE} z ) on 2 arrays and 1 constant value storing the results in a 4th array.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="const_val"></param>
	/// <param name="const_val_c"></param>
	/// <param name="dest"></param>
	template<class T, size_t SIZE_A, size_t SIZE_DEST>
	inline constexpr void fma_const_multiply_add(
		const std::array<T,SIZE_A>& arr_a,
		const T cnst_val,
		const T cnst_val_c,
		std::array<T,SIZE_DEST>& dest)
	{
		uint32_t complete = 0;

		try
		{
			static_assert(SIZE_A <= SIZE_DEST, "Compile Error! The destiantion array is too small!");

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, arr_a.size());

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&arr_a, &cnst_val, &cnst_val_c, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T const_val = cnst_val;
						const T const_val_c = cnst_val_c;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi8(const_val);
								const __m512i _c = _mm512_set1_epi8(const_val_c);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									int8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi8(const_val);
								const __m256i _c = _mm256_set1_epi8(const_val_c);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									int8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi8(const_val);
								const __m512i _c = _mm512_set1_epi8(const_val_c);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									uint8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi8(const_val);
								const __m256i _c = _mm256_set1_epi8(const_val_c);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									uint8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi16(const_val);
								const __m512i _c = _mm512_set1_epi16(const_val_c);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi16(const_val);
								const __m256i _c = _mm256_set1_epi16(const_val_c);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi16(const_val);
								const __m512i _c = _mm512_set1_epi16(const_val_c);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi16(const_val);
								const __m256i _c = _mm256_set1_epi16(const_val_c);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi32(const_val);
								const __m512i _c = _mm512_set1_epi32(const_val_c);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi32(const_val);
								const __m256i _c = _mm256_set1_epi32(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi32(const_val);
								const __m512i _c = _mm512_set1_epi32(const_val_c);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi32(const_val);
								const __m256i _c = _mm256_set1_epi32(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi64(const_val);
								const __m512i _c = _mm512_set1_epi64(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c								
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi64x(const_val);
								const __m256i _c = _mm256_set1_epi64x(const_val_c);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c								
									int64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
										d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi64(const_val);
								const __m512i _c = _mm512_set1_epi64(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi64x(const_val);
								const __m256i _c = _mm256_set1_epi64x(const_val_c);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c
									uint64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
										d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _a, _dest;
								const __m512 _b = _mm512_set1_ps(const_val);
								const __m512 _c = _mm512_set1_ps(const_val_c);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_ps(&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256 _a, _dest;
								const __m256 _b = _mm256_set1_ps(const_val);
								const __m256 _c = _mm256_set1_ps(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_ps(&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _a, _dest;
								const __m512d _b = _mm512_set1_pd(const_val);
								const __m512d _c = _mm512_set1_pd(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_pd(&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256d _a, _dest;
								const __m256d _b = _mm256_set1_pd(const_val);
								const __m256d _c = _mm256_set1_pd(const_val_c);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_pd(&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<T>((arr_a[i] * const_val) + const_val_c);
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
			std::cerr << "Exception thrown in fma_const_multiply_add::fma(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply_add: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply_add: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma_const_multiply_add: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma_const_multiply_add

#pragma endregion
#pragma region vector
	/// <summary>
	/// <para>Performes Fused Multiply Add (( a * b) + z ) on 3 vectors storing the results in a 4th vector.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <typeparam name="C_DEST"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="arr_b"></param>
	/// <param name="arr_c"></param>
	/// <param name="dest"></param>
	template<typename T>
	inline constexpr void fma(
		const std::vector<T>& arr_a,
		const std::vector<T>& arr_b,
		const std::vector<T>& arr_c,
		std::vector<T>& dest)
	{
		size_t smallest = 0;
		uint32_t complete = 0;

		try
		{
			//Determin the smallest container
			smallest = tpa::util::min(arr_a.size(), arr_b.size());
			smallest = tpa::util::min(smallest, arr_c.size());

			//Resize if nessary
			if (smallest > dest.size())
			{
				dest.resize(arr_a.size());
			}//End if

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&arr_a, &arr_b, &arr_c, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);
									_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									int8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									int8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);
									_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									uint8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									uint8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);
									_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);
									_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi32((__m512i*)&arr_b[i]);
									_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi32((__m512i*)&arr_b[i]);
									_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi64((__m512i*)&arr_b[i]);
									_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c								
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c								
									int64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
										d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif

									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi64((__m512i*)&arr_b[i]);
									_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _c, _dest;

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c
									uint64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
										d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _a, _b, _c, _dest;

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_ps(&arr_a[i]);
									_b = _mm512_load_ps(&arr_b[i]);
									_c = _mm512_load_ps(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256 _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_ps(&arr_a[i]);
									_b = _mm256_load_ps(&arr_b[i]);
									_c = _mm256_load_ps(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _a, _b, _c, _dest;

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_pd(&arr_a[i]);
									_b = _mm512_load_pd(&arr_b[i]);
									_c = _mm512_load_pd(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256d _a, _b, _c, _dest;

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_pd(&arr_a[i]);
									_b = _mm256_load_pd(&arr_b[i]);
									_c = _mm256_load_pd(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<T>((arr_a[i] * arr_b[i]) + arr_c[i]);
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
			std::cerr << "Exception thrown in tpa::fma(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma

	/// <summary>
	/// <para>Performes Fused Multiply Add (( {CONTAINER} a * {CONTAINER} b) + {CONSTANT VALUE} z ) on 2 vectors and a constant value. storing the results in a 4th vector.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="arr_b"></param>
	/// <param name="const_val"></param>
	/// <param name="dest"></param>
	template<class T>
	inline constexpr void fma_const_add(
		const std::vector<T>& arr_a,
		const std::vector<T>& arr_b,
		const T cnst_val,
		std::vector<T>& dest)
	{
		size_t smallest = 0;
		uint32_t complete = 0;

		try
		{
			//Determin the smallest container
			smallest = tpa::util::min(arr_a.size(), arr_b.size());

			//Resize if nessary
			if (smallest > dest.size())
			{
				dest.resize(arr_a.size());
			}//End if

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&arr_a, &arr_b, &cnst_val, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T const_val = cnst_val;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									int8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									int8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi8((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									uint8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									uint8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_b = _mm512_loadu_epi16((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi32((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi32((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi64((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c								
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi64(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c								
									int64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
										d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _b, _dest;
								const __m512i _c = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_b = _mm512_load_epi64((__m512i*)&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _b, _dest;
								const __m256i _c = _mm256_set1_epi64(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_b = _mm256_load_si256((__m256i*) & arr_b[i]);

									//Perform FMA (a * b) + c
									uint64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
										d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _a, _b, _dest;
								const __m512 _c = _mm512_set1_ps(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_ps(&arr_a[i]);
									_b = _mm512_load_ps(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256 _a, _b, _dest;
								const __m256 _c = _mm256_set1_ps(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_ps(&arr_a[i]);
									_b = _mm256_load_ps(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _a, _b, _dest;
								const __m512d _c = _mm512_set1_pd(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_pd(&arr_a[i]);
									_b = _mm512_load_pd(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256d _a, _b, _dest;
								const __m512d _c = _mm512_set1_pd(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_pd(&arr_a[i]);
									_b = _mm256_load_pd(&arr_b[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<T>((arr_a[i] * arr_b[i]) + const_val);
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
			std::cerr << "Exception thrown in tpa::fma_const_add(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_add: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_add: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma_const_add: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma_const_add

	/// <summary>
	/// <para>Performes Fused Multiply Add (( {CONTAINER} a * {CONSTANT_VALUE} b) + {CONTAINER} z ) on 2 vectors and 1 constant value storing the results in a 4th container.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="const_val"></param>
	/// <param name="arr_c"></param>
	/// <param name="dest"></param>
	template<class T>
	inline constexpr void fma_const_multiply(
		const std::vector<T>& arr_a,
		const T cnst_val,
		const std::vector<T>& arr_c,
		std::vector<T>& dest)
	{
		size_t smallest = 0;
		uint32_t complete = 0;

		try
		{
			//Determin the smallest container
			smallest = tpa::util::min(arr_a.size(), arr_c.size());

			//Resize if nessesary
			if (smallest > dest.size())
			{
				dest.resize(arr_a.size());
			}//End if

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&arr_a, &cnst_val, &arr_c, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T const_val = cnst_val;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									int8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									int8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi8(const_val);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi8((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									uint8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi8(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									uint8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi16(const_val);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);
									_c = _mm512_loadu_epi16((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi16(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi32(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi32((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi32(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c								
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi64x(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c								
									int64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
										d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _c, _dest;
								const __m512i _b = _mm512_set1_epi64(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);
									_c = _mm512_load_epi64((__m512i*)&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _c, _dest;
								const __m256i _b = _mm256_set1_epi64x(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);
									_c = _mm256_load_si256((__m256i*) & arr_c[i]);

									//Perform FMA (a * b) + c
									uint64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
										d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _a, _c, _dest;
								const __m512 _b = _mm512_set1_ps(const_val);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_ps(&arr_a[i]);
									_c = _mm512_load_ps(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256 _a, _c, _dest;
								const __m256 _b = _mm256_set1_ps(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_ps(&arr_a[i]);
									_c = _mm256_load_ps(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _a, _c, _dest;
								const __m512 _b = _mm512_set1_pd(const_val);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_pd(&arr_a[i]);
									_c = _mm512_load_pd(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256d _a, _c, _dest;
								const __m256 _b = _mm256_set1_pd(const_val);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_pd(&arr_a[i]);
									_c = _mm256_load_pd(&arr_c[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<T>((arr_a[i] * const_val) + arr_c[i]);
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
			std::cerr << "Exception thrown in tpa::fma_const_multiply(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma_const_multiply: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma_const_multiply

	/// <summary>
	/// <para>Performes Fused Multiply Add (( {CONTAINER} a * {CONSTANT_VALUE} b) + {CONSTANT_VALUE} z ) on 1 vector and 2 constant values storing the results in a 4th vector.</para>
	/// <para>This implementation uses Multi-Threading and SIMD.</para>
	/// <para>For SIMD to be used at runtime, the CPU must have the FMA Instruction Set, if not available will use scaler.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="arr_a"></param>
	/// <param name="const_val"></param>
	/// <param name="const_val_c"></param>
	/// <param name="dest"></param>
	template<class T>
	inline constexpr void fma_const_multiply_add(
		const std::vector<T>& arr_a,
		const T cnst_val,
		const T cnst_val_c,
		std::vector<T>& dest)
	{
		size_t smallest = 0;
		uint32_t complete = 0;

		try
		{
			//Resize if nessary
			if (arr_a.size() > dest.size())
			{
				dest.resize(arr_a.size());
			}//End if

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&arr_a, &cnst_val, &cnst_val_c, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

						const T const_val = cnst_val;
						const T const_val_c = cnst_val_c;

#pragma region byte
						if constexpr (std::is_same<T, int8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi8(const_val);
								const __m512i _c = _mm512_set1_epi8(const_val_c);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									int8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m512i_i8[x] * _b.m512i_i8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi8(const_val);
								const __m256i _c = _mm256_set1_epi8(const_val_c);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									int8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<int8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<int8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi8(const_val);
								const __m512i _c = _mm512_set1_epi8(const_val_c);

								for (; i < end; i += 64)
								{
									if ((i + 64) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi8((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									uint8_t d[64] = {};
									for (size_t x = 0; x < 64; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m512i_u8[x] * _b.m512i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm512_loadu_epi8((__m256i*) & d);

									_dest = _mm512_add_epi8(_dest, _c);

									//Store Result
									_mm512_storeu_epi8((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi8(const_val);
								const __m256i _c = _mm256_set1_epi8(const_val_c);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									uint8_t d[32] = {};
									for (size_t x = 0; x < 32; ++x)
									{
#ifdef _WIN32
										d[x] = static_cast<uint8_t>(_a.m256i_u8[x] * _b.m256i_u8[x]);
#else
										d[x] = static_cast<uint8_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi8(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi16(const_val);
								const __m512i _c = _mm512_set1_epi16(const_val_c);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi16(const_val);
								const __m256i _c = _mm256_set1_epi16(const_val_c);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi16(const_val);
								const __m512i _c = _mm512_set1_epi16(const_val_c);

								for (; i < end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_loadu_epi16((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi16(_a, _b);
									_dest = _mm512_add_epi16(_dest, _c);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi16(const_val);
								const __m256i _c = _mm256_set1_epi16(const_val_c);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi16(_a, _b);
									_dest = _mm256_add_epi16(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi32(const_val);
								const __m512i _c = _mm512_set1_epi32(const_val_c);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi32(const_val);
								const __m256i _c = _mm256_set1_epi32(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi32(const_val);
								const __m512i _c = _mm512_set1_epi32(const_val_c);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi32((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullo_epi32(_a, _b);
									_dest = _mm512_add_epi32(_dest, _c);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi32(const_val);
								const __m256i _c = _mm256_set1_epi32(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c	
									_dest = _mm256_mullo_epi32(_a, _b);
									_dest = _mm256_add_epi32(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi64(const_val);
								const __m512i _c = _mm512_set1_epi64(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c								
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi64x(const_val);
								const __m256i _c = _mm256_set1_epi64x(const_val_c);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c								
									int64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_i64[x] * _b.m256i_i64[x];
#else
										d[x] = static_cast<int64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasAVX2
#endif
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _a, _dest;
								const __m512i _b = _mm512_set1_epi64(const_val);
								const __m512i _c = _mm512_set1_epi64(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_epi64((__m512i*)&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_mullox_epi64(_a, _b);
									_dest = _mm512_add_epi64(_dest, _c);

									//Store Result
									_mm512_store_epi64((__m512i*)&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _a, _dest;
								const __m256i _b = _mm256_set1_epi64x(const_val);
								const __m256i _c = _mm256_set1_epi64x(const_val_c);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_si256((__m256i*) & arr_a[i]);

									//Perform FMA (a * b) + c
									uint64_t d[4] = {};
									for (size_t x = 0; x < 4; ++x)
									{
#ifdef _WIN32
										d[x] = _a.m256i_u64[x] * _b.m256i_u64[x];
#else
										d[x] = static_cast<uint64_t>(_a[x] * _b[x]);
#endif
									}//End for
									_dest = _mm256_load_si256((__m256i*) & d);

									_dest = _mm256_add_epi64(_dest, _c);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512 _a, _dest;
								const __m512 _b = _mm512_set1_ps(const_val);
								const __m512 _c = _mm512_set1_ps(const_val_c);

								for (; i < end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_ps(&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm512_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256 _a, _dest;
								const __m256 _b = _mm256_set1_ps(const_val);
								const __m256 _c = _mm256_set1_ps(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_ps(&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_ps(_a, _b, _c);

									//Store Result
									_mm256_store_ps(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>())
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512d _a, _dest;
								const __m512d _b = _mm512_set1_pd(const_val);
								const __m512d _c = _mm512_set1_pd(const_val_c);

								for (; i < end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm512_load_pd(&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm512_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm512_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasFMA && tpa::hasAVX)
							{
								__m256d _a, _dest;
								const __m256d _b = _mm256_set1_pd(const_val);
								const __m256d _c = _mm256_set1_pd(const_val_c);

								for (; i < end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Load
									_a = _mm256_load_pd(&arr_a[i]);

									//Perform FMA (a * b) + c
									_dest = _mm256_fmadd_pd(_a, _b, _c);

									//Store Result
									_mm256_store_pd(&dest[i], _dest);
								}//End for
							}//End if hasFMA
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<T>((arr_a[i] * const_val) + const_val_c);
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
			std::cerr << "Exception thrown in fma_const_multiply_add::fma(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply_add: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::fma_const_multiply_add: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "\nException thrown in tpa::fma_const_multiply_add: unknown!\n";
			std::cerr << "\nMost likely tried to use AVX-512 but it is not present on this machine.\n";
		}//End catch
	}//End of fma_const_multiply_add
#pragma endregion 
}//End of namespace