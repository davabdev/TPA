#pragma once
/*
*	Matrix SIMD functions for TPA Library
*	By: David Aaron Braun
*	2021-08-14
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

#undef sqrt
#undef cbrt

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>SIMD Matrix Math Functions.</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa::simd {
#pragma region generic

	/// <summary>
	/// <para>Computes root functions on containers of numbers</para>
	/// <para>Uses Multi-Threading and SIMD where available.</para>
	/// <para>Containers of different types are allowed but not reccommended.</para>
	/// <para>Containers may be of different value_types but it is not reccommended as this prevends SIMD optimizations.</para>
	/// <para>Containers do not have to be a particular size values are computed upto the max size of the destination container.</para>
	/// <para>Use templated predicates from tpa::rt</para>
	/// <para>tpa::rt::SQUARE</para>
	/// <para>tpa::rt::INVERSE_SQUARE</para>
	/// <para>tpa::rt::CUBE</para>
	/// </summary>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="NUM"></typeparam>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	/// <param name="n"></param>
	template<tpa::rt INSTR, typename CONTAINER_A, typename CONTAINER_B, typename NUM = uint32_t>
	inline constexpr void root(
		const CONTAINER_A& source,
		CONTAINER_B& dest,
		const NUM n = 0)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			using T = CONTAINER_A::value_type;
			using RES = CONTAINER_B::value_type;

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
				temp = tpa::tp->addTask([&source, &dest, &n, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;						
						size_t i = beg;

						const NUM nroot = n;
#pragma region float
					if constexpr (std::is_same<T, float>() && std::is_same<RES, float>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							const __m512 _one = _mm512_set1_ps(1.0f);
							__m512 _DESTi;

							for (; i+16 < end; i += 16)
							{
								//Set Values
								_DESTi = _mm512_load_ps(&source[i]);

								//Calc
								if constexpr (INSTR == tpa::rt::SQUARE)
								{
									_DESTi = _mm512_sqrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_SQUARE)
								{
									_DESTi = _mm512_invsqrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::CUBE)
								{
									_DESTi = _mm512_cbrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_CUBE)
								{
#ifdef _WIN32
									_DESTi = _mm512_invcbrt_ps(_DESTi);
#elif
									_DESTi = _mm512_div_ps(_one, _mm512_cbrt_ps(_DESTi));
#endif
								}//End if
								else if constexpr (INSTR == tpa::rt::N_ROOT)
								{
									break; //Use scaler
								}//End if		
								else if constexpr (INSTR == tpa::rt::INVERSE_N_ROOT)
								{
									break; //Use scaler
								}//End if								
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::root<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm512_store_ps(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX) [[likely]]
						{
							__m256 _DESTi;

							for (; i+8 < end; i += 8)
							{		
								//Set Values
								_DESTi = _mm256_load_ps(&source[i]);

								//Calc
								if constexpr (INSTR == tpa::rt::SQUARE)
								{
									_DESTi = _mm256_sqrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_SQUARE)
								{
									_DESTi = _mm256_invsqrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::CUBE)
								{
									_DESTi = _mm256_cbrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_CUBE)
								{
									_DESTi = _mm256_invcbrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::N_ROOT)
								{
									break; //Use scaler
								}//End if		
								else if constexpr (INSTR == tpa::rt::INVERSE_N_ROOT)
								{
									break; //Use scaler
								}//End if								
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::root<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm256_store_ps(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX
						else if (tpa::has_SSE) [[likely]]
						{
							__m128 _DESTi;

							for (; (i + 4) < end; i += 4)
							{
								//Set Values
								_DESTi = _mm_load_ps(&source[i]);

								//Calc
								if constexpr (INSTR == tpa::rt::SQUARE)
								{
									_DESTi = _mm_sqrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_SQUARE)
								{
									_DESTi = _mm_invsqrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::CUBE)
								{
									_DESTi = _mm_cbrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_CUBE)
								{
									_DESTi = _mm_invcbrt_ps(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::N_ROOT)
								{
									break; //Use scaler
								}//End if		
								else if constexpr (INSTR == tpa::rt::INVERSE_N_ROOT)
								{
									break; //Use scaler
								}//End if								
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::root<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm_store_ps(&dest[i], _DESTi);
							}//End for
						}//End if hasSSE
#endif						
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>() && std::is_same<RES, double>())
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512)
						{
							const __m512d _one = _mm512_set1_pd(1.0);
							__m512d _DESTi;

							for (; i+8 < end; i += 8)
							{
								//Set Values
								_DESTi = _mm512_load_pd(&source[i]);

								//Calc
								if constexpr (INSTR == tpa::rt::SQUARE)
								{
									_DESTi = _mm512_sqrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_SQUARE)
								{
									_DESTi = _mm512_invsqrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::CUBE)
								{
									_DESTi = _mm512_cbrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_CUBE)
								{
#ifdef _WIN32
									_DESTi = _mm512_invcbrt_pd(_DESTi);								
#elif
									_DESTi = _mm512_div_pd(_one, _mm512_cbrt_pd(_DESTi));
#endif
								}//End if
								else if constexpr (INSTR == tpa::rt::N_ROOT)
								{
									break; //Use scaler
								}//End if		
								else if constexpr (INSTR == tpa::rt::INVERSE_N_ROOT)
								{
									break; //Use scaler
								}//End if								
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::root<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
									}();
								}//End else

								//Store Result
								_mm512_store_pd(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX) [[likely]]
						{
							__m256d _DESTi;

							for (; i+4 < end; i += 4)
							{
								//Set Values
								_DESTi = _mm256_load_pd(&source[i]);

								//Calc
								if constexpr (INSTR == tpa::rt::SQUARE)
								{
									_DESTi = _mm256_sqrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_SQUARE)
								{
									_DESTi = _mm256_invsqrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::CUBE)
								{
									_DESTi = _mm256_cbrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_CUBE)
								{
									_DESTi = _mm256_invcbrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::N_ROOT)
								{
									break; //Use scaler
								}//End if		
								else if constexpr (INSTR == tpa::rt::INVERSE_N_ROOT)
								{
									break; //Use scaler
								}//End if								
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::root<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm256_store_pd(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX
						else if (tpa::has_SSE2) [[likely]]
						{
							__m128d _DESTi;

							for (; (i + 2) < end; i += 2)
							{
								//Set Values
								_DESTi = _mm_load_pd(&source[i]);

								//Calc
								if constexpr (INSTR == tpa::rt::SQUARE)
								{
									_DESTi = _mm_sqrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_SQUARE)
								{
									_DESTi = _mm_invsqrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::CUBE)
								{
									_DESTi = _mm_cbrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::INVERSE_CUBE)
								{
									_DESTi = _mm_invcbrt_pd(_DESTi);
								}//End if
								else if constexpr (INSTR == tpa::rt::N_ROOT)
								{
									break; //Use scaler
								}//End if		
								else if constexpr (INSTR == tpa::rt::INVERSE_N_ROOT)
								{
									break; //Use scaler
								}//End if								
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::root<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
									}();
								}//End else

								//Store Result
								_mm_store_pd(&dest[i], _DESTi);
							}//End for
						}//End if hasSSE2
#endif						
					}//End if
#pragma endregion
#pragma region generic						
						for (; i != end; ++i)
						{
							if constexpr (INSTR == tpa::rt::SQUARE)
							{
								dest[i] = static_cast<RES>(tpa::util::sqrt(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::rt::INVERSE_SQUARE)
							{
								dest[i] = static_cast<RES>(tpa::util::isqrt(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::rt::CUBE)
							{
								dest[i] = static_cast<RES>(tpa::util::cbrt(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::rt::INVERSE_CUBE)
							{
								dest[i] = static_cast<RES>(tpa::util::icbrt(source[i]));
							}//End if
							else if constexpr (INSTR == tpa::rt::N_ROOT)
							{
								dest[i] = static_cast<RES>(tpa::util::n_root(source[i], nroot));
							}//End if
							else if constexpr (INSTR == tpa::rt::INVERSE_N_ROOT)
							{
								dest[i] = static_cast<RES>(tpa::util::n_iroot(source[i], nroot));
							}//End if
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::root<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
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
			std::cerr << "Exception thrown in tpa::simd::root: " << ex.what() << "\n";
			std::cerr << "tpa::simd::root will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::root(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::root: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::root: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::root: unknown!\n";
		}//End catch
	}//End of root()
#pragma endregion
}