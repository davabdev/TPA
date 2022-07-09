#pragma once
/*
*	Exponent functions for TPA Library
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

#include "../_util.hpp"
#include "../ThreadPool.hpp"
#include "../excepts.hpp"
#include "../size_t_lit.hpp"
#include "../tpa_macros.hpp"
#include "../predicates.hpp"
#include "../tpa_concepts.hpp"
#include "../InstructionSet.hpp"
#include "simd.hpp"

#undef min
#undef max
#undef abs
#undef floor
#undef ceil
#undef round
#undef exp
#undef exp2
#undef exp10
#undef expm1

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>SIMD Matrix Math Functions.</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa{
#pragma region generic

	/// <summary>
	/// <para>Computes e (Euler's number, 2.7182818...) raised to the powers in 'source' and stores the results in 'dest'</para>
	/// <para>This implementation uses Multi-Threading and SIMD (where available).</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	/// <param name="suppress_exceptions"></param>
	template<typename CONTAINER_A, typename CONTAINER_B>
	inline constexpr void exp(
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
#ifdef TPA_X86_64
				if (tpa::hasAVX512)
				{
					__m512 _num;

					for (; (i + 16) < end; i += 16)
					{
						_num = _mm512_load_ps(&source[i]);

						_num = _mm512_exp_ps(_num);

						_mm512_store_ps(&dest[i], _num);
					}//End for
				}//End if hasAVX512
				else if (tpa::hasAVX)
				{
					__m256 _num;

					for (; (i + 8) < end; i += 8)
					{
						_num = _mm256_load_ps(&source[i]);

						_num = _mm256_exp_ps(_num);

						_mm256_store_ps(&dest[i], _num);
					}//End for
				}//End if hasAVX
				else if (tpa::has_SSE)
				{
					__m128 _num;

					for (; (i + 4) < end; i += 4)
					{
						_num = _mm_load_ps(&source[i]);

						_num = _mm_exp_ps(_num);

						_mm_store_ps(&dest[i], _num);
					}//End for
				}//End if has_SSE
#endif
			}//End if
#pragma endregion
#pragma region double
			if constexpr (std::is_same<T, double>() && std::is_same<RES, double>())
			{
#ifdef TPA_X86_64
				if (tpa::hasAVX512)
				{
					__m512d _num;

					for (; (i + 8) < end; i += 8)
					{
						_num = _mm512_load_pd(&source[i]);

						_num = _mm512_exp_pd(_num);

						_mm512_store_pd(&dest[i], _num);
					}//End for
				}//End if hasAVX512
				else if (tpa::hasAVX)
				{
					__m256d _num;

					for (; (i + 4) < end; i += 4)
					{
						_num = _mm256_load_pd(&source[i]);

						_num = _mm256_exp_pd(_num);

						_mm256_store_pd(&dest[i], _num);
					}//End for
				}//End if hasAVX
				else if (tpa::has_SSE2)
				{
					__m128d _num;

					for (; (i + 2) < end; i += 2)
					{
						_num = _mm_load_pd(&source[i]);

						_num = _mm_exp_pd(_num);

						_mm_store_pd(&dest[i], _num);
					}//End for
				}//End if has_SSE
#endif
			}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<RES>(tpa::util::exp(source[i]));
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
			std::cerr << "Exception thrown in tpa::simd::exp: " << ex.what() << "\n";
			std::cerr << "tpa::simd::exp will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp: unknown!\n";
		}//End catch
	}//End of exp()

	/// <summary>
	/// <para>Computes 2 raised to the powers in 'source' and stores the results in 'dest'</para>
	/// <para>This implementation uses Multi-Threading and SIMD (where available).</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	/// <param name="suppress_exceptions"></param>
	template<typename CONTAINER_A, typename CONTAINER_B>
	inline constexpr void exp2(
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
#ifdef TPA_X86_64
							if (tpa::hasAVX512)
							{
								__m512 _num;

								for (; (i + 16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_num = _mm512_exp2_ps(_num);

									_mm512_store_ps(&dest[i], _num);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _num;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm256_load_ps(&source[i]);

									_num = _mm256_exp2_ps(_num);

									_mm256_store_ps(&dest[i], _num);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE)
							{
								__m128 _num;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm_load_ps(&source[i]);

									_num = _mm_exp2_ps(_num);

									_mm_store_ps(&dest[i], _num);
								}//End for
							}//End if has_SSE
#endif
						}//End if
#pragma endregion
#pragma region double
						if constexpr (std::is_same<T, double>() && std::is_same<RES, double>())
						{
#ifdef TPA_X86_64
							if (tpa::hasAVX512)
							{
								__m512d _num;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_num = _mm512_exp2_pd(_num);

									_mm512_store_pd(&dest[i], _num);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _num;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm256_load_pd(&source[i]);

									_num = _mm256_exp2_pd(_num);

									_mm256_store_pd(&dest[i], _num);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE2)
							{
								__m128d _num;

								for (; (i + 2) < end; i += 2)
								{
									_num = _mm_load_pd(&source[i]);

									_num = _mm_exp2_pd(_num);

									_mm_store_pd(&dest[i], _num);
								}//End for
							}//End if has_SSE
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<RES>(tpa::util::exp2(source[i]));
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
			std::cerr << "Exception thrown in tpa::simd::exp2: " << ex.what() << "\n";
			std::cerr << "tpa::simd::exp2 will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp2(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp2: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp2: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp2: unknown!\n";
		}//End catch
	}//End of exp2()

	/// <summary>
	/// <para>Computes 10 raised to the powers in 'source' and stores the results in 'dest'</para>
	/// <para>This implementation uses Multi-Threading and SIMD (where available).</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	/// <param name="suppress_exceptions"></param>
	template<typename CONTAINER_A, typename CONTAINER_B>
	inline constexpr void exp10(
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
#ifdef TPA_X86_64
							if (tpa::hasAVX512)
							{
								__m512 _num;

								for (; (i + 16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_num = _mm512_exp10_ps(_num);

									_mm512_store_ps(&dest[i], _num);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _num;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm256_load_ps(&source[i]);

									_num = _mm256_exp10_ps(_num);

									_mm256_store_ps(&dest[i], _num);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE)
							{
								__m128 _num;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm_load_ps(&source[i]);

									_num = _mm_exp10_ps(_num);

									_mm_store_ps(&dest[i], _num);
								}//End for
							}//End if has_SSE
#endif
						}//End if
#pragma endregion
#pragma region double
						if constexpr (std::is_same<T, double>() && std::is_same<RES, double>())
						{
#ifdef TPA_X86_64
							if (tpa::hasAVX512)
							{
								__m512d _num;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_num = _mm512_exp10_pd(_num);

									_mm512_store_pd(&dest[i], _num);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _num;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm256_load_pd(&source[i]);

									_num = _mm256_exp10_pd(_num);

									_mm256_store_pd(&dest[i], _num);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE2)
							{
								__m128d _num;

								for (; (i + 2) < end; i += 2)
								{
									_num = _mm_load_pd(&source[i]);

									_num = _mm_exp10_pd(_num);

									_mm_store_pd(&dest[i], _num);
								}//End for
							}//End if has_SSE
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<RES>(tpa::util::exp10(source[i]));
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
			std::cerr << "Exception thrown in tpa::simd::exp10: " << ex.what() << "\n";
			std::cerr << "tpa::simd::exp10 will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp10(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp10: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp10: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::exp10: unknown!\n";
		}//End catch
	}//End of exp10()

	/// <summary>
	/// <para>Computes e (Euler's number, 2.7182818...) raised to the powers in 'source' minus 1 and stores the results in 'dest'</para>
	/// <para>This implementation uses Multi-Threading and SIMD (where available).</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <param name="source"></param>
	/// <param name="dest"></param>
	/// <param name="suppress_exceptions"></param>
	template<typename CONTAINER_A, typename CONTAINER_B>
	inline constexpr void expm1(
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
#ifdef TPA_X86_64
							if (tpa::hasAVX512)
							{
								__m512 _num;

								for (; (i + 16) < end; i += 16)
								{
									_num = _mm512_load_ps(&source[i]);

									_num = _mm512_expm1_ps(_num);

									_mm512_store_ps(&dest[i], _num);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _num;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm256_load_ps(&source[i]);

									_num = _mm256_expm1_ps(_num);

									_mm256_store_ps(&dest[i], _num);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE)
							{
								__m128 _num;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm_load_ps(&source[i]);

									_num = _mm_expm1_ps(_num);

									_mm_store_ps(&dest[i], _num);
								}//End for
							}//End if has_SSE
#endif
						}//End if
#pragma endregion
#pragma region double
						if constexpr (std::is_same<T, double>() && std::is_same<RES, double>())
						{
#ifdef TPA_X86_64
							if (tpa::hasAVX512)
							{
								__m512d _num;

								for (; (i + 8) < end; i += 8)
								{
									_num = _mm512_load_pd(&source[i]);

									_num = _mm512_expm1_pd(_num);

									_mm512_store_pd(&dest[i], _num);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _num;

								for (; (i + 4) < end; i += 4)
								{
									_num = _mm256_load_pd(&source[i]);

									_num = _mm256_expm1_pd(_num);

									_mm256_store_pd(&dest[i], _num);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE2)
							{
								__m128d _num;

								for (; (i + 2) < end; i += 2)
								{
									_num = _mm_load_pd(&source[i]);

									_num = _mm_expm1_pd(_num);

									_mm_store_pd(&dest[i], _num);
								}//End for
							}//End if has_SSE
#endif
						}//End if
#pragma endregion
#pragma region generic
						for (; i < end; ++i)
						{
							dest[i] = static_cast<RES>(tpa::util::expm1(source[i]));
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
			std::cerr << "Exception thrown in tpa::simd::expm1: " << ex.what() << "\n";
			std::cerr << "tpa::simd::expm1 will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::expm1(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::expm1: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::expm1: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::expm1: unknown!\n";
		}//End catch
	}//End of expm1()
#pragma endregion
}//End of namespace