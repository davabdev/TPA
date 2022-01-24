#pragma once
/*
* Truely Parallel Algorithms Library - Statistic Functions
* By: David Aaron Braun
* 2022-01-23
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

#include "../ThreadPool.hpp"
#include "../excepts.hpp"
#include "../size_t_lit.hpp"
#include "../_util.hpp"

#ifdef _M_AMD64
#include <immintrin.h>
#elif defined (_M_ARM64)
#ifdef _WIN32
#include "arm64_neon.h"
#else
#include "arm_neon.h"
#endif
#endif


/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa {

	/// <summary>
	/// <para>TPA Library: Statistical Functions</para>
	/// </summary>
	namespace stat {

		/// <summary>
		/// <para>Compute and return the average of the passed parameters.</para>
		/// <para>Note that if you pass only integers, integer division will be used.
		/// To get a floating-point result, pass at least one (1) floating-point type</para>
		/// </summary>
		/// <param name="a"></param>
		/// <param name="...args"></param>
		/// <returns></returns>
		template<tpa::util::calculatable T, tpa::util::calculatable...Ts>
		[[nodiscard]] inline constexpr auto mean(T a, Ts... args) ->  decltype(a + (args + ...))
		{
			return (a + (args + ...)) / (sizeof...(args) + 1);
		}//End of mean


        /// <summary>
        /// <para>Calculates and returns the average of the values in 'arr'</para>
        /// <para>The return type is the value type of the container.</para>
        /// <para>Uses Multi-Threading and SIMD (where available)</para>
        /// </summary>
        /// <typeparam name="CONTAINER_T"></typeparam>
        /// <param name="arr"></param>
        /// <returns></returns>
        template<typename CONTAINER_T>
        [[nodiscard]] inline constexpr CONTAINER_T::value_type mean(const CONTAINER_T& arr) requires tpa::util::contiguous_seqeunce<CONTAINER_T>
        {
            using T = CONTAINER_T::value_type;

            try
            {
                uint32_t complete = 0;

                T sum = 0;

                std::vector<std::pair<size_t, size_t>> sections;
                tpa::util::prepareThreading(sections, arr.size());

                std::vector<std::shared_future<T>> results;
                results.reserve(tpa::nThreads);

                std::shared_future<T> temp;

                for (const auto& sec : sections)
                {
                    temp = tpa::tp->addTask([&arr, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

                        T temp_val = 0;

#pragma region float
                        if constexpr (std::is_same<T, float>())
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX512)
                            {
                                __m512 _a{}, _b{}, _c{}, _d{}, _sum{};

                                for (; (i + 64uz) < end; i += 64uz)
                                {
                                    _a = _mm512_load_ps(&arr[i]);
                                    _b = _mm512_load_ps(&arr[i+16uz]);
                                    _c = _mm512_load_ps(&arr[i+32uz]);
                                    _d = _mm512_load_ps(&arr[i+48uz]);

                                    _sum = _mm512_add_ps(_a, _b);
                                    _sum = _mm512_add_ps(_sum, _c);
                                    _sum = _mm512_add_ps(_sum, _d);

                                    for (size_t x = 0uz; x < 16uz; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m512_f32[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for

                            }//End if hasAVX512
                            else if (tpa::hasAVX)
                            {
                                __m256 _a{}, _b{}, _c{}, _d{}, _sum{};

                                for (; (i + 32uz) < end; i += 32uz)
                                {
                                    _a = _mm256_load_ps(&arr[i]);
                                    _b = _mm256_load_ps(&arr[i+8uz]);
                                    _c = _mm256_load_ps(&arr[i+16uz]);
                                    _d = _mm256_load_ps(&arr[i+24uz]);

                                    _sum = _mm256_add_ps(_a, _b);
                                    _sum = _mm256_add_ps(_sum, _c);
                                    _sum = _mm256_add_ps(_sum, _d);

                                    for (size_t x = 0uz; x != 8uz; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256_f32[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for

                            }//End if hasAVX
                            else if (tpa::has_SSE)
                            {
                                __m128 _a{}, _b{}, _c{}, _d{}, _sum{};

                                for (; (i + 16uz) < end; i += 16uz)
                                {
                                    _a = _mm_load_ps(&arr[i]);
                                    _b = _mm_load_ps(&arr[i+4uz]);
                                    _c = _mm_load_ps(&arr[i+8uz]);
                                    _d = _mm_load_ps(&arr[i+12uz]);

                                    _sum = _mm_add_ps(_a, _b);
                                    _sum = _mm_add_ps(_sum, _c);
                                    _sum = _mm_add_ps(_sum, _d);

                                    for (size_t x = 0uz; x < 4uz; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m128_f32[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for

                            }//End if has_SSE
#endif
                        }//End if
#pragma endregion
#pragma region generic      
                            for (; i != end; ++i)
                            {
                                temp_val += arr[i];
                            }//End for
#pragma endregion

                            return temp_val;
                        });//End of lambda

                    results.emplace_back(std::move(temp));
                }//End for 

                for (const auto& fut : results)
                {
                    sum += fut.get();
                    complete += 1;
                }//End for

                //Check all threads completed
                if (complete != tpa::nThreads)
                {
                    throw tpa::exceptions::NotAllThreadsCompleted(complete);
                }//End if

                //Finish
                return sum / static_cast<T>(arr.size());

            }//End try
            catch (const std::future_error& ex)
            {
                std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
                std::cerr << "Exception thrown in tpa::stat::mean(): " << ex.code()
                    << " " << ex.what() << "\n";
                return static_cast<T>(0);
            }//End catch
            catch (const std::bad_alloc& ex)
            {
                std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
                std::cerr << "Exception thrown in tpa::stat::mean(): " << ex.what() << "\n";
                return static_cast<T>(0);
            }//End catch
            catch (const std::exception& ex)
            {
                std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
                std::cerr << "Exception thrown in tpa::stat::mean(): " << ex.what() << "\n";
                return static_cast<T>(0);
            }//End catch
            catch (...)
            {
                std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
                std::cerr << "Exception thrown in tpa::stat::mean(): unknown!\n";
                return static_cast<T>(0);
            }//End catch
        }//End of mean
	}//End of namespace
}//End of namespace