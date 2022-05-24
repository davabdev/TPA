#pragma once
/*
* Truly Parallel Algorithms Library - Numeric - accumulate function
* By: David Aaron Braun
* 2022-05-24
* Parallel implementation of accumulate
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include <iostream>
#include <functional>
#include <future>
#include <utility>
#include <atomic>
#include <type_traits>
#include <array>
#include <vector>

#include "../tpa.hpp"
#include "../excepts.hpp"
#include "../size_t_lit.hpp"

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa
{
#pragma region generic
    /// <summary>
    /// <para>Computes the sum of the elements in the container.</para>
    /// <para>This implementation is Multi-Threaded Only. No SIMD.</para>
    /// <para>Explicitly requires that you specify a predicate such as std::plus&lt;T&gt;()</para>
    /// <para>This implementation is about as fast as std::reduce, but more reliable.</para>
    /// </summary>
    /// <typeparam name="CONTAINER_T"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="P"></typeparam>
    /// <param name="arr"> - The container you're summing</param>
    /// <param name="pred"> - The predicate </param>
    /// <returns></returns>
    template<typename RETURN_TYPE, typename CONTAINER_T, class P>
    [[nodiscard]] inline constexpr RETURN_TYPE accumulate(
        const CONTAINER_T& arr, 
        const P pred)
        requires tpa::util::contiguous_seqeunce<CONTAINER_T>
	{
        try
        {
            uint32_t complete = 0u;

            RETURN_TYPE sum = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, arr.size());

            std::vector<std::shared_future<RETURN_TYPE>> results;
            results.reserve(tpa::nThreads);

            std::shared_future<RETURN_TYPE> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&arr, &pred, &sec]()
                {
                    const size_t beg = sec.first;
                    const size_t end = sec.second;
                    size_t i = beg;

                    RETURN_TYPE temp_val = 0;

#pragma region generic      
                    for (; i != end; ++i)
                    {
                        temp_val = pred(std::move(temp_val), arr[i]);
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
            return sum;
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): " << ex.code()
                << " " << ex.what() << "\n";
            return static_cast<RETURN_TYPE>(0);
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): " << ex.what() << "\n";
            return static_cast<RETURN_TYPE>(0);
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): " << ex.what() << "\n";
            return static_cast<RETURN_TYPE>(0);
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): unknown!\n";
            return static_cast<RETURN_TYPE>(0);
        }//End catch
	}//End of accumulate


    /// <summary>
    /// <para>Computes the sum of the elements in the container.</para>
    /// <para>This implementation uses SIMD and Multi-Threading.</para>
    /// </summary>
    /// <typeparam name="RETURN_TYPE"></typeparam>
    /// <typeparam name="CONTAINER_T"></typeparam>
    /// <param name="arr"></param>
    /// <returns></returns>
    template<typename RETURN_TYPE, typename CONTAINER_T>
    [[nodiscard]] inline constexpr RETURN_TYPE accumulate(const CONTAINER_T& arr)
    requires tpa::util::contiguous_seqeunce<CONTAINER_T>
    {
        try
        {
            using T = CONTAINER_T::value_type; 
            uint32_t complete = 0u;

            RETURN_TYPE sum = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, arr.size());

            std::vector<std::shared_future<RETURN_TYPE>> results;
            results.reserve(tpa::nThreads);

            std::shared_future<RETURN_TYPE> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&arr, &sec]()
                {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

                        RETURN_TYPE temp_val = 0;
#pragma region byte
                        if constexpr (std::is_same<T, int8_t>())
                        {
#ifdef TPA_X86_64
                            if (tpa::hasAVX512_ByteWord)
                            {
                                __m512i _sum;

                                for (; (i + 64uz) < end; i += 64uz)
                                {
                                    //Load Values
                                    _sum = _mm512_loadu_epi8(&arr[i]);
                                    
                                    //Store Result      
                                    temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm512_sum_epi8(_sum));
                                }//End for
                            }//End if
                            else if (tpa::hasAVX2)
                            {
                                __m256i _sum;

                                for (; (i + 32uz) < end; i += 32uz)
                                {         
                                    //Load Values
                                    _sum = _mm256_load_si256((__m256i*) &arr[i]);                                    

                                    //Store Result      
                                    temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm256_sum_epi8(_sum));
                                }//End for
                            }//End if hasAVX2
                            else if (tpa::has_SSE2)
                            {
                                __m128i _sum;

                                for (; (i + 16uz) < end; i += 16uz)
                                {
                                    //Load Values
                                    _sum = _mm_load_si128((__m128i*) &arr[i]);

                                    //Store Result      
                                    temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm_sum_epi8(_sum));
                                }//End for
                            }//End if
#endif
                        }//End if
#pragma endregion
#pragma region unsigned byte
                        else if constexpr (std::is_same<T, uint8_t>())
                        {
#ifdef TPA_X86_64
                            if (tpa::hasAVX512_ByteWord)
                            {
                                __m512i _sum;

                                for (; (i + 64uz) < end; i += 64uz)
                                {
                                    //Load Values
                                    _sum = _mm512_loadu_epi8(&arr[i]);

                                    //Store Result      
                                    temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm512_sum_epi8(_sum));
                                }//End for
                            }//End if
                            else if (tpa::hasAVX2)
                            {
                                __m256i _sum;

                                for (; (i + 32uz) < end; i += 32uz)
                                {
                                    //Load Values
                                    _sum = _mm256_load_si256((__m256i*) & arr[i]);

                                    //Store Result      
                                    temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm256_sum_epi8(_sum));
                                }//End for
                            }//End if hasAVX2
                            else if (tpa::has_SSE2)
                            {
                                __m128i _sum;

                                for (; (i + 16uz) < end; i += 16uz)
                                {
                                    //Load Values
                                    _sum = _mm_load_si128((__m128i*) & arr[i]);

                                    //Store Result      
                                    temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm_sum_epi8(_sum));
                                }//End for
                            }//End if
#endif
                        }//End if
#pragma endregion
#pragma region short
                        else if constexpr (std::is_same<T, int16_t>())
                        {
#ifdef TPA_X86_64
                            if (tpa::hasAVX512_ByteWord)
                            {
                                __m512i _sum;

                                for (; (i + 32uz) < end; i += 32uz)
                                {
                                    //Load Values
                                    _sum = _mm512_loadu_epi16(&arr[i]);

                                    //Store Result      
                                    temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm512_sum_epi16(_sum));
                                }//End for
                            }//End if
                            else if (tpa::hasAVX2)
                            {
                                __m256i _sum;

                                for (; (i + 16uz) < end; i += 16uz)
                                {
                                    //Load Values
                                    _sum = _mm256_load_si256((__m256i*) &arr[i]);

                                    //Store Result      
                                    temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm256_sum_epi16(_sum));
                                }//End for
                            }//End if hasAVX2
                            else if (tpa::has_SSE2)
                            {
                                __m128i _sum;

                                for (; (i + 8uz) < end; i += 8uz)
                                {
                                    //Load Values
                                    _sum = _mm_load_si128((__m128i*) & arr[i]);

                                    //Store Result      
                                    temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm_sum_epi16(_sum));
                                }//End for
                            }//End if
#endif
                        }//End if
#pragma endregion
#pragma region unsigned short
                        else if constexpr (std::is_same<T, uint16_t>())
                        {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512_ByteWord)
                        {
                            __m512i _sum;

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                //Load Values
                                _sum = _mm512_loadu_epi16(&arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm512_sum_epi16(_sum));
                            }//End for
                        }//End if
                        else if (tpa::hasAVX2)
                        {
                            __m256i _sum;

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                //Load Values
                                _sum = _mm256_load_si256((__m256i*) & arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm256_sum_epi16(_sum));
                            }//End for
                        }//End if hasAVX2
                        else if (tpa::has_SSE2)
                        {
                            __m128i _sum;

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                //Load Values
                                _sum = _mm_load_si128((__m128i*) & arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm_sum_epi16(_sum));
                            }//End for
                        }//End if
#endif
                        }//End if
#pragma endregion
#pragma region int
                        else if constexpr (std::is_same<T, int32_t>())
                        {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512)
                        {
                            __m512i _sum;

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                //Load Values
                                _sum = _mm512_load_epi32(&arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm512_sum_epi32(_sum));
                            }//End for
                        }//End if
                        else if (tpa::hasAVX2)
                        {
                            __m256i _sum;

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                //Load Values
                                _sum = _mm256_load_si256((__m256i*) & arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm256_sum_epi32(_sum));
                            }//End for
                        }//End if hasAVX2
                        else if (tpa::has_SSE2)
                        {
                            __m128i _sum;

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                //Load Values
                                _sum = _mm_load_si128((__m128i*) & arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm_sum_epi32(_sum));
                            }//End for
                        }//End if
#endif
                        }//End if
#pragma endregion
#pragma region unsigned int
                        else if constexpr (std::is_same<T, uint32_t>() == true)
                        {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512)
                        {
                            __m512i _sum;

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                //Load Values
                                _sum = _mm512_load_epi32(&arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm512_sum_epi32(_sum));
                            }//End for
                        }//End if
                        else if (tpa::hasAVX2)
                        {
                            __m256i _sum;

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                //Load Values
                                _sum = _mm256_load_si256((__m256i*) & arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm256_sum_epi32(_sum));
                            }//End for
                        }//End if hasAVX2
                        else if (tpa::has_SSE2)
                        {
                            __m128i _sum;

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                //Load Values
                                _sum = _mm_load_si128((__m128i*) & arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm_sum_epi32(_sum));
                            }//End for
                        }//End if
#endif
                        }//End if
#pragma endregion
#pragma region long
                        else if constexpr (std::is_same<T, int64_t>())
                        {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512)
                        {
                            __m512i _sum;

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                //Load Values
                                _sum = _mm512_load_epi64(&arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm512_sum_epi64(_sum));
                            }//End for
                        }//End if
                        else if (tpa::hasAVX2)
                        {
                            __m256i _sum;

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                //Load Values
                                _sum = _mm256_load_si256((__m256i*) & arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm256_sum_epi64(_sum));
                            }//End for
                        }//End if hasAVX2
                        else if (tpa::has_SSE2)
                        {
                            __m128i _sum;

                            for (; (i + 2uz) < end; i += 2uz)
                            {
                                //Load Values
                                _sum = _mm_load_si128((__m128i*) & arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm_sum_epi64(_sum));
                            }//End for
                        }//End if
#endif
                        }//End if
#pragma endregion
#pragma region unsigned long
                        else if constexpr (std::is_same<T, uint64_t>())
                        {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512)
                        {
                            __m512i _sum;

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                //Load Values
                                _sum = _mm512_load_epi64(&arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm512_sum_epi64(_sum));
                            }//End for
                        }//End if
                        else if (tpa::hasAVX2)
                        {
                            __m256i _sum;

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                //Load Values
                                _sum = _mm256_load_si256((__m256i*) & arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm256_sum_epi64(_sum));
                            }//End for
                        }//End if hasAVX2
                        else if (tpa::has_SSE2)
                        {
                            __m128i _sum;

                            for (; (i + 2uz) < end; i += 2uz)
                            {
                                //Load Values
                                _sum = _mm_load_si128((__m128i*) & arr[i]);

                                //Store Result      
                                temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm_sum_epi64(_sum));
                            }//End for
                        }//End if
#endif
                        }//End if
#pragma endregion
#pragma region float
                    else if constexpr (std::is_same<T, float>())
                    {
#ifdef TPA_X86_64
                    if (tpa::hasAVX512)
                    {
                        __m512 _sum;

                        for (; (i + 16uz) < end; i += 16uz)
                        {
                            //Load Values
                            _sum = _mm512_load_ps(&arr[i]);

                            //Store Result      
                            temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm512_sum_ps(_sum));
                        }//End for
                    }//End if
                    else if (tpa::hasAVX2)
                    {
                        __m256 _sum;

                        for (; (i + 8uz) < end; i += 8uz)
                        {
                            //Load Values
                            _sum = _mm256_load_ps(&arr[i]);

                            //Store Result      
                            temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm256_sum_ps(_sum));
                        }//End for
                    }//End if hasAVX2
                    else if (tpa::has_SSE)
                    {
                        __m128 _sum;

                        for (; (i + 4uz) < end; i += 4uz)
                        {
                            //Load Values
                            _sum = _mm_load_ps( &arr[i]);

                            //Store Result      
                            temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm_sum_ps(_sum));
                        }//End for
                    }//End if
#endif
                    }//End if
#pragma endregion
#pragma region double
                    else if constexpr (std::is_same<T, double>())
                    {
#ifdef TPA_X86_64
                    if (tpa::hasAVX512)
                    {
                        __m512d _sum;

                        for (; (i + 8uz) < end; i += 8uz)
                        {
                            //Load Values
                            _sum = _mm512_load_pd(&arr[i]);

                            //Store Result      
                            temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm512_sum_pd(_sum));
                        }//End for
                    }//End if
                    else if (tpa::hasAVX2)
                    {
                        __m256d _sum;

                        for (; (i + 4uz) < end; i += 4uz)
                        {
                            //Load Values
                            _sum = _mm256_load_pd(&arr[i]);

                            //Store Result      
                            temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm256_sum_pd(_sum));
                        }//End for
                    }//End if hasAVX2
                    else if (tpa::has_SSE2)
                    {
                        __m128d _sum;

                        for (; (i + 2uz) < end; i += 2uz)
                        {
                            //Load Values
                            _sum = _mm_load_pd(&arr[i]);

                            //Store Result      
                            temp_val += static_cast<RETURN_TYPE>(tpa::util::_mm_sum_pd(_sum));
                        }//End for
                    }//End if
#endif
                    }//End if
#pragma endregion
#pragma region generic      
                    for (; i != end; ++i)
                    {                    
                        temp_val += static_cast<RETURN_TYPE>(arr[i]);
                    }//End for                   
#pragma endregion
                        return temp_val;
                });//End of lambda

                results.emplace_back(std::move(temp));
            }//End for 

            for (const auto& fut : results)
            {
                sum += fut.get();
                complete += 1u;
            }//End for

            //Check all threads completed
            if (complete != tpa::nThreads)
            {
                throw tpa::exceptions::NotAllThreadsCompleted(complete);
            }//End if

            //Finish
            return sum;
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): " << ex.code()
                << " " << ex.what() << "\n";
            return static_cast<RETURN_TYPE>(0);
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): " << ex.what() << "\n";
            return static_cast<RETURN_TYPE>(0);
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): " << ex.what() << "\n";
            return static_cast<RETURN_TYPE>(0);
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): unknown!\n";
            return static_cast<RETURN_TYPE>(0);
        }//End catch
    }//End of accumulate
#pragma endregion
}//End of namespace
