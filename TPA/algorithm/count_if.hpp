#pragma once
/*
* Truly Parallel Algorithms Library - Algorithm - count function
* By: David Aaron Braun
* 2022-02-19
* Parallel implementation of count_if
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include <type_traits>
#include <utility>
#include <iostream>
#include <future>

#include "../tpa.hpp"
#include "../ThreadPool.hpp"
#include "../_util.hpp"
#include "../size_t_lit.hpp"
#include "../tpa_macros.hpp"
#include "../predicates.hpp"
#include "../tpa_concepts.hpp"
#include "../InstructionSet.hpp"
#include "../simd/simd.hpp"

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa {

    /// <summary>
    /// <para>Count the number of items in the container 'arr' which match the constraints of the unary predicate 'PRED'</para>
    /// <para>The return type is templated and must be specifed.</para>
    /// <para>This implementation uses multi-threading only, no SIMD.</para>
    /// </summary>
    /// <typeparam name="RETURN_T"></typeparam>
    /// <typeparam name="ARR"></typeparam>
    /// <typeparam name="UNARYPREDICATE"></typeparam>
    /// <param name="arr"></param>
    /// <param name="pred"></param>
    /// <returns></returns>
    template <typename RETURN_T, typename ARR, class UNARYPREDICATE>
    inline constexpr RETURN_T count_if(const ARR& arr, const UNARYPREDICATE& pred) requires tpa::util::contiguous_seqeunce<ARR> && tpa::util::calculatable<RETURN_T>
    {
        try
        {
            using T = ARR::value_type;

            uint32_t complete = 0;
            RETURN_T count = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, arr.size());

            std::vector<std::shared_future<RETURN_T>> results;
            results.reserve(tpa::nThreads);

            std::shared_future<RETURN_T> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&arr, &pred, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

                        RETURN_T cnt = 0;

#pragma region generic

                        for (; i != end; ++i)
                        {
                            if (pred(arr[i]))
                            {
                                ++cnt;
                            }//End if
                        }//End for
#pragma endregion
                        return cnt;
                    });//End of lambda

                results.emplace_back(std::move(temp));
            }//End for

            for (const auto& fut : results)
            {
                count += fut.get();
                complete += 1u;
            }//End for

            //Check all threads completed
            if (complete != tpa::nThreads)
            {
                throw tpa::exceptions::NotAllThreadsCompleted(complete);
            }//End if

            return count;
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count_if(): " << ex.code()
                << " " << ex.what() << "\n";
            return static_cast<RETURN_T>(0);
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count_if(): " << ex.what() << "\n";
            return static_cast<RETURN_T>(0);
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count_if(): " << ex.what() << "\n";
            return static_cast<RETURN_T>(0);
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count_if(): unknown!\n";
            return static_cast<RETURN_T>(0);
        }//End catch
    }//End of count_if

    /// <summary>
    /// <para>Count the number of items in the container 'arr' which match the constraints of the unary predicate 'PRED'</para>
    /// <para>The return type is templated and must be specifed.</para>
    /// <para>This implementation uses multi-threading and SIMD where available.</para>
    /// <para>WARNING! 8-bit and 16-bit integral types have very limted 
    /// range and using SIMD to count them can produce incorrect results when you pass containers of more than 1,000,000 elements.</para>
    /// <para>If passing an array of 8-bit with > 1,000,000 or 16-bith with > 100,000,000 elements set useSIMD to false.</para>
    /// <para>Takes 1 templated predicate from tpa::cond</para>
    /// <para>tpa::cond::EQUAL_TO		            </para>	
    /// <para>tpa::cond::NOT_EQUAL_TO			    </para>
    /// <para>tpa::cond::LESS_THAN		            </para>
    /// <para>tpa::cond::LESS_THAN_OR_EQUAL_TO      </para>
    /// <para>tpa::cond::GREATER_THAN               </para>
    /// <para>tpa::cond::GREATER_THAN_OR_EQUAL_TO   </para>
    /// <para>tpa::cond::POWER_OF</para>
    /// <para>tpa::cond::DIVISIBLE_BY</para>
    /// <para>Other options not taking a parameter: </para>
    /// <para>tpa::cond::PRIME</para>
    /// <para>tpa::cond::EVEN</para>
    /// <para>tpa::cond::ODD</para>
    /// </summary>
    /// <typeparam name="RETURN_T"></typeparam>
    /// <typeparam name="ARR"></typeparam>
    /// <typeparam name="P"></typeparam>
    /// <param name="arr"></param>
    /// <param name="param"></param>
    /// <param name="useSIMD"></param>
    /// <returns></returns>
    template <tpa::cond COND, typename RETURN_T, typename ARR, typename P = uint64_t>
    inline constexpr RETURN_T count_if(const ARR& arr, const P param = 0ull, const bool useSIMD = true) requires tpa::util::contiguous_seqeunce<ARR>&& tpa::util::calculatable<RETURN_T>
    {
        try
        {
            using T = ARR::value_type;

            uint32_t complete = 0;
            RETURN_T count = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, arr.size());

            std::vector<std::shared_future<RETURN_T>> results;
            results.reserve(tpa::nThreads);

            std::shared_future<RETURN_T> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&arr, &param, &useSIMD, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

                        const P para = param;

                        RETURN_T cnt = 0;

#pragma region int
                if constexpr (std::is_same<T, int32_t>())
                {
                    //Calc
                    if constexpr (COND == tpa::cond::EVEN)
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _ZERO = _mm256_set1_epi32(0);
                            const __m256i _TWO = _mm256_set1_epi32(2);
                            __m256i _count = _mm256_setzero_si256();

                            __m256i _ARR, _REM, _MASK;

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                //Set Values
                                _ARR = _mm256_load_si256((__m256i*) & arr[i]);

                                _REM = _mm256_rem_epi32(_ARR, _TWO);

                                _MASK = _mm256_cmpeq_epi32(_REM, _ZERO);

                                _count = _mm256_sub_epi32(_count, _MASK);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));
                        }//End if hasAVX2
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _ZERO = _mm_set1_epi32(0);
                            const __m128i _TWO = _mm_set1_epi32(2);
                            __m128i _count = _mm_setzero_si128();

                            __m128i _ARR, _REM, _MASK;

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                //Set Values
                                _ARR = _mm_load_si128((__m128i*) & arr[i]);

                                _REM = _mm_rem_epi32(_ARR, _TWO);

                                _MASK = _mm_cmpeq_epi32(_REM, _ZERO);

                                _count = _mm_sub_epi32(_count, _MASK);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if hasSSE2
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::ODD)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _NEG_ONE = _mm256_set1_epi32(-1);
                            const __m256i _ZERO = _mm256_set1_epi32(0);
                            const __m256i _TWO = _mm256_set1_epi32(2);
                            
                            __m256i _count = _mm256_setzero_si256();

                            __m256i _ARR, _REM, _MASK;

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                //Set Values
                                _ARR = _mm256_load_si256((__m256i*) & arr[i]);

                                _REM = _mm256_rem_epi32(_ARR, _TWO);

                                _MASK = _mm256_cmpeq_epi32(_REM, _ZERO);
                                _MASK = _mm256_xor_si256(_MASK, _NEG_ONE);//Not Equal

                                _count = _mm256_sub_epi32(_count, _MASK);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));
                        }//End if hasAVX2
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _NEG_ONE = _mm_set1_epi32(-1);
                            const __m128i _ZERO = _mm_set1_epi32(0);
                            const __m128i _TWO = _mm_set1_epi32(2);
                            
                            __m128i _count = _mm_setzero_si128();

                            __m128i _ARR, _REM, _MASK;

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                //Set Values
                                _ARR = _mm_load_si128((__m128i*) & arr[i]);

                                _REM = _mm_rem_epi32(_ARR, _TWO);

                                _MASK = _mm_cmpeq_epi32(_REM, _ZERO);
                                _MASK = _mm_xor_si128(_MASK, _NEG_ONE);//Not Equal

                                _count = _mm_sub_epi32(_count, _MASK);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if hasSSE2
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::DIVISIBLE_BY)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _NEG_ONE = _mm256_set1_epi32(-1);
                            const __m256i _ZERO = _mm256_set1_epi32(0);
                            const __m256i _divisor = _mm256_set1_epi32(static_cast<int32_t>(para));

                            __m256i _count = _mm256_setzero_si256();
                            
                            __m256i _ARR, _REM, _MASK;

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                //Set Values
                                _ARR = _mm256_load_si256((__m256i*) & arr[i]);

                                _REM = _mm256_rem_epi32(_ARR, _divisor);

                                _MASK = _mm256_cmpeq_epi32(_REM, _ZERO);

                                _count = _mm256_sub_epi32(_count, _MASK);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));
                        }//End if hasAVX2
                        else if (tpa::has_SSE2)
                        {
                            const __m128i _NEG_ONE = _mm_set1_epi32(-1);
                            const __m128i _ZERO = _mm_set1_epi32(0);
                            const __m128i _divisor = _mm_set1_epi32(static_cast<int32_t>(para));

                            __m128i _count = _mm_setzero_si128();

                            __m128i _ARR, _REM, _MASK;

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                //Set Values
                                _ARR = _mm_load_si128((__m128i*) & arr[i]);

                                _REM = _mm_rem_epi32(_ARR, _divisor);

                                _MASK = _mm_cmpeq_epi32(_REM, _ZERO);

                                _count = _mm_sub_epi32(_count, _MASK);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if hasSSE2
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::POWER_OF)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256 _MAGIC = _mm256_set1_ps(0.000001f);
                            const __m256 _POWER_D = _mm256_set1_ps(static_cast<float>(para));

                            const __m256 LOG_OF_POWER = _mm256_log_ps(_POWER_D);

                            __m256i _count = _mm256_setzero_si256();
                            __m256i _N_INT = _mm256_setzero_si256();
                            __m256 _N_DBL = _mm256_setzero_ps();
                            __m256 LOG_OF_N = _mm256_setzero_ps();
                            __m256 DIVIDE_LOG = _mm256_setzero_ps();
                            __m256 TRUNCATED = _mm256_setzero_ps();

                            __m256 CMP_MASK = _mm256_setzero_ps();
                                                      
                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                //Set Values
                                _N_INT = _mm256_load_si256((__m256i*) &arr[i]);
                                _N_DBL = _mm256_cvtepi32_ps(_N_INT);
                                
                                LOG_OF_N = _mm256_log_ps(_N_DBL);

                                DIVIDE_LOG = _mm256_div_ps(LOG_OF_N, LOG_OF_POWER);

                                TRUNCATED = _mm256_sub_ps(DIVIDE_LOG, _mm256_trunc_ps(DIVIDE_LOG));

                                CMP_MASK = _mm256_cmp_ps(TRUNCATED, _MAGIC, _CMP_LT_OQ);

                                _count = _mm256_sub_epi32(_count, _mm256_castps_si256(CMP_MASK));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));
                        }//End if hasAVX2
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128 _MAGIC = _mm_set1_ps(0.000001f);
                            const __m128 _POWER_D = _mm_set1_ps(static_cast<float>(para));

                            const __m128 LOG_OF_POWER = _mm_log_ps(_POWER_D);

                            __m128i _count = _mm_setzero_si128();
                            __m128i _N_INT = _mm_setzero_si128();
                            __m128 _N_DBL = _mm_setzero_ps();
                            __m128 LOG_OF_N = _mm_setzero_ps();
                            __m128 DIVIDE_LOG = _mm_setzero_ps();
                            __m128 TRUNCATED = _mm_setzero_ps();

                            __m128 CMP_MASK = _mm_setzero_ps();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                //Set Values
                                _N_INT = _mm_load_si128((__m128i*) & arr[i]);
                                _N_DBL = _mm_cvtepi32_ps(_N_INT);

                                LOG_OF_N = _mm_log_ps(_N_DBL);

                                DIVIDE_LOG = _mm_div_ps(LOG_OF_N, LOG_OF_POWER);

                                TRUNCATED = _mm_sub_ps(DIVIDE_LOG, _mm_trunc_ps(DIVIDE_LOG));

                                CMP_MASK = _mm_cmp_ps(TRUNCATED, _MAGIC, _CMP_LT_OQ);

                                _count = _mm_sub_epi32(_count, _mm_castps_si128(CMP_MASK));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if hasSSE2
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::EQUAL_TO)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi32(static_cast<int32_t>(para));
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_cmpeq_epi32(_arr, _val);

                                _count = _mm256_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi32(static_cast<int32_t>(para));
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_cmpeq_epi32(_arr, _val);

                                _count = _mm_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::NOT_EQUAL_TO)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi32(static_cast<int32_t>(para));
                            const __m256i _NEG_ONE = _mm256_set1_epi32(-1);
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_cmpeq_epi32(_arr, _val);
                                _mask = _mm256_xor_si256(_mask, _NEG_ONE);//Not Equal

                                _count = _mm256_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi32(static_cast<int32_t>(para));
                            const __m128i _NEG_ONE = _mm_set1_epi32(-1);
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_cmpeq_epi32(_arr, _val);
                                _mask = _mm_xor_si128(_mask, _NEG_ONE);//Not Equal

                                _count = _mm_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::GREATER_THAN)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi32(static_cast<int32_t>(para));
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_cmpgt_epi32(_arr, _val);

                                _count = _mm256_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi32(static_cast<int32_t>(para));
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_cmpgt_epi32(_arr, _val);

                                _count = _mm_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if      
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::GREATER_THAN_OR_EQUAL_TO)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi32(static_cast<int32_t>(para));
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_or_si256(_mm256_cmpeq_epi32(_arr, _val), _mm256_cmpgt_epi32(_arr, _val));

                                _count = _mm256_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi32(static_cast<int32_t>(para));
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_or_si128(_mm_cmpeq_epi32(_arr, _val), _mm_cmpgt_epi32(_arr, _val));

                                _count = _mm_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if 
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::LESS_THAN)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi32(static_cast<int32_t>(para));
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_cmpgt_epi32(_val, _arr);

                                _count = _mm256_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi32(static_cast<int32_t>(para));
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_cmpgt_epi32(_val, _arr);

                                _count = _mm_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if  
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::LESS_THAN_OR_EQUAL_TO)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi32(static_cast<int32_t>(para));
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_or_si256(_mm256_cmpeq_epi32(_arr, _val), _mm256_cmpgt_epi32(_val, _arr));

                                _count = _mm256_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi32(static_cast<int32_t>(para));
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_or_si128(_mm_cmpeq_epi32(_arr, _val), _mm_cmpgt_epi32(_val, _arr));

                                _count = _mm_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::PERFECT)
                    {
                        
                    }//End if
                    else if constexpr (COND == tpa::cond::PERFECT_SQUARE)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            __m256i _ARR = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _sqrt = _mm256_setzero_si256();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _ARR = _mm256_load_si256((__m256i*) &arr[i]);

                                _sqrt = _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(_ARR)));
                                _sqrt = _mm256_mullo_epi32(_sqrt, _sqrt);

                                _mask = _mm256_cmpeq_epi32(_sqrt, _ARR);

                                _count = _mm256_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));
                        }//End if
                        else if (tpa::has_SSE41)
                        {
                            __m128i _ARR = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _sqrt = _mm_setzero_si128();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _ARR = _mm_load_si128((__m128i*) & arr[i]);

                                _sqrt = _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(_ARR)));
                                _sqrt = _mm_mullo_epi32(_sqrt, _sqrt);

                                _mask = _mm_cmpeq_epi32(_sqrt, _ARR);

                                _count = _mm_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::FIBONACCI)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {                            
                            const __m256i _four = _mm256_set1_epi32(4);
                            const __m256i _five = _mm256_set1_epi32(5);

                            __m256i _ARR = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();

                            __m256i _mult = _mm256_setzero_si256();
                            __m256i _x_add = _mm256_setzero_si256();
                            __m256i _x_minus = _mm256_setzero_si256();
                            __m256i _sqrt_a = _mm256_setzero_si256();
                            __m256i _sqrt_b = _mm256_setzero_si256();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _ARR = _mm256_load_si256((__m256i*) &arr[i]);

                                _mult = _mm256_mullo_epi32(_five, _ARR);
                                _mult = _mm256_mullo_epi32(_mult, _ARR);

                                _x_add = _mm256_add_epi32(_mult, _four);
                                _x_minus = _mm256_sub_epi32(_mult, _four);

                                _sqrt_a = _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(_x_add)));
                                _sqrt_a = _mm256_mullo_epi32(_sqrt_a, _sqrt_a);

                                _sqrt_b = _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(_x_minus)));
                                _sqrt_b = _mm256_mullo_epi32(_sqrt_b, _sqrt_b);

                                _mask = _mm256_or_si256(_mm256_cmpeq_epi32(_sqrt_a, _x_add), _mm256_cmpeq_epi32(_sqrt_b, _x_minus));

                                _count = _mm256_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));
                        }//End if
                        else if (tpa::has_SSE41)
                        {
                            const __m128i _four = _mm_set1_epi32(4);
                            const __m128i _five = _mm_set1_epi32(5);

                            __m128i _ARR = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();

                            __m128i _mult = _mm_setzero_si128();
                            __m128i _x_add = _mm_setzero_si128();
                            __m128i _x_minus = _mm_setzero_si128();
                            __m128i _sqrt_a = _mm_setzero_si128();
                            __m128i _sqrt_b = _mm_setzero_si128();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _ARR = _mm_load_si128((__m128i*) & arr[i]);

                                _mult = _mm_mullo_epi32(_five, _ARR);
                                _mult = _mm_mullo_epi32(_mult, _ARR);

                                _x_add = _mm_add_epi32(_mult, _four);
                                _x_minus = _mm_sub_epi32(_mult, _four);

                                _sqrt_a = _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(_x_add)));
                                _sqrt_a = _mm_mullo_epi32(_sqrt_a, _sqrt_a);

                                _sqrt_b = _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(_x_minus)));
                                _sqrt_b = _mm_mullo_epi32(_sqrt_b, _sqrt_b);

                                _mask = _mm_or_si128(_mm_cmpeq_epi32(_sqrt_a, _x_add), _mm_cmpeq_epi32(_sqrt_b, _x_minus));

                                _count = _mm_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if
#endif
                    }//End if
                    else if constexpr (COND == tpa::cond::TRIBONOCCI)
                    {
                        
                    }//End if
                    else if constexpr (COND == tpa::cond::PRIME)
                    {
                                  
                    }//End if
                    else if constexpr (COND == tpa::cond::SYLVESTER)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            const __m256i _syl0 = _mm256_set1_epi64x(2ull);
                            const __m256i _syl1 = _mm256_set1_epi64x(3ull);
                            const __m256i _syl2 = _mm256_set1_epi64x(7ull);
                            const __m256i _syl3 = _mm256_set1_epi64x(43ull);
                            const __m256i _syl4 = _mm256_set1_epi64x(1807ull);
                            const __m256i _syl5 = _mm256_set1_epi64x(3263443ull);
                            const __m256i _syl6 = _mm256_set1_epi64x(10650056950807ull);

                            __m256i _ARR = _mm256_setzero_si256();                          
                            __m256i _mask0 = _mm256_setzero_si256();
                            __m256i _mask1 = _mm256_setzero_si256();
                            __m256i _mask2 = _mm256_setzero_si256();
                            __m256i _mask3 = _mm256_setzero_si256();
                            __m256i _mask4 = _mm256_setzero_si256();
                            __m256i _mask5 = _mm256_setzero_si256();
                            __m256i _mask6 = _mm256_setzero_si256();
                            __m256i _mask_comb = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();
                           
                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _ARR = _mm256_cvtepi32_epi64(_mm_load_si128((__m128i*) &arr[i]));
                                
                                _mask0 = _mm256_cmpeq_epi64(_ARR, _syl0);
                                _mask1 = _mm256_cmpeq_epi64(_ARR, _syl1);
                                _mask2 = _mm256_cmpeq_epi64(_ARR, _syl2);
                                _mask3 = _mm256_cmpeq_epi64(_ARR, _syl3);
                                _mask4 = _mm256_cmpeq_epi64(_ARR, _syl4);
                                _mask5 = _mm256_cmpeq_epi64(_ARR, _syl5);
                                _mask6 = _mm256_cmpeq_epi64(_ARR, _syl6);
                               
                                _mask_comb = _mm256_or_si256(_mask0, _mask1);
                                _mask_comb = _mm256_or_si256(_mask_comb, _mask2);
                                _mask_comb = _mm256_or_si256(_mask_comb, _mask3);
                                _mask_comb = _mm256_or_si256(_mask_comb, _mask4);
                                _mask_comb = _mm256_or_si256(_mask_comb, _mask5);
                                _mask_comb = _mm256_or_si256(_mask_comb, _mask6);
                                
                                _count = _mm256_sub_epi64(_count, _mask_comb);                                
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi64(_count));
                            
                        }//End if
                        else if (tpa::has_SSE41)
                        {
                            //Not Implemented.
                        }//End if
#endif
                    }//End if
                    else
                    {
                        [] <bool flag = false>()
                        {
                            static_assert(flag, " You have specified an invalid predicate function in tpa::count_if<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
                        }();
                    }//End else
                }//End if
#pragma endregion
#pragma region generic

                        for (; i != end; ++i)
                        {
                            //Calc
                            if constexpr (COND == tpa::cond::EVEN)
                            {
                                if (tpa::util::isEven(arr[i]))
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::ODD)
                            {
                                if (tpa::util::isOdd(arr[i]))
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::DIVISIBLE_BY)
                            {
                                if constexpr (std::is_floating_point<T>())
                                {
                                    if (std::fmod(arr[i], param) == 0.0)
                                    {
                                        ++cnt;
                                    }//End if
                                }//End if
                                else
                                {
                                    if (arr[i] % param == 0)
                                    {
                                        ++cnt;
                                    }//End if
                                }//End else
                            }//End if
                            else if constexpr (COND == tpa::cond::POWER_OF)
                            {
                                if (tpa::util::isPower(arr[i], param))
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::EQUAL_TO)
                            {
                                if (arr[i] == param)
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::NOT_EQUAL_TO)
                            {
                                if (arr[i] != param)
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::GREATER_THAN)
                            {
                                if (arr[i] > param)
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::GREATER_THAN_OR_EQUAL_TO)
                            {
                                if (arr[i] >= param)
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::LESS_THAN)
                            {
                                if (arr[i] < param)
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::LESS_THAN_OR_EQUAL_TO)
                            {
                                if (arr[i] <= param)
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::PRIME)
                            {
                                if (tpa::util::isPrime(arr[i]))
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::PERFECT)
                            {
                                
                            }//End if                            
                            else if constexpr (COND == tpa::cond::PERFECT_SQUARE)
                            {
                                if (tpa::util::isPerfectSquare(arr[i]))
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::FIBONACCI)
                            {
                                if (tpa::util::isFibonacci(arr[i]))
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::SYLVESTER)
                            {
                                if (tpa::util::isSylvester(arr[i]))
                                {
                                    ++cnt;
                                }//End if
                            }//End if
                            else if constexpr (COND == tpa::cond::TRIBONOCCI)
                            {
                                [] <bool flag = false>()
                                {
                                    static_assert(flag, "tpa::find_if<TRIBONOCCI>() Is not yet implemented.");
                                }();
                            }//End if
                            else
                            {
                                [] <bool flag = false>()
                                {
                                    static_assert(flag, " You have specified an invalid predicate function in tpa::count_if<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
                                }();
                            }//End else
                        }//End for
#pragma endregion
                        return cnt;
                    });//End of lambda

                results.emplace_back(std::move(temp));
            }//End for

            for (const auto& fut : results)
            {
                count += fut.get();
                complete += 1u;
            }//End for

            //Check all threads completed
            if (complete != tpa::nThreads)
            {
                throw tpa::exceptions::NotAllThreadsCompleted(complete);
            }//End if

            return count;
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count_if(): " << ex.code()
                << " " << ex.what() << "\n";
            return static_cast<RETURN_T>(0);
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count_if(): " << ex.what() << "\n";
            return static_cast<RETURN_T>(0);
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count_if(): " << ex.what() << "\n";
            return static_cast<RETURN_T>(0);
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count_if(): unknown!\n";
            return static_cast<RETURN_T>(0);
        }//End catch
    }//End of count_if
}//End namespace