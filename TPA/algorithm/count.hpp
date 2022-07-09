#pragma once
/*
* Truly Parallel Algorithms Library - Algorithm - count function
* By: David Aaron Braun
* 2022-02-10
* Parallel implementation of count
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
    /// <para>Count the occurances of 'value' in the container 'arr'</para>
    /// <para>The return type must be specifed as a template argument.</para>
    /// <para>The type of 'value' must be identical to the value_type of 'arr'</para>
    /// <para>Uses multi-threading and SIMD where available</para>
    /// <para>WARNING! 8-bit and 16-bit integral types have very limted 
    /// range and using SIMD to count them can produce incorrect results when you pass containers of more than 1,000,000 elements.</para>
    /// <para>If passing an array of 8-bit with > 1,000,000 or 16-bith with > 100,000,000 elements set useSIMD to false.</para>
    /// </summary>
    /// <typeparam name="RETURN_T"></typeparam>
    /// <typeparam name="ARR"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <param name="arr"></param>
    /// <param name="value"></param>
    /// <param name="useSIMD"> Set me to false to fix incorrect results from large 8-bit and 16-bit int containers</param>
    /// <returns></returns>
    template <typename RETURN_T, typename ARR, typename T>
    inline RETURN_T count(const ARR& arr, T value, const bool useSIMD = true) requires tpa::util::contiguous_seqeunce<ARR> && tpa::util::calculatable<T> && tpa::util::calculatable<RETURN_T>
    {
        try
        {
            static_assert(std::is_same<T,ARR::value_type>(), "Error in tpa::count: The type of 'value' must be identical to the value_type of the container 'arr'");

            uint32_t complete = 0u;
            RETURN_T count = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, arr.size());

            std::vector<std::shared_future<RETURN_T>> results;
            results.reserve(nThreads);

            std::shared_future<RETURN_T> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&arr, &value, &useSIMD, &sec]()
                {
                    const size_t beg = sec.first;
                    const size_t end = sec.second;
                    size_t i = beg;

                    const T val = value;
                    RETURN_T cnt = 0;

#pragma region byte
                    if constexpr (std::is_same<T, int8_t>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512_ByteWord && useSIMD)
                        {
                            const __m512i _val = _mm512_set1_epi8(val);
                            __m512i _arr = _mm512_setzero_si512();
                            __mmask64 _mask;
                            __m512i _count = _mm512_setzero_si512();

                            for (; (i + 64uz) < end; i += 64uz)
                            {
                                _arr = _mm512_loadu_epi8(&arr[i]);

                                _mask = _mm512_cmpeq_epi8_mask(_arr, _val);

                                _count = _mm512_sub_epi8(_count, _mm512_set1_epi8(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm512_sum_epi8(_count));
                        }//End if
                        else if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi8(val);
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) &arr[i]);

                                _mask = _mm256_cmpeq_epi8(_arr, _val);

                                _count = _mm256_sub_epi8(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi8(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi8(val);
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_cmpeq_epi8(_arr, _val);

                                _count = _mm_sub_epi8(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi8(_count));
                        }//End if
#endif
                    }//End if
#pragma endregion
#pragma region unsigned byte
                    if constexpr (std::is_same<T, uint8_t>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512_ByteWord && useSIMD)
                        {
                            const __m512i _val = _mm512_set1_epi8(val);
                            __m512i _arr = _mm512_setzero_si512();
                            __mmask64 _mask;
                            __m512i _count = _mm512_setzero_si512();

                            for (; (i + 64uz) < end; i += 64uz)
                            {
                                _arr = _mm512_loadu_epi8(&arr[i]);

                                _mask = _mm512_cmpeq_epi8_mask(_arr, _val);

                                _count = _mm512_sub_epi8(_count, _mm512_set1_epi8(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm512_sum_epi8(_count));
                        }//End if
                        else if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi8(val);
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_cmpeq_epi8(_arr, _val);

                                _count = _mm256_sub_epi8(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi8(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi8(val);
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_cmpeq_epi8(_arr, _val);

                                _count = _mm_sub_epi8(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi8(_count));
                        }//End if
#endif
                    }//End if
#pragma endregion
#pragma region short
                    if constexpr (std::is_same<T, int16_t>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512_ByteWord && useSIMD)
                        {
                            const __m512i _val = _mm512_set1_epi16(val);
                            __m512i _arr = _mm512_setzero_si512();
                            __mmask32 _mask;
                            __m512i _count = _mm512_setzero_si512();

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                _arr = _mm512_loadu_epi16(&arr[i]);

                                _mask = _mm512_cmpeq_epi16_mask(_arr, _val);

                                _count = _mm512_sub_epi16(_count, _mm512_set1_epi16(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm512_sum_epi16(_count));
                        }//End if
                        else if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi16(val);
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_cmpeq_epi16(_arr, _val);

                                _count = _mm256_sub_epi16(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi16(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi16(val);
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_cmpeq_epi16(_arr, _val);

                                _count = _mm_sub_epi16(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi16(_count));
                        }//End if
#endif
                    }//End if
#pragma endregion
#pragma region unsigned short
                    if constexpr (std::is_same<T, uint16_t>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512_ByteWord && useSIMD)
                        {
                            const __m512i _val = _mm512_set1_epi16(val);
                            __m512i _arr = _mm512_setzero_si512();
                            __mmask32 _mask;
                            __m512i _count = _mm512_setzero_si512();

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                _arr = _mm512_loadu_epi16(&arr[i]);

                                _mask = _mm512_cmpeq_epi16_mask(_arr, _val);

                                _count = _mm512_sub_epi16(_count, _mm512_set1_epi16(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm512_sum_epi16(_count));
                        }//End if
                        else if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi16(val);
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_cmpeq_epi16(_arr, _val);

                                _count = _mm256_sub_epi16(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi16(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi16(val);
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_cmpeq_epi16(_arr, _val);

                                _count = _mm_sub_epi16(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi16(_count));
                        }//End if
#endif
                    }//End if
#pragma endregion
#pragma region int                    
                    if constexpr (std::is_same<T, int32_t>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512 && useSIMD)
                        {
                            const __m512i _val = _mm512_set1_epi32(val);
                            __m512i _arr = _mm512_setzero_si512();
                            __mmask16 _mask;
                            __m512i _count = _mm512_setzero_epi32();

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _arr = _mm512_load_epi32(&arr[i]);

                                _mask = _mm512_cmpeq_epi32_mask(_arr, _val);

                                _count = _mm512_sub_epi32(_count, _mm512_set1_epi32(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm512_sum_epi32(_count));
                        }//End if
                        else if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi32(val);
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) &arr[i]);

                                _mask = _mm256_cmpeq_epi32(_arr, _val);

                                _count = _mm256_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi32(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi32(val);
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm_load_si128((__m128i*) &arr[i]);

                                _mask = _mm_cmpeq_epi32(_arr, _val);

                                _count = _mm_sub_epi32(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi32(_count));
                        }//End if
#endif
                    }//End if
#pragma endregion
#pragma region unsigned int
                    if constexpr (std::is_same<T, uint32_t>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512 && useSIMD)
                        {
                            const __m512i _val = _mm512_set1_epi32(val);
                            __m512i _arr = _mm512_setzero_si512();
                            __mmask16 _mask;
                            __m512i _count = _mm512_setzero_epi32();

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _arr = _mm512_load_epi32(&arr[i]);

                                _mask = _mm512_cmpeq_epi32_mask(_arr, _val);

                                _count = _mm512_sub_epi32(_count, _mm512_set1_epi32(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm512_sum_epi32(_count));
                        }//End if
                        else if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi32(val);
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
                            const __m128i _val = _mm_set1_epi32(val);
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
#pragma endregion
#pragma region long
                    if constexpr (std::is_same<T, int64_t>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512 && useSIMD)
                        {
                            const __m512i _val = _mm512_set1_epi64(val);
                            __m512i _arr = _mm512_setzero_si512();
                            __mmask8 _mask;
                            __m512i _count = _mm512_setzero_si512();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm512_load_epi64(&arr[i]);

                                _mask = _mm512_cmpeq_epi64_mask(_arr, _val);

                                _count = _mm512_sub_epi64(_count, _mm512_set1_epi64(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm512_sum_epi64(_count));
                        }//End if
                        else if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi64x(val);
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_cmpeq_epi64(_arr, _val);

                                _count = _mm256_sub_epi64(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi64(_count));

                        }//End if
                        else if (tpa::has_SSE41 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi64x(val);
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 2uz) < end; i += 2uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_cmpeq_epi64(_arr, _val);

                                _count = _mm_sub_epi64(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi64(_count));
                        }//End if
#endif
                    }//End if
#pragma endregion
#pragma region unsigned long
                    if constexpr (std::is_same<T, uint64_t>())
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX512 && useSIMD)
                        {
                            const __m512i _val = _mm512_set1_epi64(val);
                            __m512i _arr = _mm512_setzero_si512();
                            __mmask8 _mask;
                            __m512i _count = _mm512_setzero_si512();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm512_load_epi64(&arr[i]);

                                _mask = _mm512_cmpeq_epi64_mask(_arr, _val);

                                _count = _mm512_sub_epi64(_count, _mm512_set1_epi64(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm512_sum_epi64(_count));
                        }//End if
                        else if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256i _val = _mm256_set1_epi64x(val);
                            __m256i _arr = _mm256_setzero_si256();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256i _count = _mm256_setzero_si256();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm256_load_si256((__m256i*) & arr[i]);

                                _mask = _mm256_cmpeq_epi64(_arr, _val);

                                _count = _mm256_sub_epi64(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_epi64(_count));

                        }//End if
                        else if (tpa::has_SSE41 && useSIMD)
                        {
                            const __m128i _val = _mm_set1_epi64x(val);
                            __m128i _arr = _mm_setzero_si128();
                            __m128i _mask = _mm_setzero_si128();
                            __m128i _count = _mm_setzero_si128();

                            for (; (i + 2uz) < end; i += 2uz)
                            {
                                _arr = _mm_load_si128((__m128i*) & arr[i]);

                                _mask = _mm_cmpeq_epi64(_arr, _val);

                                _count = _mm_sub_epi64(_count, _mask);
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_epi64(_count));
                        }//End if
#endif
                    }//End if
#pragma endregion
#pragma region float
                    if constexpr (std::is_same<T, float>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512 && useSIMD)
                        {
                            const __m512 _val = _mm512_set1_ps(val);
                            __m512 _arr = _mm512_setzero_ps();
                            __mmask16 _mask;
                            __m512 _count = _mm512_setzero_ps();

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _arr = _mm512_load_ps(&arr[i]);

                                _mask = _mm512_cmpeq_ps_mask(_arr, _val);

                                _count = _mm512_sub_ps(_count, _mm512_set1_ps(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm512_sum_ps(_count));
                        }//End if
                        else if (tpa::hasAVX2 && useSIMD)
                        {
                            const __m256 _val = _mm256_set1_ps(val);
                            __m256 _arr = _mm256_setzero_ps();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256 _count = _mm256_setzero_ps();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm256_load_ps(&arr[i]);

                                _mask = _mm256_castps_si256(_mm256_cmp_ps(_arr, _val, _CMP_EQ_OQ));

                                _count = _mm256_sub_ps(_count, _mm256_cvtepi32_ps(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_ps(_count));

                        }//End if
                        else if (tpa::has_SSE && useSIMD)
                        {
                            const __m128 _val = _mm_set1_ps(val);
                            __m128 _arr = _mm_setzero_ps();
                            __m128i _mask = _mm_setzero_si128();
                            __m128 _count = _mm_setzero_ps();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm_load_ps(&arr[i]);

                                _mask = _mm_castps_si128(_mm_cmpeq_ps(_arr, _val));

                                _count = _mm_sub_ps(_count, _mm_cvtepi32_ps(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_ps(_count));
                        }//End if
#endif
                    }//End if
#pragma endregion
#pragma region double
                    if constexpr (std::is_same<T, double>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512 && useSIMD)
                        {
                            const __m512d _val = _mm512_set1_pd(val);
                            __m512d _arr = _mm512_setzero_pd();
                            __mmask8 _mask;
                            __m512d _count = _mm512_setzero_pd();

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _arr = _mm512_load_pd(&arr[i]);

                                _mask = _mm512_cmpeq_pd_mask(_arr, _val);

                                _count = _mm512_sub_pd(_count, _mm512_set1_pd(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm512_sum_pd(_count));
                        }//End if
                        else if (tpa::hasAVX && useSIMD)
                        {
                            const __m256d _val = _mm256_set1_pd(val);
                            __m256d _arr = _mm256_setzero_pd();
                            __m256i _mask = _mm256_setzero_si256();
                            __m256d _count = _mm256_setzero_pd();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm256_load_pd(&arr[i]);

                                _mask = _mm256_castpd_si256(_mm256_cmp_pd(_arr, _val, _CMP_EQ_OQ));

                                _count = _mm256_sub_pd(_count, tpa::simd::_mm256_cvtepi64_pd(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm256_sum_pd(_count));

                        }//End if
                        else if (tpa::has_SSE2 && useSIMD)
                        {
                            const __m128d _val = _mm_set1_pd(val);
                            __m128d _arr = _mm_setzero_pd();
                            __m128i _mask = _mm_setzero_si128();
                            __m128d _count = _mm_setzero_pd();

                            for (; (i + 4uz) < end; i += 4uz)
                            {
                                _arr = _mm_load_pd(&arr[i]);

                                _mask = _mm_castpd_si128(_mm_cmpeq_pd(_arr, _val));

                                _count = _mm_sub_pd(_count, tpa::simd::_mm_cvtepi64_pd(_mask));
                            }//End for

                            cnt = static_cast<RETURN_T>(tpa::simd::_mm_sum_pd(_count));
                        }//End if
#endif
                    }//End if
#pragma endregion
#pragma region generic
                    for (; i != end; ++i)
                    {
                        if (arr[i] == val)
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
            if (complete != nThreads)
            {
                throw tpa::exceptions::NotAllThreadsCompleted(complete);
            }//End if

            return count;
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count(): " << ex.code()
                << " " << ex.what() << "\n";
            return static_cast<RETURN_T>(0);
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count(): " << ex.what() << "\n";
            return static_cast<RETURN_T>(0);
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count(): " << ex.what() << "\n";
            return static_cast<RETURN_T>(0);
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::count(): unknown!\n";
            return static_cast<RETURN_T>(0);
        }//End catch
    }//End of count
}//End of namespace