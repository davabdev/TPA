#pragma once
/*
*	Truly Parallel Algorithms Library - Algorithm - min_element function
*	By: David Aaron Braun
*	2021-06-30
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
#include "../simd/simd.hpp"

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa {

#pragma region generic

	/// <summary>
	/// <para>Returns a copy of the smallest element in the container.</para>
    /// <para>This parallel implementation uses Multi-Threading and SIMD.</para>
    /// <para>The return type is the value_type of the container.</para>
    /// <para>If passing an container containing no elements, will throw an exception and return 0.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_T"></typeparam>
	/// <typeparam name="T"></typeparam>
	/// <param name="arr"></param>
	/// <returns></returns>
	template<class CONTAINER_T, typename T = CONTAINER_T::value_type>
	[[nodiscard]] inline constexpr T min_element(const CONTAINER_T& arr)
    requires tpa::util::contiguous_seqeunce<CONTAINER_T>
	{
        try
        {
            uint32_t complete = 0;

            //Guard against zero-element arrays
            if (arr.size() == 0)
            {
                throw tpa::exceptions::EmptyArray();
            }//End if

            T min = arr[0];

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

                        T temp_min = arr[beg];
#pragma region byte
                    if constexpr (std::is_same<T, int8_t>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512_ByteWord)
                        {
                            __m512i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 256)
                            {
                                if ((i + 256) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm512_loadu_epi8((__m512i*)&arr[i]);
                                _second = _mm512_loadu_epi8((__m512i*)&arr[i + 64]);
                                _third = _mm512_loadu_epi8((__m512i*)&arr[i + 128]);
                                _forth = _mm512_loadu_epi8((__m512i*)&arr[i + 192]);

                                //Compute min
                                _min = _mm512_min_epi8(_first, _second);
                                _min = _mm512_min_epi8(_min, _third);
                                _min = _mm512_min_epi8(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 64; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m512i_i8[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 128)
                            {
                                if ((i + 128) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm256_load_si256((__m256i*) & arr[i]);
                                _second = _mm256_load_si256((__m256i*) & arr[i + 32]);
                                _third = _mm256_load_si256((__m256i*) & arr[i + 64]);
                                _forth = _mm256_load_si256((__m256i*) & arr[i + 96]);

                                //Compute min
                                _min = _mm256_min_epi8(_first, _second);
                                _min = _mm256_min_epi8(_min, _third);
                                _min = _mm256_min_epi8(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 32; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m256i_i8[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
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
                            __m512i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 256)
                            {
                                if ((i + 256) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm512_loadu_epi8((__m512i*)&arr[i]);
                                _second = _mm512_loadu_epi8((__m512i*)&arr[i + 64]);
                                _third = _mm512_loadu_epi8((__m512i*)&arr[i + 128]);
                                _forth = _mm512_loadu_epi8((__m512i*)&arr[i + 192]);

                                //Compute min
                                _min = _mm512_min_epu8(_first, _second);
                                _min = _mm512_min_epu8(_min, _third);
                                _min = _mm512_min_epu8(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 64; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m512i_u8[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 128)
                            {
                                if ((i + 128) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm256_load_si256((__m256i*) & arr[i]);
                                _second = _mm256_load_si256((__m256i*) & arr[i + 32]);
                                _third = _mm256_load_si256((__m256i*) & arr[i + 64]);
                                _forth = _mm256_load_si256((__m256i*) & arr[i + 96]);

                                //Compute min
                                _min = _mm256_min_epu8(_first, _second);
                                _min = _mm256_min_epu8(_min, _third);
                                _min = _mm256_min_epu8(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 32; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m256i_u8[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
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
                            __m512i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 128)
                            {
                                if ((i + 128) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm512_loadu_epi16((__m512i*)&arr[i]);
                                _second = _mm512_loadu_epi16((__m512i*)&arr[i + 32]);
                                _third = _mm512_loadu_epi16((__m512i*)&arr[i + 64]);
                                _forth = _mm512_loadu_epi16((__m512i*)&arr[i + 96]);

                                //Compute min
                                _min = _mm512_min_epi16(_first, _second);
                                _min = _mm512_min_epi16(_min, _third);
                                _min = _mm512_min_epi16(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 32; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m512i_i16[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 64)
                            {
                                if ((i + 64) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm256_load_si256((__m256i*) & arr[i]);
                                _second = _mm256_load_si256((__m256i*) & arr[i + 16]);
                                _third = _mm256_load_si256((__m256i*) & arr[i + 32]);
                                _forth = _mm256_load_si256((__m256i*) & arr[i + 48]);

                                //Compute min
                                _min = _mm256_min_epi16(_first, _second);
                                _min = _mm256_min_epi16(_min, _third);
                                _min = _mm256_min_epi16(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 16; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m256i_i16[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
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
                            __m512i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 128)
                            {
                                if ((i + 128) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm512_loadu_epi16((__m512i*)&arr[i]);
                                _second = _mm512_loadu_epi16((__m512i*)&arr[i + 32]);
                                _third = _mm512_loadu_epi16((__m512i*)&arr[i + 64]);
                                _forth = _mm512_loadu_epi16((__m512i*)&arr[i + 96]);

                                //Compute min
                                _min = _mm512_min_epu16(_first, _second);
                                _min = _mm512_min_epu16(_min, _third);
                                _min = _mm512_min_epu16(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 32; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m512i_u16[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 64)
                            {
                                if ((i + 64) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm256_load_si256((__m256i*) & arr[i]);
                                _second = _mm256_load_si256((__m256i*) & arr[i + 16]);
                                _third = _mm256_load_si256((__m256i*) & arr[i + 32]);
                                _forth = _mm256_load_si256((__m256i*) & arr[i + 48]);

                                //Compute min
                                _min = _mm256_min_epu16(_first, _second);
                                _min = _mm256_min_epu16(_min, _third);
                                _min = _mm256_min_epu16(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 16; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m256i_u16[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
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
                            __m512i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 64)
                            {
                                if ((i + 64) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm512_load_epi32((__m512i*)&arr[i]);
                                _second = _mm512_load_epi32((__m512i*)&arr[i + 16]);
                                _third = _mm512_load_epi32((__m512i*)&arr[i + 32]);
                                _forth = _mm512_load_epi32((__m512i*)&arr[i + 48]);

                                //Compute min
                                _min = _mm512_min_epi32(_first, _second);
                                _min = _mm512_min_epi32(_min, _third);
                                _min = _mm512_min_epi32(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 16; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m512i_i32[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm256_load_si256((__m256i*) & arr[i]);
                                _second = _mm256_load_si256((__m256i*) & arr[i + 8]);
                                _third = _mm256_load_si256((__m256i*) & arr[i + 16]);
                                _forth = _mm256_load_si256((__m256i*) & arr[i + 24]);

                                //Compute min
                                _min = _mm256_min_epi32(_first, _second);
                                _min = _mm256_min_epi32(_min, _third);
                                _min = _mm256_min_epi32(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 8; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m256i_i32[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
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
                            __m512i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 64)
                            {
                                if ((i + 64) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm512_load_epi32((__m512i*)&arr[i]);
                                _second = _mm512_load_epi32((__m512i*)&arr[i + 16]);
                                _third = _mm512_load_epi32((__m512i*)&arr[i + 32]);
                                _forth = _mm512_load_epi32((__m512i*)&arr[i + 48]);

                                //Compute min
                                _min = _mm512_min_epu32(_first, _second);
                                _min = _mm512_min_epu32(_min, _third);
                                _min = _mm512_min_epu32(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 16; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m512i_u32[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm256_load_si256((__m256i*) & arr[i]);
                                _second = _mm256_load_si256((__m256i*) & arr[i + 8]);
                                _third = _mm256_load_si256((__m256i*) & arr[i + 16]);
                                _forth = _mm256_load_si256((__m256i*) & arr[i + 24]);

                                //Compute min
                                _min = _mm256_min_epu32(_first, _second);
                                _min = _mm256_min_epu32(_min, _third);
                                _min = _mm256_min_epu32(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 8; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m256i_u32[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
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
                            __m512i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm512_load_epi64((__m512i*)&arr[i]);
                                _second = _mm512_load_epi64((__m512i*)&arr[i + 8]);
                                _third = _mm512_load_epi64((__m512i*)&arr[i + 16]);
                                _forth = _mm512_load_epi64((__m512i*)&arr[i + 24]);

                                //Compute min
                                _min = _mm512_min_epi64(_first, _second);
                                _min = _mm512_min_epi64(_min, _third);
                                _min = _mm512_min_epi64(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 8; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m512i_i64[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 16)
                            {
                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm256_load_si256((__m256i*) & arr[i]);
                                _second = _mm256_load_si256((__m256i*) & arr[i + 4]);
                                _third = _mm256_load_si256((__m256i*) & arr[i + 8]);
                                _forth = _mm256_load_si256((__m256i*) & arr[i + 12]);

                                //Compute min
                                _min = _mm256_castpd_si256(_mm256_min_pd(_mm256_castsi256_pd(_first), _mm256_castsi256_pd(_second)));
                                _min = _mm256_castpd_si256(_mm256_min_pd(_mm256_castsi256_pd(_min), _mm256_castsi256_pd(_third)));
                                _min = _mm256_castpd_si256(_mm256_min_pd(_mm256_castsi256_pd(_min), _mm256_castsi256_pd(_forth)));

                                //Reduce min
                                for (size_t x = 0; x != 4; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m256i_i64[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
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
                            __m512i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm512_load_epi64((__m512i*) &arr[i]);
                                _second = _mm512_load_epi64((__m512i*)&arr[i + 8]);
                                _third = _mm512_load_epi64((__m512i*) &arr[i + 16]);
                                _forth = _mm512_load_epi64((__m512i*) &arr[i + 24]);

                                //Compute min
                                _min = _mm512_min_epu64(_first, _second);
                                _min = _mm512_min_epu64(_min, _third);
                                _min = _mm512_min_epu64(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 8; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m512i_u64[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _first, _second, _third, _forth, _min;

                            for (; i != end; i += 16)
                            {
                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm256_load_si256((__m256i*) &arr[i]);
                                _second = _mm256_load_si256((__m256i*)&arr[i + 4]);
                                _third = _mm256_load_si256((__m256i*) &arr[i + 8]);
                                _forth = _mm256_load_si256((__m256i*) &arr[i + 12]);

                                //Compute min
                                _min = _mm256_castpd_si256(_mm256_min_pd(_mm256_castsi256_pd(_first), _mm256_castsi256_pd(_second)));
                                _min = _mm256_castpd_si256(_mm256_min_pd(_mm256_castsi256_pd(_min), _mm256_castsi256_pd(_third)));
                                _min = _mm256_castpd_si256(_mm256_min_pd(_mm256_castsi256_pd(_min), _mm256_castsi256_pd(_forth)));

                                //Reduce min
                                for (size_t x = 0; x != 4; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m256i_u64[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
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
                            __m512 _first, _second, _third, _forth, _min;

                            for (; i != end; i += 64)
                            {
                                if ((i + 64) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm512_load_ps(&arr[i]);
                                _second = _mm512_load_ps(&arr[i + 16]);
                                _third = _mm512_load_ps(&arr[i + 32]);
                                _forth = _mm512_load_ps(&arr[i + 48]);

                                //Compute min
                                _min = _mm512_min_ps(_first, _second);
                                _min = _mm512_min_ps(_min, _third);
                                _min = _mm512_min_ps(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 16; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m512_f32[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX512
                        else if (tpa::hasAVX)
                        {         
                            __m256 _first, _second, _third, _forth, _min;

                            for (; i != end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm256_load_ps(&arr[i]);
                                _second = _mm256_load_ps(&arr[i+8]);
                                _third = _mm256_load_ps(&arr[i + 16]);
                                _forth = _mm256_load_ps(&arr[i + 24]);

                                //Compute min
                                _min = _mm256_min_ps(_first, _second);
                                _min = _mm256_min_ps(_min, _third);
                                _min = _mm256_min_ps(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 8; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m256_f32[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX
#endif
                    }//End if
#pragma endregion
#pragma region double
                    else if constexpr (std::is_same<T, double>())
                    {
#ifdef TPA_X86_64
                        if (tpa::hasAVX512)
                        {                 
                            __m512d _first, _second, _third, _forth, _min;

                            for (; i != end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm512_load_pd(&arr[i]);
                                _second = _mm512_load_pd(&arr[i + 8]);
                                _third = _mm512_load_pd(&arr[i + 16]);
                                _forth = _mm512_load_pd(&arr[i + 24]);

                                //Compute min
                                _min = _mm512_min_pd(_first, _second);
                                _min = _mm512_min_pd(_min, _third);
                                _min = _mm512_min_pd(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 8; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m512d_f64[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX512
                        else if (tpa::hasAVX)
                        {         
                            __m256d _first, _second, _third, _forth, _min;

                            for (; i != end; i += 16)
                            {
                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Data
                                _first = _mm256_load_pd(&arr[i]);
                                _second = _mm256_load_pd(&arr[i+4]);
                                _third = _mm256_load_pd(&arr[i + 8]);
                                _forth = _mm256_load_pd(&arr[i + 12]);

                                //Compute min
                                _min = _mm256_min_pd(_first, _second);
                                _min = _mm256_min_pd(_min, _third);
                                _min = _mm256_min_pd(_min, _forth);

                                //Reduce min
                                for (size_t x = 0; x != 4; ++x)
                                {
#ifdef _MSC_VER
                                    temp_min = tpa::util::min(temp_min, _min.m256d_f64[x]);
#else
                                    temp_min = tpa::util::min(temp_min, _min[x]);
#endif
                                }//End for                                    
                            }//End for
                        }//End if hasAVX
#endif
                    }//End if
#pragma endregion
#pragma region generic      
                    for (; i != end; ++i)
                    {
                        temp_min = tpa::util::min(temp_min, arr[i]);
                    }//End for
#pragma endregion

                        return temp_min;
                    });//End of lambda

                results.emplace_back(std::move(temp));
            }//End for 

            for (const auto& fut : results)
            {
                min = tpa::util::min(min, fut.get());
                complete += 1;
            }//End for

            //Check all threads completed
            if (complete != tpa::nThreads)
            {
                throw tpa::exceptions::NotAllThreadsCompleted(complete);
            }//End if

            //Finish
            return min;
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::min_element(): " << ex.code()
                << " " << ex.what() << "\n";
            return static_cast<T>(0);
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::min_element(): " << ex.what() << "\n";
            return static_cast<T>(0);
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::min_element(): " << ex.what() << "\n";
            return static_cast<T>(0);
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::min_element(): unknown!\n";
            return static_cast<T>(0);
        }//End catch
	}//End of min_element

    /// <summary>
    /// <para>Returns a copy of the smallest element in the container.</para>
    /// <para>Requires a predicate function which should return FALSE when lhs is less than rhs.</para>
    /// <para>This parallel implementation uses Multi-Threading Only.</para>
    /// <para>The return type is the value_type of the container.</para>
    /// <para>If passing an container containing no elements, will throw an exception and return 0.</para>
    /// <para>IMPORTANT: This implementation is intended to be used with non-numeric custom classes, if your container's value_type is numeric, use the implementation without a predicate for a performance increase!</para>
    /// </summary>
    /// <typeparam name="CONTAINER_T"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="PRED"></typeparam>
    /// <param name="arr"></param>
    /// <param name="p"></param>
    /// <returns></returns>
    template<class CONTAINER_T, typename T = CONTAINER_T::value_type, class PRED>
    [[nodiscard]] inline constexpr T min_element(const  CONTAINER_T& arr, const PRED pred)
    requires tpa::util::contiguous_seqeunce<CONTAINER_T>
    {
        try
        {
            uint32_t complete = 0;

            //Guard against zero-element arrays
            if (arr.size() == 0)
            {
                throw tpa::exceptions::EmptyArray();
            }//End if

            T min = arr[0];

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, arr.size());

            std::vector<std::shared_future<T>> results;
            results.reserve(tpa::nThreads);

            std::shared_future<T> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&arr, &pred, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

                        T temp_min = arr[beg];
#pragma region generic
                        for (; i != end; ++i)
                        {
                            if (pred(temp_min, arr[i]))
                            {
                                temp_min = arr[i];
                            }//End if
                        }//End for
#pragma endregion
                        return temp_min;
                    });//End of lambda

                results.emplace_back(std::move(temp));
            }//End for

            for (const auto& fut : results)
            {
                if (pred(min, fut.get()))
                {
                    min = fut.get();
                }//End if

                complete += 1;
            }//End for

            //Check all threads completed
            if (complete != tpa::nThreads)
            {
                throw tpa::exceptions::NotAllThreadsCompleted(complete);
            }//End if

            //Finish
            return min;
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::min_element(): " << ex.code()
                << " " << ex.what() << "\n";
            return static_cast<T>(0);
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::min_element(): " << ex.what() << "\n";
            return static_cast<T>(0);
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::min_element(): " << ex.what() << "\n";
            return static_cast<T>(0);
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::min_element(): unknown!\n";
            return static_cast<T>(0);
        }//End catch
    }//End of min_element
#pragma endregion
}//End of namespace
