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
#include <algorithm>
#include <execution>
#include <array>
#include <vector>
#include <forward_list>
#include <unordered_set>
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
#include "../tpa_macros.hpp"
#include "../predicates.hpp"
#include "../tpa_concepts.hpp"
#include "../InstructionSet.hpp"
#include "simd.hpp"

#ifdef _M_AMD64
#include <immintrin.h>
#elif defined (_M_ARM64)
#ifdef _MSC_VER 
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
		[[nodiscard]] inline constexpr auto mean(T a, Ts... args) noexcept ->  decltype(a + (args + ...))
		{
			return (a + (args + ...)) / (sizeof...(args) + 1);
		}//End of mean


        /// <summary>
        /// <para>Calculates and returns the average of the values in 'arr'</para>
        /// <para>The return type is a template parameter which must satisfy the requirements of tpa::util::calculatable.</para>
        /// <para>Uses Multi-Threading and SIMD (where available).</para>
        /// <para>To use SIMD the container's value_type and the return type must be indentical in order to avoid overflow.</para>
        /// <para>If you know the values in your container will not cause overflow you can pass 'true' to the parameter: ignore_overflow, this will force the use of SIMD even in cases where overflow is possible (may cause an incorrect result!)</para>
        /// </summary>
        /// <typeparam name="RETURN_T"></typeparam>
        /// <typeparam name="CONTAINER_T"></typeparam>
        /// <param name="arr"></param>
        /// <param name="ignore_overflow"></param>
        /// <returns></returns>
        template<tpa::util::calculatable RETURN_T, typename CONTAINER_T>
        [[nodiscard]] inline constexpr RETURN_T mean(const CONTAINER_T& arr, const bool ignore_overflow = false) requires tpa::util::contiguous_seqeunce<CONTAINER_T>
        {
            using T = CONTAINER_T::value_type;

            static_assert(tpa::util::calculatable<T>, "Error in tpa::stat::mean! The value_type of this container does not meet the requirements of tpa::util::calculatable! ");

            try
            {
                uint32_t complete = 0u;

                RETURN_T sum = 0;

                std::vector<std::pair<size_t, size_t>> sections;
                tpa::util::prepareThreading(sections, arr.size());

                std::vector<std::shared_future<RETURN_T>> results;
                results.reserve(tpa::nThreads);

                std::shared_future<RETURN_T> temp;

                for (const auto& sec : sections)
                {
                    temp = tpa::tp->addTask([&arr, &ignore_overflow, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

                        RETURN_T temp_val = 0;

#pragma region byte
                    if constexpr (std::is_same<T, int8_t>())
                    {
                        if (std::is_same<T, RETURN_T>() || ignore_overflow)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX512_ByteWord)
                            {
                                __m512i _a{}, _b{}, _c{}, _d{}, _sum{};

                                for (; (i + 256uz) < end; i += 256uz)
                                {
                                    _a = _mm512_loadu_epi8(&arr[i]);
                                    _b = _mm512_loadu_epi8(&arr[i + 64uz]);
                                    _c = _mm512_loadu_epi8(&arr[i + 128uz]);
                                    _d = _mm512_loadu_epi8(&arr[i + 192uz]);

                                    _sum = _mm512_add_epi8(_a, _b);
                                    _sum = _mm512_add_epi8(_sum, _c);
                                    _sum = _mm512_add_epi8(_sum, _d);

                                    for (size_t x = 0uz; x < 64uz; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += static_cast<RETURN_T>(_sum.m512i_i8[x]);
#else
                                        temp_val += static_cast<RETURN_T>(_sum[x]);
#endif
                                    }//End for
                                }//End for

                            }//End if hasAVX512
                            else if (tpa::hasAVX2)
                            {
                                __m256i _a{}, _b{}, _c{}, _d{}, _sum{};

                                for (; (i + 128uz) < end; i += 128uz)
                                {
                                    _a = _mm256_load_si256((__m256i*) &arr[i]);
                                    _b = _mm256_load_si256((__m256i*) &arr[i + 32uz]);
                                    _c = _mm256_load_si256((__m256i*) &arr[i + 64uz]);
                                    _d = _mm256_load_si256((__m256i*) &arr[i + 96uz]);

                                    _sum = _mm256_add_epi8(_a, _b);
                                    _sum = _mm256_add_epi8(_sum, _c);
                                    _sum = _mm256_add_epi8(_sum, _d);

                                    for (size_t x = 0uz; x != 32uz; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256i_i8[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for

                            }//End if hasAVX
                            else if (tpa::has_SSE2)
                            {
                                __m128i _a{}, _b{}, _c{}, _d{}, _sum{};

                                for (; (i + 64uz) < end; i += 64uz)
                                {
                                    _a = _mm_load_si128((__m128i*) &arr[i]);
                                    _b = _mm_load_si128((__m128i*) &arr[i + 16uz]);
                                    _c = _mm_load_si128((__m128i*) &arr[i + 32uz]);
                                    _d = _mm_load_si128((__m128i*) &arr[i + 48uz]);

                                    _sum = _mm_add_epi8(_a, _b);
                                    _sum = _mm_add_epi8(_sum, _c);
                                    _sum = _mm_add_epi8(_sum, _d);

                                    for (size_t x = 0uz; x < 16uz; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m128i_i8[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for

                            }//End if has_SSE
#endif
                        }//End if
                    }//End if
#pragma endregion
#pragma region unsigned byte
                if constexpr (std::is_same<T, uint8_t>())
                {
                    if (std::is_same<T, RETURN_T>() || ignore_overflow)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX512_ByteWord)
                        {
                            __m512i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 256uz) < end; i += 256uz)
                            {
                                _a = _mm512_loadu_epi8(&arr[i]);
                                _b = _mm512_loadu_epi8(&arr[i + 64uz]);
                                _c = _mm512_loadu_epi8(&arr[i + 128uz]);
                                _d = _mm512_loadu_epi8(&arr[i + 192uz]);

                                _sum = _mm512_add_epi8(_a, _b);
                                _sum = _mm512_add_epi8(_sum, _c);
                                _sum = _mm512_add_epi8(_sum, _d);

                                for (size_t x = 0uz; x < 64uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += static_cast<RETURN_T>(_sum.m512i_u8[x]);
#else
                                    temp_val += static_cast<RETURN_T>(_sum[x]);
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 128uz) < end; i += 128uz)
                            {
                                _a = _mm256_load_si256((__m256i*) & arr[i]);
                                _b = _mm256_load_si256((__m256i*) & arr[i + 32uz]);
                                _c = _mm256_load_si256((__m256i*) & arr[i + 64uz]);
                                _d = _mm256_load_si256((__m256i*) & arr[i + 96uz]);

                                _sum = _mm256_add_epi8(_a, _b);
                                _sum = _mm256_add_epi8(_sum, _c);
                                _sum = _mm256_add_epi8(_sum, _d);

                                for (size_t x = 0uz; x != 32uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m256i_u8[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX
                        else if (tpa::has_SSE2)
                        {
                            __m128i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 64uz) < end; i += 64uz)
                            {
                                _a = _mm_load_si128((__m128i*) & arr[i]);
                                _b = _mm_load_si128((__m128i*) & arr[i + 16uz]);
                                _c = _mm_load_si128((__m128i*) & arr[i + 32uz]);
                                _d = _mm_load_si128((__m128i*) & arr[i + 48uz]);

                                _sum = _mm_add_epi8(_a, _b);
                                _sum = _mm_add_epi8(_sum, _c);
                                _sum = _mm_add_epi8(_sum, _d);

                                for (size_t x = 0uz; x < 16uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m128i_u8[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if has_SSE
#endif
                    }//End if
                }//End if
#pragma endregion
#pragma region short
                if constexpr (std::is_same<T, int16_t>())
                {
                    if (std::is_same<T, RETURN_T>() || ignore_overflow)
                    {
    #ifdef _M_AMD64
                        if (tpa::hasAVX512_ByteWord)
                        {
                            __m512i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 128uz) < end; i += 128uz)
                            {
                                _a = _mm512_loadu_epi16(&arr[i]);
                                _b = _mm512_loadu_epi16(&arr[i + 32uz]);
                                _c = _mm512_loadu_epi16(&arr[i + 64uz]);
                                _d = _mm512_loadu_epi16(&arr[i + 96uz]);

                                _sum = _mm512_add_epi16(_a, _b);
                                _sum = _mm512_add_epi16(_sum, _c);
                                _sum = _mm512_add_epi16(_sum, _d);

                                for (size_t x = 0uz; x < 32uz; ++x)
                                {
    #ifdef _WIN32
                                    temp_val += static_cast<RETURN_T>(_sum.m512i_i16[x]);
    #else
                                    temp_val += static_cast<RETURN_T>(_sum[x]);
    #endif
                                }//End for
                            }//End for

                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 64uz) < end; i += 64uz)
                            {
                                _a = _mm256_load_si256((__m256i*) & arr[i]);
                                _b = _mm256_load_si256((__m256i*) & arr[i + 16uz]);
                                _c = _mm256_load_si256((__m256i*) & arr[i + 32uz]);
                                _d = _mm256_load_si256((__m256i*) & arr[i + 48uz]);

                                _sum = _mm256_add_epi16(_a, _b);
                                _sum = _mm256_add_epi16(_sum, _c);
                                _sum = _mm256_add_epi16(_sum, _d);

                                for (size_t x = 0uz; x != 16uz; ++x)
                                {
    #ifdef _WIN32
                                    temp_val += _sum.m256i_i16[x];
    #else
                                    temp_val += _sum[x];
    #endif
                                }//End for
                            }//End for

                        }//End if hasAVX
                        else if (tpa::has_SSE2)
                        {
                            __m128i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                _a = _mm_load_si128((__m128i*) & arr[i]);
                                _b = _mm_load_si128((__m128i*) & arr[i + 8uz]);
                                _c = _mm_load_si128((__m128i*) & arr[i + 16uz]);
                                _d = _mm_load_si128((__m128i*) & arr[i + 24uz]);

                                _sum = _mm_add_epi16(_a, _b);
                                _sum = _mm_add_epi16(_sum, _c);
                                _sum = _mm_add_epi16(_sum, _d);

                                for (size_t x = 0uz; x < 8uz; ++x)
                                {
    #ifdef _WIN32
                                    temp_val += _sum.m128i_i16[x];
    #else
                                    temp_val += _sum[x];
    #endif
                                }//End for
                            }//End for

                        }//End if has_SSE
    #endif
                    }//End if
                }//End if
#pragma endregion
#pragma region unsigned short
                if constexpr (std::is_same<T, uint16_t>())
                {
                    if (std::is_same<T, RETURN_T>() || ignore_overflow)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX512_ByteWord)
                        {
                            __m512i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 128uz) < end; i += 128uz)
                            {
                                _a = _mm512_loadu_epi16(&arr[i]);
                                _b = _mm512_loadu_epi16(&arr[i + 32uz]);
                                _c = _mm512_loadu_epi16(&arr[i + 64uz]);
                                _d = _mm512_loadu_epi16(&arr[i + 96uz]);

                                _sum = _mm512_add_epi16(_a, _b);
                                _sum = _mm512_add_epi16(_sum, _c);
                                _sum = _mm512_add_epi16(_sum, _d);

                                for (size_t x = 0uz; x < 32uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += static_cast<RETURN_T>(_sum.m512i_u16[x]);
#else
                                    temp_val += static_cast<RETURN_T>(_sum[x]);
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 64uz) < end; i += 64uz)
                            {
                                _a = _mm256_load_si256((__m256i*) & arr[i]);
                                _b = _mm256_load_si256((__m256i*) & arr[i + 16uz]);
                                _c = _mm256_load_si256((__m256i*) & arr[i + 32uz]);
                                _d = _mm256_load_si256((__m256i*) & arr[i + 48uz]);

                                _sum = _mm256_add_epi16(_a, _b);
                                _sum = _mm256_add_epi16(_sum, _c);
                                _sum = _mm256_add_epi16(_sum, _d);

                                for (size_t x = 0uz; x != 16uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m256i_u16[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX
                        else if (tpa::has_SSE2)
                        {
                            __m128i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                _a = _mm_load_si128((__m128i*) & arr[i]);
                                _b = _mm_load_si128((__m128i*) & arr[i + 8uz]);
                                _c = _mm_load_si128((__m128i*) & arr[i + 16uz]);
                                _d = _mm_load_si128((__m128i*) & arr[i + 24uz]);

                                _sum = _mm_add_epi16(_a, _b);
                                _sum = _mm_add_epi16(_sum, _c);
                                _sum = _mm_add_epi16(_sum, _d);

                                for (size_t x = 0uz; x < 8uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m128i_u16[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if has_SSE
#endif
                    }//End if
                }//End if
#pragma endregion
#pragma region int
                if constexpr (std::is_same<T, int32_t>())
                {
                    if (std::is_same<T, RETURN_T>() || ignore_overflow)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX512)
                        {
                            __m512i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 64uz) < end; i += 64uz)
                            {
                                _a = _mm512_load_epi32(&arr[i]);
                                _b = _mm512_load_epi32(&arr[i + 16uz]);
                                _c = _mm512_load_epi32(&arr[i + 32uz]);
                                _d = _mm512_load_epi32(&arr[i + 48uz]);

                                _sum = _mm512_add_epi32(_a, _b);
                                _sum = _mm512_add_epi32(_sum, _c);
                                _sum = _mm512_add_epi32(_sum, _d);

                                for (size_t x = 0uz; x < 16uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += static_cast<RETURN_T>(_sum.m512i_i32[x]);
#else
                                    temp_val += static_cast<RETURN_T>(_sum[x]);
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                _a = _mm256_load_si256((__m256i*) & arr[i]);
                                _b = _mm256_load_si256((__m256i*) & arr[i + 8uz]);
                                _c = _mm256_load_si256((__m256i*) & arr[i + 16uz]);
                                _d = _mm256_load_si256((__m256i*) & arr[i + 24uz]);

                                _sum = _mm256_add_epi32(_a, _b);
                                _sum = _mm256_add_epi32(_sum, _c);
                                _sum = _mm256_add_epi32(_sum, _d);

                                for (size_t x = 0uz; x != 8uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m256i_i32[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX
                        else if (tpa::has_SSE2)
                        {
                            __m128i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _a = _mm_load_si128((__m128i*) & arr[i]);
                                _b = _mm_load_si128((__m128i*) & arr[i + 4uz]);
                                _c = _mm_load_si128((__m128i*) & arr[i + 8uz]);
                                _d = _mm_load_si128((__m128i*) & arr[i + 12uz]);

                                _sum = _mm_add_epi32(_a, _b);
                                _sum = _mm_add_epi32(_sum, _c);
                                _sum = _mm_add_epi32(_sum, _d);

                                for (size_t x = 0uz; x < 4uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m128i_i32[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if has_SSE
#endif
                    }//End if
                }//End if
#pragma endregion
#pragma region unsigned int
                if constexpr (std::is_same<T, uint32_t>())
                {
                    if (std::is_same<T, RETURN_T>() || ignore_overflow)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX512)
                        {
                            __m512i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 64uz) < end; i += 64uz)
                            {
                                _a = _mm512_load_epi32(&arr[i]);
                                _b = _mm512_load_epi32(&arr[i + 16uz]);
                                _c = _mm512_load_epi32(&arr[i + 32uz]);
                                _d = _mm512_load_epi32(&arr[i + 48uz]);

                                _sum = _mm512_add_epi32(_a, _b);
                                _sum = _mm512_add_epi32(_sum, _c);
                                _sum = _mm512_add_epi32(_sum, _d);

                                for (size_t x = 0uz; x < 16uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += static_cast<RETURN_T>(_sum.m512i_u32[x]);
#else
                                    temp_val += static_cast<RETURN_T>(_sum[x]);
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                _a = _mm256_load_si256((__m256i*) & arr[i]);
                                _b = _mm256_load_si256((__m256i*) & arr[i + 8uz]);
                                _c = _mm256_load_si256((__m256i*) & arr[i + 16uz]);
                                _d = _mm256_load_si256((__m256i*) & arr[i + 24uz]);

                                _sum = _mm256_add_epi32(_a, _b);
                                _sum = _mm256_add_epi32(_sum, _c);
                                _sum = _mm256_add_epi32(_sum, _d);

                                for (size_t x = 0uz; x != 8uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m256i_u32[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX
                        else if (tpa::has_SSE2)
                        {
                            __m128i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _a = _mm_load_si128((__m128i*) & arr[i]);
                                _b = _mm_load_si128((__m128i*) & arr[i + 4uz]);
                                _c = _mm_load_si128((__m128i*) & arr[i + 8uz]);
                                _d = _mm_load_si128((__m128i*) & arr[i + 12uz]);

                                _sum = _mm_add_epi32(_a, _b);
                                _sum = _mm_add_epi32(_sum, _c);
                                _sum = _mm_add_epi32(_sum, _d);

                                for (size_t x = 0uz; x < 4uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m128i_u32[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if has_SSE
#endif
                    }//End if
                }//End if
#pragma endregion
#pragma region long
                if constexpr (std::is_same<T, int64_t>())
                {
                    if (std::is_same<T, RETURN_T>() || ignore_overflow)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX512)
                        {
                            __m512i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                _a = _mm512_load_epi64(&arr[i]);
                                _b = _mm512_load_epi64(&arr[i + 8uz]);
                                _c = _mm512_load_epi64(&arr[i + 16uz]);
                                _d = _mm512_load_epi64(&arr[i + 24uz]);

                                _sum = _mm512_add_epi64(_a, _b);
                                _sum = _mm512_add_epi64(_sum, _c);
                                _sum = _mm512_add_epi64(_sum, _d);

                                for (size_t x = 0uz; x < 8uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += static_cast<RETURN_T>(_sum.m512i_i64[x]);
#else
                                    temp_val += static_cast<RETURN_T>(_sum[x]);
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _a = _mm256_load_si256((__m256i*) & arr[i]);
                                _b = _mm256_load_si256((__m256i*) & arr[i + 4uz]);
                                _c = _mm256_load_si256((__m256i*) & arr[i + 8uz]);
                                _d = _mm256_load_si256((__m256i*) & arr[i + 12uz]);

                                _sum = _mm256_add_epi64(_a, _b);
                                _sum = _mm256_add_epi64(_sum, _c);
                                _sum = _mm256_add_epi64(_sum, _d);

                                for (size_t x = 0uz; x != 4uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m256i_i64[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX
                        else if (tpa::has_SSE2)
                        {
                            __m128i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _a = _mm_load_si128((__m128i*) & arr[i]);
                                _b = _mm_load_si128((__m128i*) & arr[i + 2uz]);
                                _c = _mm_load_si128((__m128i*) & arr[i + 4uz]);
                                _d = _mm_load_si128((__m128i*) & arr[i + 6uz]);

                                _sum = _mm_add_epi64(_a, _b);
                                _sum = _mm_add_epi64(_sum, _c);
                                _sum = _mm_add_epi64(_sum, _d);

                                for (size_t x = 0uz; x < 2uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m128i_i64[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if has_SSE
#endif
                    }//End if
                }//End if
#pragma endregion
#pragma region unsigned long
                if constexpr (std::is_same<T, uint64_t>())
                {
                    if (std::is_same<T, RETURN_T>() || ignore_overflow)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX512)
                        {
                            __m512i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 32uz) < end; i += 32uz)
                            {
                                _a = _mm512_load_epi64(&arr[i]);
                                _b = _mm512_load_epi64(&arr[i + 8uz]);
                                _c = _mm512_load_epi64(&arr[i + 16uz]);
                                _d = _mm512_load_epi64(&arr[i + 24uz]);

                                _sum = _mm512_add_epi64(_a, _b);
                                _sum = _mm512_add_epi64(_sum, _c);
                                _sum = _mm512_add_epi64(_sum, _d);

                                for (size_t x = 0uz; x < 8uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += static_cast<RETURN_T>(_sum.m512i_u64[x]);
#else
                                    temp_val += static_cast<RETURN_T>(_sum[x]);
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX512
                        else if (tpa::hasAVX2)
                        {
                            __m256i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 16uz) < end; i += 16uz)
                            {
                                _a = _mm256_load_si256((__m256i*) & arr[i]);
                                _b = _mm256_load_si256((__m256i*) & arr[i + 4uz]);
                                _c = _mm256_load_si256((__m256i*) & arr[i + 8uz]);
                                _d = _mm256_load_si256((__m256i*) & arr[i + 12uz]);

                                _sum = _mm256_add_epi64(_a, _b);
                                _sum = _mm256_add_epi64(_sum, _c);
                                _sum = _mm256_add_epi64(_sum, _d);

                                for (size_t x = 0uz; x != 4uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m256i_u64[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if hasAVX
                        else if (tpa::has_SSE2)
                        {
                            __m128i _a{}, _b{}, _c{}, _d{}, _sum{};

                            for (; (i + 8uz) < end; i += 8uz)
                            {
                                _a = _mm_load_si128((__m128i*) & arr[i]);
                                _b = _mm_load_si128((__m128i*) & arr[i + 2uz]);
                                _c = _mm_load_si128((__m128i*) & arr[i + 4uz]);
                                _d = _mm_load_si128((__m128i*) & arr[i + 6uz]);

                                _sum = _mm_add_epi64(_a, _b);
                                _sum = _mm_add_epi64(_sum, _c);
                                _sum = _mm_add_epi64(_sum, _d);

                                for (size_t x = 0uz; x < 2uz; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m128i_u64[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for

                        }//End if has_SSE
#endif
                    }//End if
                }//End if
#pragma endregion
#pragma region float
                        if constexpr (std::is_same<T, float>())
                        {
                            if (std::is_same<T, RETURN_T>() || ignore_overflow)
                            {
#ifdef _M_AMD64
                                if (tpa::hasAVX512)
                                {
                                    __m512 _a{}, _b{}, _c{}, _d{}, _sum{};

                                    for (; (i + 64uz) < end; i += 64uz)
                                    {
                                        _a = _mm512_load_ps(&arr[i]);
                                        _b = _mm512_load_ps(&arr[i + 16uz]);
                                        _c = _mm512_load_ps(&arr[i + 32uz]);
                                        _d = _mm512_load_ps(&arr[i + 48uz]);

                                        _sum = _mm512_add_ps(_a, _b);
                                        _sum = _mm512_add_ps(_sum, _c);
                                        _sum = _mm512_add_ps(_sum, _d);

                                        for (size_t x = 0uz; x < 16uz; ++x)
                                        {
#ifdef _WIN32
                                            temp_val += static_cast<RETURN_T>(_sum.m512_f32[x]);
#else
                                            temp_val += static_cast<RETURN_T>(_sum[x]);
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
                                        _b = _mm256_load_ps(&arr[i + 8uz]);
                                        _c = _mm256_load_ps(&arr[i + 16uz]);
                                        _d = _mm256_load_ps(&arr[i + 24uz]);

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
                                        _b = _mm_load_ps(&arr[i + 4uz]);
                                        _c = _mm_load_ps(&arr[i + 8uz]);
                                        _d = _mm_load_ps(&arr[i + 12uz]);

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
                        }//End if
#pragma endregion
#pragma region double
                    if constexpr (std::is_same<T, double>())
                    {
                        if (std::is_same<T, RETURN_T>() || ignore_overflow)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX512)
                            {
                                __m512d _a{}, _b{}, _c{}, _d{}, _sum{};

                                for (; (i + 32uz) < end; i += 32uz)
                                {
                                    _a = _mm512_load_pd(&arr[i]);
                                    _b = _mm512_load_pd(&arr[i + 8uz]);
                                    _c = _mm512_load_pd(&arr[i + 16uz]);
                                    _d = _mm512_load_pd(&arr[i + 24uz]);

                                    _sum = _mm512_add_pd(_a, _b);
                                    _sum = _mm512_add_pd(_sum, _c);
                                    _sum = _mm512_add_pd(_sum, _d);

                                    for (size_t x = 0uz; x < 8uz; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += static_cast<RETURN_T>(_sum.m512d_f64[x]);
#else
                                        temp_val += static_cast<RETURN_T>(_sum[x]);
#endif
                                    }//End for
                                }//End for

                            }//End if hasAVX512
                            else if (tpa::hasAVX)
                            {
                                __m256d _a{}, _b{}, _c{}, _d{}, _sum{};

                                for (; (i + 16uz) < end; i += 16uz)
                                {
                                    _a = _mm256_load_pd(&arr[i]);
                                    _b = _mm256_load_pd(&arr[i + 4uz]);
                                    _c = _mm256_load_pd(&arr[i + 8uz]);
                                    _d = _mm256_load_pd(&arr[i + 12uz]);

                                    _sum = _mm256_add_pd(_a, _b);
                                    _sum = _mm256_add_pd(_sum, _c);
                                    _sum = _mm256_add_pd(_sum, _d);

                                    for (size_t x = 0uz; x != 4uz; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256d_f64[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for

                            }//End if hasAVX
                            else if (tpa::has_SSE2)
                            {
                                __m128d _a{}, _b{}, _c{}, _d{}, _sum{};

                                for (; (i + 8uz) < end; i += 8uz)
                                {
                                    _a = _mm_load_pd(&arr[i]);
                                    _b = _mm_load_pd(&arr[i + 2uz]);
                                    _c = _mm_load_pd(&arr[i + 4uz]);
                                    _d = _mm_load_pd(&arr[i + 6uz]);

                                    _sum = _mm_add_pd(_a, _b);
                                    _sum = _mm_add_pd(_sum, _c);
                                    _sum = _mm_add_pd(_sum, _d);

                                    for (size_t x = 0uz; x < 2uz; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m128d_f64[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for

                            }//End if has_SSE
#endif
                        }//End if
                    }//End if
#pragma endregion
#pragma region generic      
                            for (; i != end; ++i)
                            {
                                temp_val += static_cast<RETURN_T>(arr[i]);
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
                return sum / static_cast<RETURN_T>(arr.size());

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

        /// <summary>
        /// <para>Compute and return the median of the passed parameters.</para>
        /// <para>This algorithim sorts the container if it is unsorted.</para>
        /// <para>Note that if you pass only integers, integer division will be used.
        /// To get a floating-point result, pass at least one (1) floating-point type</para>
        /// </summary>
        /// <param name="a"></param>
        /// <param name="...args"></param>
        /// <returns></returns>
        template<tpa::util::calculatable T, tpa::util::calculatable...Ts>
        [[nodiscard]] inline constexpr auto median(T a, Ts... args) noexcept ->  decltype(a + (args + ...))
        {
            std::array<T, sizeof...(args) + 1> arr{};
            arr[0] = static_cast<T>(a);
            size_t x = 1uz;
            for (auto i : std::initializer_list<std::common_type_t<Ts...> >{ args... })
            {
                arr[x] = i;
                ++x;
            }//End for

            if (!std::is_sorted(std::execution::par_unseq, arr.cbegin(), arr.cend()))
            {
                std::sort(std::execution::par_unseq, arr.begin(), arr.end());
            }//End if

            if (arr.size() % 2uz != 0uz)
            {
                return static_cast<T>(arr[arr.size() / 2uz]);
            }//End if
            else
            {
                return static_cast<T>(arr[(arr.size() - 1uz) / 2uz] + arr[arr.size() / 2]) / 2;
            }//End else
        }//End of median

        template<tpa::util::calculatable RETURN_T, typename CONTAINER_T>
        [[nodiscard]] inline constexpr RETURN_T median(CONTAINER_T& arr) noexcept requires tpa::util::contiguous_seqeunce<CONTAINER_T>
        {
            using T = CONTAINER_T::value_type;

            static_assert(tpa::util::calculatable<T>, "Error in tpa::stat::median! The value_type of this container does not meet the requirements of tpa::util::calculatable! ");

            if (!std::is_sorted(std::execution::par_unseq, arr.cbegin(), arr.cend()))
            {
                std::sort(std::execution::par_unseq, arr.begin(), arr.end());
            }//End if

            if (arr.size() % 2uz != 0uz)
            {
                return static_cast<RETURN_T>(arr[arr.size() / 2uz]);
            }//End if
            else
            {
                return static_cast<RETURN_T>( arr[(arr.size() - 1uz) / 2uz] + arr[arr.size() / 2]) / (RETURN_T)2;
            }//End else
        }//End of median

        /// <summary>
        /// <para>Compute and return the mode(s) of the passed parameters.</para>
        /// <para>Returns an std vector of std pair (T,size_t)  where the 'first' member is the mode(s) and 'second' member is the number of occurrences</para>
        /// <para>This function does not require the input to be sorted and does not sort the input.</para>        
        /// </summary>
        /// <param name="a"></param>
        /// <param name="...args"></param>
        /// <returns></returns>
        template<tpa::util::calculatable T, tpa::util::calculatable...Ts>
        [[nodiscard]] inline constexpr std::vector<std::pair<T,size_t>> mode(T a, Ts... args) noexcept
        {
            std::array<T, sizeof...(args) + 1> arr{};
            arr[0] = static_cast<T>(a);
            size_t x = 1uz;
            for (auto i : std::initializer_list<std::common_type_t<Ts...> >{ args... })
            {
                arr[x] = i;
                ++x;
            }//End for

            std::unordered_set<T> seen;

            size_t max_count = 0;
            std::vector<std::pair<T, size_t>> ret;

            for (auto i = arr.begin(); i != arr.end(); ++i)
            {
                if (*std::find(std::execution::par_unseq, seen.cbegin(), seen.cend(), *i) == *seen.cend())
                {
                    const size_t count = std::count(std::execution::par_unseq, i, arr.end(), *i);

                    if (count > max_count)
                    {
                        max_count = count;
                        ret = { {*i, max_count} };
                    }//End if
                    else if (count == max_count)
                    {
                        ret.emplace_back(*i, max_count);
                    }//End if
                    seen.insert(*i);
                }//End if
            }//End for

            return ret;
        }//End of mode

        /// <summary>
        /// <para>Compute and return the mode(s) of the passed parameters.</para>
        /// <para>Returns an std vector of std pair (T,size_t)  where the 'first' member is the mode(s) and 'second' member is the number of occurrences</para> 
        /// <para>Returns the modes in sorted order.</para>
        /// </summary>
        /// <typeparam name="CONTAINER_T"></typeparam>
        /// <typeparam name="T"></typeparam>
        /// <param name="arr"></param>
        /// <returns></returns>
        template<typename CONTAINER_T, typename T = CONTAINER_T::value_type>
        [[nodiscard]] inline constexpr std::vector<std::pair<T, size_t>> mode(CONTAINER_T& arr) requires tpa::util::contiguous_seqeunce<CONTAINER_T>
        {   
            std::unordered_set<T> seen;

            size_t max_count = 0;
            std::vector<std::pair<T, size_t>> ret;

            for (auto i = arr.begin(); i != arr.end(); ++i)
            {
                if (*std::find(std::execution::par_unseq, seen.cbegin(), seen.cend(), *i) == *seen.cend())
                {
                    const size_t count = std::count(std::execution::par_unseq, i, arr.end(), *i);

                    if (count > max_count)
                    {
                        max_count = count;
                        ret = { {*i, max_count} };
                    }//End if
                    else if (count == max_count)
                    {
                        ret.emplace_back(*i, max_count);
                    }//End if
                    seen.insert(*i);
                }//End if
            }//End for

            return ret;
        }//End of mode
	}//End of namespace
}//End of namespace