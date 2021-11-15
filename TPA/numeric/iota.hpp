#pragma once
/*
* Truly Parallel Algorithms Library - Numeric - iota function
* By: David Aaron Braun
* 2021-01-24
* Parallel implementation of iota
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
#include <array>
#include <vector>

#ifdef _M_AMD64
#include <immintrin.h>
#elif defined (_M_ARM64)
#ifdef _WIN32
#include "arm64_neon.h"
#else
#include "arm_neon.h"
#endif
#endif

#include "../tpa.hpp"
#include "../ThreadPool.hpp"
#include "../_util.hpp"

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa 
{
#pragma region generic
    /// <summary>
    /// Fills the range [first, last) with sequentially increasing values, starting with specified value and repetitively evaluating ++value.
    /// </summary>
    /// <typeparam name="CONTAINER_T"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <param name="CONTAINER_T"></param>
    /// <param name="value"></param>
    template <typename CONTAINER, typename T = CONTAINER::value_type>
    inline constexpr void iota(CONTAINER& arr, const T value = 0)
    requires tpa::util::contiguous_seqeunce<CONTAINER>
    {
        try
        {
            static_assert(std::is_same<CONTAINER::value_type, T>() == true, "Compile Error! The container must be of the same value type as the initial value type!");

            uint32_t complete = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, arr.size());

            std::vector<std::shared_future<uint32_t>> results;
            results.reserve(tpa::nThreads);

            std::shared_future<uint32_t> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&arr, &value, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;
                        T val;

#pragma region byte
                        if constexpr (std::is_same<T, int8_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {          
                                const __m256i _adder =
                                    _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

                                __m256i _Val, _Res;

                                for (; i != end; i += 32)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 32) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi8(static_cast<int8_t>(val));

                                    _Res = _mm256_add_epi8(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned byte
                        else if constexpr (std::is_same<T, uint8_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder =
                                    _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

                                __m256i _Val, _Res;

                                for (; i != end; i += 32)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 32) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi8(static_cast<uint8_t>(val));

                                    _Res = _mm256_add_epi8(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region short
                        else if constexpr (std::is_same<T, int16_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder =
                                    _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

                                __m256i _Val, _Res;

                                for (; i != end; i += 16)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 16) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi16(static_cast<int16_t>(val));

                                    _Res = _mm256_add_epi16(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned short
                        else if constexpr (std::is_same<T, uint16_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder =
                                    _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

                                __m256i _Val, _Res;

                                for (; i != end; i += 16)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 16) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi16(static_cast<uint16_t>(val));

                                    _Res = _mm256_add_epi16(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region int
                        else if constexpr (std::is_same<T, int32_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder =
                                    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

                                __m256i _Val, _Res;

                                for (; i != end; i += 8)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 8) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi32(static_cast<int32_t>(val));

                                    _Res = _mm256_add_epi32(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned int
                        else if constexpr (std::is_same<T, uint32_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder =
                                    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

                                __m256i _Val, _Res;

                                for (; i != end; i += 8)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 8) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi32(static_cast<uint32_t>(val));

                                    _Res = _mm256_add_epi32(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region long
                        else if constexpr (std::is_same<T, int64_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder = _mm256_setr_epi64x(0, 1, 2, 3);

                                __m256i _Val, _Res;

                                for (; i != end; i += 4)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi64x(static_cast<int64_t>(val));

                                    _Res = _mm256_add_epi64(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if has AVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned long
                        else if constexpr (std::is_same<T, uint64_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder = _mm256_setr_epi64x(0, 1, 2, 3);

                                __m256i _Val, _Res;

                                for (; i != end; i += 4)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi64x(static_cast<uint64_t>(val));

                                    _Res = _mm256_add_epi64(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region float
                        else if constexpr (std::is_same<T, float>() == true)
                        {
#ifdef _M_AMD64
                        if (tpa::hasAVX)
                        {
                            const __m256 _adder = _mm256_setr_ps(
                                0.f,
                                1.f,
                                2.f,
                                3.f,
                                4.f,
                                5.f,
                                6.f,
                                7.f);

                            __m256 _Val, _Res;

                            for (; i != end; i += 8)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_ps(static_cast<float>(val));

                                _Res = _mm256_add_ps(_Val, _adder);

                                _mm256_store_ps(&arr[i], _Res);
                            }//End for
                        }//Endf if hasAVX
#endif
                        }//End if
#pragma endregion
#pragma region double 
                        else if constexpr (std::is_same<T, double>() == true)
                        {
#ifdef _M_AMD64
                        if (tpa::hasAVX)
                        {
                            const __m256d _adder = _mm256_setr_pd(
                                0.0,
                                1.0,
                                2.0,
                                3.0);

                            __m256d _Val, _Res;

                            for (; i != end; i += 4)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 4) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_pd(static_cast<double> (val));

                                _Res = _mm256_add_pd(_Val, _adder);

                                _mm256_store_pd(&arr[i], _Res);
                            }//End for
                        }//End if hasAVX
#endif
                    }//End if
#pragma endregion
#pragma region generic

                        for (; i != end; ++i)
                        {
                            val = static_cast<T>(value + i);
                            arr[i] = val;
                            ++val;
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
            if (complete != tpa::nThreads)
            {
                throw tpa::exceptions::NotAllThreadsCompleted(complete);
            }//End if
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): " << ex.code()
                << " " << ex.what() << "\n";
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): " << ex.what() << "\n";
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): " << ex.what() << "\n";
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): unknown!\n";
        }//End catch
    }//End of iota
#pragma endregion
#pragma region array
    
    /// <summary>
    /// Fills the range [first, last) with sequentially increasing values, starting with specified value and repetitively evaluating ++value.
    /// </summary>
    /// <typeparam name="CONTAINER_T"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <param name="CONTAINER_T"></param>
    /// <param name="value"></param>
    template <size_t SIZE, typename T>
    inline constexpr void iota(std::array<T,SIZE>& arr, const T value = 0)
    {
        try
        {             
            uint32_t complete = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, arr.size());

            std::vector<std::shared_future<uint32_t>> results;
            results.reserve(tpa::nThreads);

            std::shared_future<uint32_t> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&arr, &value, &sec]()
                {
                    const size_t beg = sec.first;
                    const size_t end = sec.second;
                    size_t i = beg;
                    T val;

#pragma region byte
                    if constexpr (std::is_same<T, int8_t>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            const __m256i _adder =
                                _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

                            __m256i _Val, _Res;

                            for (; i != end; i += 32)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_epi8(static_cast<int8_t>(val));

                                _Res = _mm256_add_epi8(_Val, _adder);

                                _mm256_store_si256((__m256i*) & arr[i], _Res);
                            }//End for
                        }//End if hasAVX2
#endif
                    }//End if
#pragma endregion
#pragma region unsigned byte
                    else if constexpr (std::is_same<T, uint8_t>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            const __m256i _adder =
                                _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

                            __m256i _Val, _Res;

                            for (; i != end; i += 32)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_epi8(static_cast<uint8_t>(val));

                                _Res = _mm256_add_epi8(_Val, _adder);

                                _mm256_store_si256((__m256i*) & arr[i], _Res);
                            }//End for
                        }//End if hasAVX2
#endif
                    }//End if
#pragma endregion
#pragma region short
                    else if constexpr (std::is_same<T, int16_t>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            const __m256i _adder =
                                _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

                            __m256i _Val, _Res;

                            for (; i != end; i += 16)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_epi16(static_cast<int16_t>(val));

                                _Res = _mm256_add_epi16(_Val, _adder);

                                _mm256_store_si256((__m256i*) & arr[i], _Res);
                            }//End for
                        }//End if hasAVX2
#endif
                    }//End if
#pragma endregion
#pragma region unsigned short
                    else if constexpr (std::is_same<T, uint16_t>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            const __m256i _adder =
                                _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

                            __m256i _Val, _Res;

                            for (; i != end; i += 16)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_epi16(static_cast<uint16_t>(val));

                                _Res = _mm256_add_epi16(_Val, _adder);

                                _mm256_store_si256((__m256i*) & arr[i], _Res);
                            }//End for
                        }//End if hasAVX2
#endif
                    }//End if
#pragma endregion
#pragma region int
                    else if constexpr (std::is_same<T, int32_t>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            const __m256i _adder =
                                _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

                            __m256i _Val, _Res;

                            for (; i != end; i += 8)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_epi32(static_cast<int32_t>(val));

                                _Res = _mm256_add_epi32(_Val, _adder);

                                _mm256_store_si256((__m256i*) & arr[i], _Res);
                            }//End for
                        }//End if hasAVX2
#endif
                    }//End if
#pragma endregion
#pragma region unsigned int
                    else if constexpr (std::is_same<T, uint32_t>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            const __m256i _adder =
                                _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

                            __m256i _Val, _Res;

                            for (; i != end; i += 8)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_epi32(static_cast<uint32_t>(val));

                                _Res = _mm256_add_epi32(_Val, _adder);

                                _mm256_store_si256((__m256i*) & arr[i], _Res);
                            }//End for
                        }//End if hasAVX2
#endif
                    }//End if
#pragma endregion
#pragma region long
                    else if constexpr (std::is_same<T, int64_t>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            const __m256i _adder = _mm256_setr_epi64x(0, 1, 2, 3);

                            __m256i _Val, _Res;

                            for (; i != end; i += 4)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 4) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_epi64x(static_cast<int64_t>(val));

                                _Res = _mm256_add_epi64(_Val, _adder);

                                _mm256_store_si256((__m256i*) & arr[i], _Res);
                            }//End for
                        }//End if has AVX2
#endif
                    }//End if
#pragma endregion
#pragma region unsigned long
                    else if constexpr (std::is_same<T, uint64_t>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            const __m256i _adder = _mm256_setr_epi64x(0, 1, 2, 3);

                            __m256i _Val, _Res;

                            for (; i != end; i += 4)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 4) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_epi64x(static_cast<uint64_t>(val));

                                _Res = _mm256_add_epi64(_Val, _adder);

                                _mm256_store_si256((__m256i*) & arr[i], _Res);
                            }//End for
                        }//End if hasAVX2
#endif
                    }//End if
#pragma endregion
#pragma region float
                    else if constexpr (std::is_same<T, float>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX)
                        {
                            const __m256 _adder = _mm256_setr_ps(
                                0.f,
                                1.f,
                                2.f,
                                3.f,
                                4.f,
                                5.f,
                                6.f,
                                7.f);

                            __m256 _Val, _Res;

                            for (; i != end; i += 8)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_ps(static_cast<float>(val));

                                _Res = _mm256_add_ps(_Val, _adder);

                                _mm256_store_ps(&arr[i], _Res);
                            }//End for
                        }//Endf if hasAVX
#endif
                    }//End if
#pragma endregion
#pragma region double 
                    else if constexpr (std::is_same<T, double>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX)
                        {
                            const __m256d _adder = _mm256_setr_pd(
                                0.0,
                                1.0,
                                2.0,
                                3.0);

                            __m256d _Val, _Res;

                            for (; i != end; i += 4)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 4) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_pd(static_cast<double> (val));

                                _Res = _mm256_add_pd(_Val, _adder);

                                _mm256_store_pd(&arr[i], _Res);
                            }//End for
                        }//End if hasAVX
#endif
                    }//End if
#pragma endregion
#pragma region generic

                    for (; i != end; ++i)
                    {
                        val = static_cast<T>(value + i);
                        arr[i] = val;
                        ++val;
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
            if (complete != tpa::nThreads)
            {
                throw tpa::exceptions::NotAllThreadsCompleted(complete);
            }//End if
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): " << ex.code()
                << " " << ex.what() << "\n";
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): " << ex.what() << "\n";
        }//End catch
        catch (const std::exception& ex)
        {
            std::unique_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): " << ex.what() << "\n";
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): unknown!\n";
        }//End catch
    }//End of iota
#pragma endregion
#pragma region vector
    /// <summary>
    /// Fills the vector with sequentially increasing values, starting with specified value and repetitively evaluating ++value.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="arr"></param>
    /// <param name="value"></param>
    template <typename T>
    inline constexpr void iota(std::vector<T>& arr, const T value = 0)
    {
        try
        {
            uint32_t complete = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, arr.size());

            std::vector<std::shared_future<uint32_t>> results;
            results.reserve(tpa::nThreads);

            std::shared_future<uint32_t> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&arr, &value, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;
                        T val;
#pragma region byte
                    if constexpr (std::is_same<T, int8_t>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX2)
                        {
                            const __m256i _adder =
                                _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

                            __m256i _Val, _Res;

                            for (; i != end; i += 32)
                            {
                                val = static_cast<T>(value + i);

                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _Val = _mm256_set1_epi8(static_cast<int8_t>(val));

                                _Res = _mm256_add_epi8(_Val, _adder);

                                _mm256_store_si256((__m256i*) & arr[i], _Res);
                            }//End for
                        }//End if hasAVX2
#endif
                    }//End if
#pragma endregion
#pragma region unsigned byte
                        else if constexpr (std::is_same<T, uint8_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder =
                                    _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

                                __m256i _Val, _Res;

                                for (; i != end; i += 32)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 32) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi8(static_cast<uint8_t>(val));

                                    _Res = _mm256_add_epi8(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region short
                        else if constexpr (std::is_same<T, int16_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder =
                                    _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

                                __m256i _Val, _Res;

                                for (; i != end; i += 16)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 16) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi16(static_cast<int16_t>(val));

                                    _Res = _mm256_add_epi16(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned short
                        else if constexpr (std::is_same<T, uint16_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder =
                                    _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

                                __m256i _Val, _Res;

                                for (; i != end; i += 16)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 16) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi16(static_cast<uint16_t>(val));

                                    _Res = _mm256_add_epi16(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region int
                        else if constexpr (std::is_same<T, int32_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder =
                                    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

                                __m256i _Val, _Res;

                                for (; i != end; i += 8)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 8) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi32(static_cast<int32_t>(val));

                                    _Res = _mm256_add_epi32(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned int
                        else if constexpr (std::is_same<T, uint32_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder =
                                    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

                                __m256i _Val, _Res;

                                for (; i != end; i += 8)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 8) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi32(static_cast<uint32_t>(val));

                                    _Res = _mm256_add_epi32(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region long
                        else if constexpr (std::is_same<T, int64_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder = _mm256_setr_epi64x(0, 1, 2, 3);

                                __m256i _Val, _Res;

                                for (; i != end; i += 4)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi64x(static_cast<int64_t>(val));

                                    _Res = _mm256_add_epi64(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if has AVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned long
                        else if constexpr (std::is_same<T, uint64_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _adder = _mm256_setr_epi64x(0, 1, 2, 3);

                                __m256i _Val, _Res;

                                for (; i != end; i += 4)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_epi64x(static_cast<uint64_t>(val));

                                    _Res = _mm256_add_epi64(_Val, _adder);

                                    _mm256_store_si256((__m256i*) & arr[i], _Res);
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region float
                        else if constexpr (std::is_same<T, float>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX)
                            {
                                const __m256 _adder = _mm256_setr_ps(
                                    0.f,
                                    1.f,
                                    2.f,
                                    3.f,
                                    4.f,
                                    5.f,
                                    6.f,
                                    7.f);

                                __m256 _Val, _Res;

                                for (; i != end; i += 8)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 8) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_ps(static_cast<float>(val));

                                    _Res = _mm256_add_ps(_Val, _adder);

                                    _mm256_store_ps(&arr[i], _Res);
                                }//End for
                            }//Endf if hasAVX
#endif
                        }//End if
#pragma endregion
#pragma region double 
                        else if constexpr (std::is_same<T, double>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX)
                            {
                                const __m256d _adder = _mm256_setr_pd(
                                    0.0,
                                    1.0,
                                    2.0,
                                    3.0);

                                __m256d _Val, _Res;

                                for (; i != end; i += 4)
                                {
                                    val = static_cast<T>(value + i);

                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    _Val = _mm256_set1_pd(static_cast<double> (val));

                                    _Res = _mm256_add_pd(_Val, _adder);

                                    _mm256_store_pd(&arr[i], _Res);
                                }//End for
                            }//End if hasAVX
#endif
                        }//End if
#pragma endregion
#pragma region generic

                        for (; i != end; ++i)
                        {
                            val = static_cast<T>(value + i);
                            arr[i] = val;
                            ++val;
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
            if (complete != tpa::nThreads)
            {
                throw tpa::exceptions::NotAllThreadsCompleted(complete);
            }//End if
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): " << ex.code()
                << " " << ex.what() << "\n";
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): " << ex.what() << "\n";
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): " << ex.what() << "\n";
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::iota(): unknown!\n";
        }//End catch
    }//End of iota
#pragma endregion

}//End of namespace