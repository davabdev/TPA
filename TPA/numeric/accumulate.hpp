#pragma once
/*
* Truly Parallel Algorithms Library - Numeric - accumulate function
* By: David Aaron Braun
* 2021-04-08
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
#include "../excepts.hpp"

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa
{
#pragma region generic
    /// <summary>
    /// <para>Computes the sum of the given value 'val' and the elements in the container.</para>
    /// <para>This implementation is Multi-Threaded Only. No SIMD.</para>
    /// <para>Explicitly requires that you specify a predicate such as std::plus&lt;T&gt;()</para>
    /// <para>This implementation is about as fast as std::reduce, but more reliable.</para>
    /// </summary>
    /// <typeparam name="CONTAINER_T"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="P"></typeparam>
    /// <param name="arr"> - The container you're summing</param>
    /// <param name="val"> - The initial value</param>
    /// <param name="pred"> - The predicate </param>
    /// <returns></returns>
    template<typename CONTAINER_T, typename T, class P>
    [[nodiscard]] inline constexpr T accumulate(
        const CONTAINER_T& arr, 
        const T val, 
        const P pred)
        requires tpa::util::contiguous_seqeunce<CONTAINER_T>
	{
        try
        {
            uint32_t complete = 0;

            T sum = val;

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

                    T temp_val = 0;

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
            return static_cast<T>(0);
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): " << ex.what() << "\n";
            return static_cast<T>(0);
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): " << ex.what() << "\n";
            return static_cast<T>(0);
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): unknown!\n";
            return static_cast<T>(0);
        }//End catch
	}//End of accumulate


    /// <summary>
    /// <para>Computes the sum of the given value 'val' and the elements in the container.</para>
    /// <para>This implementation uses SIMD and Multi-Threading.</para>
    /// <para>Valid Templated Predicates:</para>
    /// <para>tpa::eqt::SUM</para>
    /// <para>tpa::eqt::DIFFERENCE_</para>
    /// <para>tpa::eqt::PRODUCT</para>
    /// <para>tpa::eqt::QUOTIENT</para>
    /// <para>tpa::eqt::REMAINDER</para>
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="CONTAINER_T"></typeparam>
    /// <param name="arr"></param>
    /// <param name="val"></param>
    /// <returns></returns>
    template<tpa::eqt INSTR, typename CONTAINER_T, typename T = CONTAINER_T::value_type>
    [[nodiscard]] inline constexpr T accumulate(const CONTAINER_T& arr, const T val = 0)
    requires tpa::util::contiguous_seqeunce<CONTAINER_T>
    {
        try
        {
            uint32_t complete = 0;

            T sum = val;

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

                        T temp_val = 1;
#pragma region byte
                        if constexpr (std::is_same<CONTAINER_T::value_type, int8_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                __m256i _first, _second;
                                __m256i _sum;

                                for (; i != end; i += 64)
                                {

                                    if ((i + 64) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Load Values
                                    _first = _mm256_load_si256((__m256i*) & arr[i]);
                                    _second = _mm256_load_si256((__m256i*) & arr[i+32]);

                                    //Calc
                                    if constexpr (INSTR == tpa::eqt::SUM)
                                    {
                                        _sum = _mm256_add_epi8(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                                    {
                                        _sum = _mm256_sub_epi8(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::PRODUCT)
                                    {
                                        int8_t d[32] = {};
                                        for (size_t x = 0; x != 32; ++x)
                                        {
#ifdef _WIN32
                                            d[x] = static_cast<int8_t>(_first.m256i_i8[x] * _second.m256i_i8[x]);
#else
                                            d[x] = static_cast<int8_t>(_first[x] * _second[x]);
#endif
                                        }//End for
                                        _sum = _mm256_load_si256((__m256i*) & d);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                                    {
                                        _sum = _mm256_div_epi8(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::REMAINDER)
                                    {
                                        _sum = _mm256_rem_epi8(_first, _second);
                                    }//End if
                                    else
                                    {
                                        [] <bool flag = false>()
                                        {
                                            static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                                        }();
                                    }//End else

                                    //Store Result                                    
                                    for (size_t x = 0; x != 32; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256i_i8[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned byte
                        else if constexpr (std::is_same<CONTAINER_T::value_type, uint8_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                __m256i _first, _second;
                                __m256i _sum;

                                for (; i != end; i += 64)
                                {

                                    if ((i + 64) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Load Values
                                    _first = _mm256_load_si256((__m256i*) &arr[i]);
                                    _second = _mm256_load_si256((__m256i*) &arr[i+32]);

                                    //Calc
                                    if constexpr (INSTR == tpa::eqt::SUM)
                                    {
                                        _sum = _mm256_add_epi8(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                                    {
                                        _sum = _mm256_sub_epi8(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::PRODUCT)
                                    {
                                        uint8_t d[32] = {};
                                        for (size_t x = 0; x != 32; ++x)
                                        {
#ifdef _WIN32
                                            d[x] = static_cast<uint8_t>(_first.m256i_u8[x] * _second.m256i_u8[x]);
#else
                                            d[x] = static_cast<uint8_t>(_first[x] * _second[x]);
#endif
                                        }//End for
                                        _sum = _mm256_load_si256((__m256i*) & d);

                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                                    {
                                        _sum = _mm256_div_epu8(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::REMAINDER)
                                    {
                                        _sum = _mm256_rem_epu8(_first, _second);
                                    }//End if
                                    else
                                    {
                                        [] <bool flag = false>()
                                        {
                                            static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                                        }();
                                    }//End else

                                    //Store Result                                    
                                    for (size_t x = 0; x != 32; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256i_u8[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region short
                        else if constexpr (std::is_same<CONTAINER_T::value_type, int16_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                __m256i _first, _second;
                                __m256i _sum;

                                for (; i != end; i += 32)
                                {

                                    if ((i + 32) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Load Values
                                    _first = _mm256_load_si256((__m256i*) & arr[i]);
                                    _second = _mm256_load_si256((__m256i*) & arr[i + 16]);

                                    //Calc
                                    if constexpr (INSTR == tpa::eqt::SUM)
                                    {
                                        _sum = _mm256_add_epi16(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                                    {
                                        _sum = _mm256_sub_epi16(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::PRODUCT)
                                    {
                                        _sum = _mm256_mullo_epi16(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                                    {
                                        _sum = _mm256_div_epi16(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::REMAINDER)
                                    {
                                        _sum = _mm256_rem_epi16(_first, _second);
                                    }//End if
                                    else
                                    {
                                        [] <bool flag = false>()
                                        {
                                            static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                                        }();
                                    }//End else

                                    //Store Result                                    
                                    for (size_t x = 0; x != 16; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256i_i16[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned short
                        else if constexpr (std::is_same<CONTAINER_T::value_type, uint16_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                __m256i _first, _second;
                                __m256i _sum;

                                for (; i != end; i += 32)
                                {

                                    if ((i + 32) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Load Values
                                    _first = _mm256_load_si256((__m256i*) &arr[i]);
                                    _second = _mm256_load_si256((__m256i*) &arr[i+16]);

                                    //Calc
                                    if constexpr (INSTR == tpa::eqt::SUM)
                                    {
                                        _sum = _mm256_add_epi16(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                                    {
                                        _sum = _mm256_sub_epi16(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::PRODUCT)
                                    {
                                        _sum = _mm256_mullo_epi16(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                                    {
                                        _sum = _mm256_div_epu16(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::REMAINDER)
                                    {
                                        _sum = _mm256_rem_epu16(_first, _second);
                                    }//End if
                                    else
                                    {
                                        [] <bool flag = false>()
                                        {
                                            static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                                        }();
                                    }//End else

                                    //Store Result                                    
                                    for (size_t x = 0; x != 16; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256i_u16[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region int
                        else if constexpr (std::is_same<CONTAINER_T::value_type, int32_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                __m256i _first, _second;
                                __m256i _sum;

                                for (; i != end; i += 16)
                                {

                                    if ((i + 16) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Load Values
                                    _first = _mm256_load_si256((__m256i*) &arr[i]);
                                    _second = _mm256_load_si256((__m256i*) &arr[i+8]);

                                    //Calc
                                    if constexpr (INSTR == tpa::eqt::SUM)
                                    {
                                        _sum = _mm256_add_epi32(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                                    {
                                        _sum = _mm256_sub_epi32(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::PRODUCT)
                                    {
                                        _sum = _mm256_mullo_epi32(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                                    {
                                        _sum = _mm256_div_epi32(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::REMAINDER)
                                    {
                                        _sum = _mm256_rem_epi32(_first, _second);
                                    }//End if
                                    else
                                    {
                                        [] <bool flag = false>()
                                        {
                                            static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                                        }();
                                    }//End else

                                    //Store Result                                    
                                    for (size_t x = 0; x != 8; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256i_i32[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned int
                        else if constexpr (std::is_same<CONTAINER_T::value_type, uint32_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                __m256i _first, _second;
                                __m256i _sum;

                                for (; i != end; i += 16)
                                {

                                    if ((i + 16) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Load Values
                                    _first = _mm256_load_si256((__m256i*) & arr[i]);
                                    _second = _mm256_load_si256((__m256i*) & arr[i + 8]);

                                    //Calc
                                    if constexpr (INSTR == tpa::eqt::SUM)
                                    {
                                        _sum = _mm256_add_epi32(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                                    {
                                        _sum = _mm256_sub_epi32(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::PRODUCT)
                                    {
                                        _sum = _mm256_mullo_epi32(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                                    {
                                        _sum = _mm256_div_epu32(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::REMAINDER)
                                    {
                                        _sum = _mm256_rem_epu32(_first, _second);
                                    }//End if
                                    else
                                    {
                                        [] <bool flag = false>()
                                        {
                                            static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                                        }();
                                    }//End else

                                    //Store Result                                    
                                    for (size_t x = 0; x != 8; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256i_u32[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region long
                        else if constexpr (std::is_same<CONTAINER_T::value_type, int64_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                __m256i _first, _second;
                                __m256i _sum;

                                for (; i != end; i += 8)
                                {

                                    if ((i + 8) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Load Values
                                    _first = _mm256_load_si256((__m256i*) & arr[i]);
                                    _second = _mm256_load_si256((__m256i*) & arr[i + 4]);

                                    //Calc
                                    if constexpr (INSTR == tpa::eqt::SUM)
                                    {
                                        _sum = _mm256_add_epi64(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                                    {
                                        _sum = _mm256_sub_epi64(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::PRODUCT)
                                    {
                                        uint64_t d[4] = {};
                                        for (size_t x = 0; x != 4; ++x)
                                        {
#ifdef _WIN32
                                            d[x] = _first.m256i_i64[x] * _second.m256i_i64[x];
#else
                                            d[x] = _first[x] * _second[x];
#endif
                                        }//End for
                                        _sum = _mm256_load_si256((__m256i*) &d);

                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                                    {
                                        _sum = _mm256_div_epi64(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::REMAINDER)
                                    {
                                        _sum = _mm256_rem_epi64(_first, _second);
                                    }//End if
                                    else
                                    {
                                        [] <bool flag = false>()
                                        {
                                            static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                                        }();
                                    }//End else

                                    //Store Result                                    
                                    for (size_t x = 0; x != 4; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256i_i64[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region unsigned long
                        else if constexpr (std::is_same<CONTAINER_T::value_type, uint64_t>() == true)
                        {
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                __m256i _first, _second;
                                __m256i _sum;

                                for (; i != end; i += 8)
                                {

                                    if ((i + 8) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Load Values
                                    _first = _mm256_load_si256((__m256i*) &arr[i]);
                                    _second = _mm256_load_si256((__m256i*) &arr[i+4]);

                                    //Calc
                                    if constexpr (INSTR == tpa::eqt::SUM)
                                    {
                                        _sum = _mm256_add_epi64(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                                    {
                                        _sum = _mm256_sub_epi64(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::PRODUCT)
                                    {
                                        uint64_t d[4] = {};
                                        for (size_t x = 0; x != 4; ++x)
                                        {
#ifdef _WIN32
                                            d[x] = _first.m256i_u64[x] * _second.m256i_u64[x];
#else
                                            d[x] = _first[x] * _second[x];
#endif
                                        }//End for
                                        _sum = _mm256_load_si256((__m256i*)& d);

                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                                    {
                                        _sum = _mm256_div_epu64(_first, _second);
                                    }//End if
                                    else if constexpr (INSTR == tpa::eqt::REMAINDER)
                                    {
                                        _sum = _mm256_rem_epu64(_first, _second);
                                    }//End if
                                    else
                                    {
                                        [] <bool flag = false>()
                                        {
                                            static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                                        }();
                                    }//End else

                                    //Store Result                                    
                                    for (size_t x = 0; x != 4; ++x)
                                    {
#ifdef _WIN32
                                        temp_val += _sum.m256i_u64[x];
#else
                                        temp_val += _sum[x];
#endif
                                    }//End for
                                }//End for
                            }//End if hasAVX2
#endif
                        }//End if
#pragma endregion
#pragma region float
                    else if constexpr (std::is_same<CONTAINER_T::value_type, float>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX)
                        {
                            __m256 _first, _second;
                            __m256 _sum;

                            for (; i != end; i += 16)
                            {

                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Values
                                _first = _mm256_load_ps(&arr[i]);
                                _second = _mm256_load_ps(&arr[i+8]);

                                //Calc
                                if constexpr (INSTR == tpa::eqt::SUM)
                                {
                                    _sum = _mm256_add_ps(_first, _second);
                                }//End if
                                else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                                {
                                    _sum = _mm256_sub_ps(_first, _second);
                                }//End if
                                else if constexpr (INSTR == tpa::eqt::PRODUCT)
                                {
                                    _sum = _mm256_mul_ps(_first, _second);
                                }//End if
                                else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                                {
                                    _sum = _mm256_div_ps(_first, _second);
                                }//End if
                                else if constexpr (INSTR == tpa::eqt::REMAINDER)
                                {
                                    break;//Use std::fmod
                                }//End if
                                else
                                {
                                    [] <bool flag = false>()
                                    {
                                        static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                                    }();
                                }//End else
  
                                //Store Result                                    
                                for (size_t x = 0; x != 8; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m256_f32[x];
#else
                                    temp_val += _sum[x];
#endif
                                }//End for
                            }//End for
                        }//End if hasAVX
#endif
                    }//End if
#pragma endregion
#pragma region double
                    else if constexpr (std::is_same<CONTAINER_T::value_type, double>() == true)
                    {
#ifdef _M_AMD64
                        if (tpa::hasAVX)
                        {
                            __m256d _first, _second;
                            __m256d _sum;

                            for (; i != end; i += 8)
                            {

                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Load Values
                                _first = _mm256_load_pd(&arr[i]);
                                _second = _mm256_load_pd(&arr[i + 4]);

                                //Calc
                                if constexpr (INSTR == tpa::eqt::SUM)
                                {
                                    _sum = _mm256_add_pd(_first, _second);
                                }//End if
                                else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                                {
                                    _sum = _mm256_sub_pd(_first, _second);
                                }//End if
                                else if constexpr (INSTR == tpa::eqt::PRODUCT)
                                {
                                    _sum = _mm256_mul_pd(_first, _second);
                                }//End if
                                else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                                {
                                    _sum = _mm256_div_pd(_first, _second);
                                }//End if
                                else if constexpr (INSTR == tpa::eqt::REMAINDER)
                                {
                                    break;//Use std::fmod
                                }//End if
                                else
                                {
                                    [] <bool flag = false>()
                                    {
                                        static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                                    }();
                                }//End else

                                //Store Result                                    
                                for (size_t x = 0; x != 4; ++x)
                                {
#ifdef _WIN32
                                    temp_val += _sum.m256d_f64[x];
#else
                                    temp_val += _sum[x];
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
                        //Calc
                        if constexpr (INSTR == tpa::eqt::SUM)
                        {
                            temp_val += arr[i];
                        }//End if
                        else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                        {
                            temp_val -= arr[i];
                        }//End if
                        else if constexpr (INSTR == tpa::eqt::PRODUCT)
                        {
                            temp_val *= arr[i];
                        }//End if
                        else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                        {
                            temp_val /= arr[i];
                        }//End if
                        else if constexpr (INSTR == tpa::eqt::REMAINDER)
                        {
                            if constexpr (std::is_floating_point<CONTAINER_T::value_type>())
                            {
                                temp_val = std::fmod(temp_val, arr[i]);
                            }//End if
                            else
                            {
                                temp_val %= arr[i];
                            }//End else
                        }//End if
                        else
                        {
                            [] <bool flag = false>()
                            {
                                static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                            }();
                        }//End else
                    }//End for                   
#pragma endregion
                        return temp_val;
                });//End of lambda

                results.emplace_back(std::move(temp));
            }//End for 

            for (const auto& fut : results)
            {
                if constexpr (INSTR == tpa::eqt::SUM)
                {
                    sum += fut.get();
                    sum -= 1;//Clean up temp_val
                }//End if
                else if constexpr (INSTR == tpa::eqt::DIFFERENCE_)
                {
                    sum += fut.get();
                    sum -= 1;//Clean up temp_val
                }//End if
                else if constexpr (INSTR == tpa::eqt::PRODUCT)
                {
                    sum *= fut.get();
                }//End if
                else if constexpr (INSTR == tpa::eqt::QUOTIENT)
                {
                    sum /= fut.get();
                }//End if
                else if constexpr (INSTR == tpa::eqt::REMAINDER)
                {
                    if constexpr (std::is_floating_point<CONTAINER_T::value_type>())
                    {
                        sum = std::fmod(sum, fut.get());
                    }//End if
                    else
                    {
                        sum %= fut.get();
                    }//End else
                }//End if
                else
                {
                    [] <bool flag = false>()
                    {
                        static_assert(flag, " You have specifed an invalid equation in tpa::accumulate<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
                    }();
                }//End else

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
            return static_cast<T>(0);
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): " << ex.what() << "\n";
            return static_cast<T>(0);
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): " << ex.what() << "\n";
            return static_cast<T>(0);
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::accumulate(): unknown!\n";
            return static_cast<T>(0);
        }//End catch
    }//End of accumulate

    /// <summary>
    /// <para>Default simplified version</para>
    /// <para>Default predicate tpa::eqt::SUM</para>
    /// <para>Default value is '0' and the default return type is the value_type of the container</para>
    /// </summary>
    /// <typeparam name="CONTAINER_T"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    template<typename CONT_T, typename T = CONT_T::value_type>
    [[nodiscard]] inline constexpr T accumulate(const CONT_T& arr, const T val = 0)
    {
        return tpa::accumulate<tpa::eqt::SUM>(arr, val);
    }//End of accumulate
#pragma endregion
}//End of namespace
