#pragma once
/*
* Truly Parallel Algorithms Library - Algorithm - copy function
* By: David Aaron Braun
* 2021-05-12
* Parallel implementation of copy
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
namespace tpa {
#pragma region generic
    
    /// <summary>
    /// <para>Copies the items in the source container to the destination container</para>
    /// <para>Containers of different types are allowed</para>
    /// <para>Containters of different value types are NOT allowed</para>
    /// <para>If the item count is not specifed it will copy as many items into the destination container as will fit</para>
    /// </summary>
    /// <typeparam name="SOURCE"></typeparam>
    /// <typeparam name="DEST"></typeparam>
    /// <param name="source"></param>
    /// <param name="dest"></param>
    /// <param name="item_count"></param>
    template <typename SOURCE, typename DEST>
    inline constexpr void copy(const SOURCE& source, DEST& dest, size_t item_count = 0)
    requires tpa::util::contiguous_seqeunce<SOURCE> && tpa::util::contiguous_seqeunce<DEST>
    {
        try
        {
            static_assert(std::is_same<SOURCE::value_type, DEST::value_type>() == true, "Compile Error! The source and destination container must be of the same value type!");

            using T = SOURCE::value_type;

            //Prevent overflow
            if (item_count <= 0 || item_count > source.size()  || item_count > dest.size())
            {
                item_count = dest.size();
            }//End if

            uint32_t complete = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, item_count);

            std::vector<std::shared_future<uint32_t>> results;
            results.reserve(nThreads);

            std::shared_future<uint32_t> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&source, &dest, &item_count, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

#pragma region byte
                        if constexpr (std::is_same<T, int8_t>() == true)
                        {
                            /*
                            for (; i < end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int8_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned byte
                        else if constexpr (std::is_same<T, uint8_t>() == true)
                        {
                            /*
                            for (; i < end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint8_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region short
                        else if constexpr (std::is_same<T, int16_t>() == true)
                        {
                            /*
                            for (; i < end; i += 16)
                            {
                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int16_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned short
                        else if constexpr (std::is_same<T, uint16_t>() == true)
                        {
                            /*
                            for (; i < end; i += 16)
                            {
                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint16_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region int
                        else if constexpr (std::is_same<T, int32_t>() == true)
                        {
                            /*
                            for (; i < end; i += 8)
                            {
                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int32_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned int
                        else if constexpr (std::is_same<T, uint32_t>() == true)
                        {
                            /*
                            for (; i < end; i += 8)
                            {
                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint32_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region long
                        else if constexpr (std::is_same<T, int64_t>() == true)
                        {
                            /*
                                for (; i < end; i += 4)
                                {
                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Copy with avx2
                                    _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                                }//End for
                                */
                                //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int64_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned long
                        else if constexpr (std::is_same<T, uint64_t>() == true)
                        {
                            /*
                                for (; i < end; i += 4)
                                {
                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Copy with avx2
                                    _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                                }//End for
                                */
                                //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint64_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region float
                        else if constexpr (std::is_same<T, float>() == true)
                        {
                            /*
                            for (; i < end; i += 8)
                            {
                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_ps(&dest[i], _mm256_load_ps(&source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(float) * (end - i));
                        }//End if
#pragma endregion
#pragma region double 
                        else if constexpr (std::is_same<T, double>() == true)
                        {
                            /*
                                for (; i < end; i += 4)
                                {
                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Copy with avx2
                                    _mm256_store_pd(&dest[i], _mm256_load_pd(&source[i]));
                                }//End for
                                */
                                //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(double) * (end - i));
                        }//End if
#pragma endregion
#pragma region generic
                        else
                        {
                            std::memmove(&dest[i], &source[i], sizeof(T) * (end - i));
                        }//End else
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
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): " << ex.code()
                << " " << ex.what() << "\n";
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): " << ex.what() << "\n";
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): " << ex.what() << "\n";
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): unknown!\n";
        }//End catch
    }//End of copy
#pragma endregion
#pragma region array
        /// <summary>
    /// <para>Copies the items in the source array to the destination array</para>
    /// <para>Containers of different types are allowed</para>
    /// <para>Arrays of different value types are NOT allowed</para>
    /// <para>If the item count is not specifed it will copy as many items into the destination Array as will fit</para>
    /// </summary>
    /// <typeparam name="SOURCE"></typeparam>
    /// <typeparam name="DEST"></typeparam>
    /// <param name="source"></param>
    /// <param name="dest"></param>
    /// <param name="item_count"></param>
    template <typename T, size_t SIZE1, size_t SIZE2>
    inline constexpr void copy(const std::array<T,SIZE1>& source, std::array<T,SIZE2>& dest, size_t item_count = 0)
    {
        try
        {
            //Prevent overflow
            if (item_count <= 0 || item_count > source.size() || item_count > dest.size())
            {
                item_count = dest.size();
            }//End if

            uint32_t complete = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, item_count);

            std::vector<std::shared_future<uint32_t>> results;
            results.reserve(nThreads);

            std::shared_future<uint32_t> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&source, &dest, &item_count, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

#pragma region byte
                        if constexpr (std::is_same<T, int8_t>() == true)
                        {
                            /*
                            for (; i < end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int8_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned byte
                        else if constexpr (std::is_same<T, uint8_t>() == true)
                        {
                            /*
                            for (; i < end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint8_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region short
                        else if constexpr (std::is_same<T, int16_t>() == true)
                        {
                            /*
                            for (; i < end; i += 16)
                            {
                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int16_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned short
                        else if constexpr (std::is_same<T, uint16_t>() == true)
                        {
                            /*
                            for (; i < end; i += 16)
                            {
                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint16_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region int
                        else if constexpr (std::is_same<T, int32_t>() == true)
                        {
                            /*
                            for (; i < end; i += 8)
                            {
                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int32_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned int
                        else if constexpr (std::is_same<T, uint32_t>() == true)
                        {
                            /*
                            for (; i < end; i += 8)
                            {
                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint32_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region long
                        else if constexpr (std::is_same<T, int64_t>() == true)
                        {
                            /*
                                for (; i < end; i += 4)
                                {
                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Copy with avx2
                                    _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                                }//End for
                                */
                                //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int64_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned long
                        else if constexpr (std::is_same<T, uint64_t>() == true)
                        {
                            /*
                                for (; i < end; i += 4)
                                {
                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Copy with avx2
                                    _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                                }//End for
                                */
                                //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint64_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region float
                        else if constexpr (std::is_same<T, float>() == true)
                        {
                            /*
                            for (; i < end; i += 8)
                            {
                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_ps(&dest[i], _mm256_load_ps(&source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(float) * (end - i));
                        }//End if
#pragma endregion
#pragma region double 
                        else if constexpr (std::is_same<T, double>() == true)
                        {
                            /*
                                for (; i < end; i += 4)
                                {
                                    if ((i + 4) > end) [[unlikely]]
                                    {
                                        break;
                                    }//End if

                                    //Copy with avx2
                                    _mm256_store_pd(&dest[i], _mm256_load_pd(&source[i]));
                                }//End for
                                */
                                //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(double) * (end - i));
                        }//End if
#pragma endregion
#pragma region generic
                        else
                        {
                            std::memmove(&dest[i], &source[i], sizeof(T) * (end - i));
                        }//End else
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
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): " << ex.code()
                << " " << ex.what() << "\n";
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): " << ex.what() << "\n";
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): " << ex.what() << "\n";
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): unknown!\n";
        }//End catch
    }//End of copy
#pragma endregion
#pragma region vector
         /// <summary>
    /// <para>Copies the items in the source vector to the destination vector</para>
    /// <para>Containers of different types are allowed</para>
    /// <para>Vectors of different value types are NOT allowed</para>
    /// <para>If the item count is not specifed it will copy as many items into the destination Vector as will fit</para>
    /// </summary>
    /// <typeparam name="SOURCE"></typeparam>
    /// <typeparam name="DEST"></typeparam>
    /// <param name="source"></param>
    /// <param name="dest"></param>
    /// <param name="item_count"></param>
    template <typename T>
    inline constexpr void copy(const std::vector<T>& source, std::vector<T>& dest, size_t item_count = 0)
    {
        try
        {
            //Prevent overflow
            if (item_count <= 0 || item_count > source.size() || item_count > dest.size())
            {
                item_count = dest.size();
            }//End if

            uint32_t complete = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, item_count);

            std::vector<std::shared_future<uint32_t>> results;
            results.reserve(nThreads);

            std::shared_future<uint32_t> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&source, &dest, &item_count, &sec]()
                    {
                    
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

#pragma region byte
                        if constexpr (std::is_same<T, int8_t>() == true)
                        {
                            /*
                            for (; i < end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int8_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned byte
                        else if constexpr (std::is_same<T, uint8_t>() == true)
                        {
                            /*
                            for (; i < end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint8_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region short
                        else if constexpr (std::is_same<T, int16_t>() == true)
                        {
                            /*
                            for (; i < end; i += 16)
                            {
                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int16_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned short
                        else if constexpr (std::is_same<T, uint16_t>() == true)
                        {
                            /*
                            for (; i < end; i += 16)
                            {
                                if ((i + 16) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint16_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region int
                        else if constexpr (std::is_same<T, int32_t>() == true)
                        {
                            /*
                            for (; i < end; i += 8)
                            {
                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int32_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned int
                        else if constexpr (std::is_same<T, uint32_t>() == true)
                        {
                            /*
                            for (; i < end; i += 8)
                            {
                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint32_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region long
                        else if constexpr (std::is_same<T, int64_t>() == true)
                        {
                        /*
                            for (; i < end; i += 4)
                            {
                                if ((i + 4) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(int64_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region unsigned long
                        else if constexpr (std::is_same<T, uint64_t>() == true)
                        {
                        /*
                            for (; i < end; i += 4)
                            {
                                if ((i + 4) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_si256((__m256i*) & dest[i], _mm256_load_si256((__m256i const*) & source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(uint64_t) * (end - i));
                        }//End if
#pragma endregion
#pragma region float
                        else if constexpr (std::is_same<T, float>() == true)
                        {
                            /*
                            for (; i < end; i += 8)
                            {
                                if ((i + 8) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_ps(&dest[i], _mm256_load_ps(&source[i]));
                            }//End for                            
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(float) * (end - i));
                        }//End if
#pragma endregion
#pragma region double 
                        else if constexpr (std::is_same<T, double>() == true)
                        {
                            /*
                            for (; i < end; i += 4)
                            {
                                if ((i + 4) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                //Copy with avx2
                                _mm256_store_pd(&dest[i], _mm256_load_pd(&source[i]));
                            }//End for
                            */
                            //Finish leftovers
                            std::memmove(&dest[i], &source[i], sizeof(double) * (end-i));
                        }//End if
#pragma endregion
#pragma region generic
                        else
                        {
                            std::memmove(&dest[i], &source[i], sizeof(T) * (end - i));
                        }//End else
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
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): " << ex.code()
                << " " << ex.what() << "\n";
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): " << ex.what() << "\n";
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): " << ex.what() << "\n";
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy(): unknown!\n";
        }//End catch
    }//End of copy
#pragma endregion
}//End of namespace