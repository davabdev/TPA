#pragma once
/*
* Truly Parallel Algorithms Library - Algorithm - copy_if function
* By: David Aaron Braun
* 2021-05-16
* Parallel implementation of copy_if
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
    /// <para>Parellel Implementation of copy_if taking a predicate function returning a bool, copying data from SOURCE to DEST if the data matches the predicate</para>
    /// <para>Does not use AVX</para>
    /// </summary>
    /// <typeparam name="SOURCE"></typeparam>
    /// <typeparam name="DEST"></typeparam>
    /// <typeparam name="UnaryPredicate"></typeparam>
    /// <param name="source"></param>
    /// <param name="dest"></param>
    /// <param name="eraseZeros"> - set to false for a performance increase (in some cases) at the expense of having to deal with the zeros later.</param>
    /// <param name="pred"></param>
    template <typename SOURCE, typename DEST, typename N>
    inline constexpr void copy_if(
        const SOURCE& source, 
        DEST& dest, 
        bool (*pred)(N), 
        bool eraseZeros = true,
        size_t item_count = 0
    ) requires tpa::util::contiguous_seqeunce<SOURCE> && tpa::util::contiguous_seqeunce<DEST>
    {
        try
        {
            static_assert(std::is_same<SOURCE::value_type, DEST::value_type>() == true, "Compile Error! The source and destination container must be of the same value type!");

            using T = SOURCE::value_type;

            //Prevent overflow
            if (item_count <= 0 || item_count > source.size() || item_count > dest.size())
            {
                item_count = dest.size();
            }//End if

            uint32_t complete = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, item_count);

            std::vector<std::shared_future<uint32_t>> results;
            results.reserve(tpa::nThreads);

            std::shared_future<uint32_t> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&source, &dest, &pred, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

#pragma region generic
                        
                        for (; i != end; ++i)
                        {
                            if (pred(source[i]) == true)
                            {
                                dest[i] = source[i];
                            }//End else
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

            if (eraseZeros == true)
            {
                dest.erase(std::remove_if(std::execution::par_unseq, dest.begin(), dest.end(), [](T x) {return x == 0; }));
            }//End if
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy_if(): " << ex.code()
                << " " << ex.what() << "\n";
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy_if(): " << ex.what() << "\n";
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy_if(): " << ex.what() << "\n";
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy()_if: unknown!\n";
        }//End catch
    }//End of copy_if

    /// <summary>
    /// <para>Parallel Implementation of copy_if using a constexpr SIMD instructions, Copies data from SOURCE to DEST if the condtion matches</para>
    /// <para>Uses AVX2</para>
    /// </summary>
    /// <typeparam name="DEST"></typeparam>
    /// <typeparam name="SOURCE"></typeparam>
    /// <typeparam name="P"></typeparam>
    /// <param name="source"></param>
    /// <param name="dest"></param>
    /// <param name="eraseZeros"> - set to false for a performance increase (in some cases) at the expense of having to deal with the zeros later.</param>
    /// <param name="param">
    /// <para> -- Optional!  Only used with:        </para>
    /// <para>tpa::cond::EQUAL_TO		            </para>	
    /// <para>tpa::cond::NOT_EQUAL_TO			    </para>
    /// <para>tpa::cond::LESS_THAN		            </para>
    /// <para>tpa::cond::LESS_THAN_OR_EQUAL_TO      </para>
    /// <para>tpa::cond::GREATER_THAN               </para>
    /// <para>tpa::cond::GREATER_THAN_OR_EQUAL_TO   </para>
    /// <para>tpa::cond::FACTOR_OF                  </para>
    /// <para>tpa::cond::POWER_OF</para>
    /// <para>Other options not taking a parameter: </para>
    /// <para>tpa::cond::PRIME</para>
    /// <para>tpa::cond::EVEN</para>
    /// <para>tpa::cond::ODD</para>
    /// </param>
    template <tpa::cond COND, typename SOURCE, typename DEST, typename P = uint64_t>
    inline constexpr void copy_if(
        const SOURCE& source, 
        DEST& dest, 
        bool eraseZeros = true, 
        P param = 0,
        size_t item_count = 0)
        requires tpa::util::contiguous_seqeunce<SOURCE>&& tpa::util::contiguous_seqeunce<DEST>
    {
        try
        {
            static_assert(std::is_same<SOURCE::value_type, DEST::value_type>() == true, "Compile Error! The source and destination container must be of the same value type!");

            using T = SOURCE::value_type;

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
                temp = tpa::tp->addTask([&source, &dest, &param, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;

#pragma region int
						if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
                            if (tpa::hasAVX2)
                            {
                                const __m256i _ZERO = _mm256_set1_epi32(0);
                                const __m256i _ONE = _mm256_set1_epi32(1);
                                const __m256i _TWO = _mm256_set1_epi32(2);
                                __m256i _TEMP;
                                __m256i _MASK;
                                __m256 _MASKps;

                                __m256i _SOURCE;

                                for (; i+8 < end; i += 8)
                                {                                    
                                    //Set Values
                                    _SOURCE = _mm256_load_si256((__m256i*) & source[i]);

                                    //Filter
                                    if constexpr (COND == tpa::cond::EVEN)
                                    {
                                        _TEMP = _mm256_rem_epi32(_SOURCE, _TWO);

                                        _MASK = _mm256_cmpeq_epi32(_TEMP, _ZERO);

                                        //Store Result
                                        _mm256_maskstore_epi32((int*)&dest[i], _MASK, _SOURCE);
                                    }//End if
                                    else if constexpr (COND == tpa::cond::ODD)
                                    {
                                        _TEMP = _mm256_rem_epi32(_SOURCE, _TWO);

                                        _MASKps = _mm256_cmp_ps(_mm256_castsi256_ps(_TEMP), _mm256_castsi256_ps(_ZERO), _CMP_NEQ_OQ);

                                        //Store Result
                                        _mm256_maskstore_epi32((int32_t*)&dest[i], _mm256_castps_si256(_MASKps), _SOURCE);
                                    }//End if
                                    else if constexpr (COND == tpa::cond::PRIME)
                                    {
                                        goto correct_prime;

                                        /*
                                        __m256i _M = _mm256_div_epi32(_SOURCE, _TWO);
                                        __m256i _I = _mm256_set1_epi32(static_cast<int32_t>(2));
                                        __m256 _MUL;

                                        _MUL = _mm256_mul_ps(
                                            _mm256_castsi256_ps(_I),
                                            _mm256_castsi256_ps(_I)
                                        );

                                        _MASKps = _mm256_cmp_ps(
                                            _MUL,
                                            _mm256_castsi256_ps(_M),
                                            _CMP_LE_OQ
                                        );

                                        //Store Result
                                        _mm256_maskstore_epi32((int32_t*)&dest[i], _mm256_castps_si256(_MASKps), _SOURCE);
                                        */

                                    }//End if
                                    else
                                    {
                                        [] <bool flag = false>()
                                        {
                                            static_assert(flag, " You have specifed an invalid predicate function in tpa::copy_if<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
                                        }();
                                    }//End else
                                }//End for
                            }//End if hasAVX2
#endif
							for (; i != end; ++i)
							{
								//Calc
								if constexpr (COND == tpa::cond::EVEN)
								{
                                    if (tpa::util::isEven(source[i]))
                                    {
                                        dest[i] = source[i];
                                    }//End if
								}//End if
                                else if constexpr (COND == tpa::cond::ODD)
                                {
                                    if (tpa::util::isOdd(source[i]))
                                    {
                                        dest[i] = source[i];
                                    }//End if
                                }//End if
                                else if constexpr (COND == tpa::cond::PRIME)
                                {
                                    correct_prime:
                                    if (tpa::util::isPrime(source[i]))
                                    {
                                        dest[i] = source[i];
                                    }//End if
                                }//End if
								else
								{
                                    [] <bool flag = false>()
                                    {
                                        static_assert(flag, " You have specifed an invalid predicate function in tpa::copy_if<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
                                    }();
								}//End else
							}//End for
						}//End if
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

            if (eraseZeros == true)
            {
                //Remove all zeros from dest
                dest.erase(std::remove_if(std::execution::par_unseq, dest.begin(), dest.end(), [](T x) {return x == static_cast<T>(0); }));
            }//End if
        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy_if(): " << ex.code()
                << " " << ex.what() << "\n";
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy_if(): " << ex.what() << "\n";
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy_if(): " << ex.what() << "\n";
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::copy_if(): unknown!\n";
        }//End catch
    }//End of copy_if
#pragma endregion
}//End of namespace
