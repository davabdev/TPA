#pragma once
/*
* Truly Parallel Algorithms Library - Algorithm - fill function
* By: David Aaron Braun
* 2021-05-19
* Parallel implementation of fill
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
#pragma region generic

    /// <summary>
    /// Fills the container with the specified value upto the specified index
    /// </summary>
    /// <typeparam name="CONTAINER"></typeparam>
    /// <typeparam name="T"></typeparam>
    /// <param name="arr"></param>
    /// <param name="val"></param>
    /// <param name="item_count"> - optional defaults to the size of the container</param>
    template<typename CONTAINER, typename T>
    inline constexpr void fill(CONTAINER& arr, const T val, size_t item_count = 0)
        requires tpa::util::contiguous_seqeunce<CONTAINER>
    {
        using CONTAINER_TYPE = CONTAINER::value_type;

        try
        {
            static_assert(std::is_same<CONTAINER_TYPE, T>(), "Compile Error! The container must be of the same value type as the supplied value!");

            //Prevent overflow
            if (item_count <= 0 || item_count > arr.size())
            {
                item_count = arr.size();
            }//End if

            uint32_t complete = 0;

            std::vector<std::pair<size_t, size_t>> sections;
            tpa::util::prepareThreading(sections, item_count);

            std::vector<std::shared_future<uint32_t>> results;
            results.reserve(tpa::nThreads);

            std::shared_future<uint32_t> temp;

            for (const auto& sec : sections)
            {
                temp = tpa::tp->addTask([&arr, &val, &sec]()
                    {
                        const size_t beg = sec.first;
                        const size_t end = sec.second;
                        size_t i = beg;
#pragma region byte
                        if constexpr (std::is_same<T, int8_t>())
                        {
                            /*
                            const __m256i _Val = _mm256_set1_epi8(static_cast<int8_t>(val));

                            for (; i < end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _mm256_store_si256((__m256i*) &arr[i], _Val);
                            }//End for

                            //Finish leftovers
                            for (; i < end; ++i)
                            {
                                arr[i] = val;
                            }//End for
                            */
                            std::memset(&arr[i], val, end - i);
                        }//End if
#pragma endregion
#pragma region unsigned byte
                        else if constexpr (std::is_same<T, uint8_t>())
                        {
                            /*
                            const __m256i _Val = _mm256_set1_epi8(static_cast<uint8_t>(val));

                            for (; i < end; i += 32)
                            {
                                if ((i + 32) > end) [[unlikely]]
                                {
                                    break;
                                }//End if

                                _mm256_store_si256((__m256i*) &arr[i], _Val);
                            }//End for

                            //Finish leftovers
                            for (; i < end; ++i)
                            {
                                arr[i] = val;
                            }//End for
                            */

#ifdef _MSC_VER
                            __stosb(&arr[i], val, end - i);
#else

                            std::memset(&arr[i], val, end - i);
#endif
                        }//End if
#pragma endregion
#pragma region short
                        else if constexpr (std::is_same<T, int16_t>())
                        {
#ifdef TPA_X86_64
                            if (tpa::hasAVX512)
                            {
                                const __m512i _Val = _mm512_set1_epi16(static_cast<int16_t>(val));

                                for (; (i + 32uz) < end; i += 32uz)
                                {
                                    _mm512_store_si512(&arr[i], _Val);
                                }//End for
                            }//End if hasAVX512
                            else if (tpa::hasAVX2)
                            {
                                const __m256i _Val = _mm256_set1_epi16(static_cast<int16_t>(val));

                                for (; (i + 16uz) < end; i += 16uz)
                                {
                                    _mm256_store_si256((__m256i*) &arr[i], _Val);
                                }//End for
                            }//End if hasAVX2
                            else if (tpa::has_SSE2)
                            {
                                const __m128i _Val = _mm_set1_epi16(static_cast<int16_t>(val));

                                for (; (i + 8uz) < end; i += 8uz)
                                {
                                    _mm_store_si128((__m128i*) &arr[i], _Val);
                                }//End for
                            }//End if hasAVX2
#endif
                            //Finish leftovers
                            for (; i != end; ++i)
                            {
                                arr[i] = val;
                            }//End for
                        }//End if
#pragma endregion
#pragma region unsigned short
                        else if constexpr (std::is_same<T, uint16_t>())
                        {
#ifdef _MSC_VER
                            __stosw(reinterpret_cast<uint16_t*>(&arr[i]), val, end - i);
#else

#ifdef TPA_X86_64
                            if (tpa::hasAVX512)
                            {
                                const __m512i _Val = _mm512_set1_epi16(static_cast<uint16_t>(val));

                                for (; (i + 32uz) < end; i += 32uz)
                                {
                                    _mm512_store_si512(&arr[i], _Val);
                                }//End for
                            }//End if hasAVX512
                            else if (tpa::hasAVX2)
                            {
                                const __m256i _Val = _mm256_set1_epi16(static_cast<uint16_t>(val));

                                for (; i + 16 < end; i += 16)
                                {
                                    _mm256_store_si256((__m256i*) & arr[i], _Val);
                                }//End for
                            }//End if hasAVX2
                            else if (tpa::has_SSE2)
                            {
                                const __m128i _Val = _mm_set1_epi16(static_cast<uint16_t>(val));

                                for (; (i + 8uz) < end; i += 8uz)
                                {
                                    _mm_store_si128((__m128i*) & arr[i], _Val);
                                }//End for
                            }//End if hasAVX2
#endif
                            //Finish leftovers
                            for (; i != end; ++i)
                            {
                                arr[i] = val;
                            }//End for
#endif
                        }//End if
#pragma endregion
#pragma region int
                        else if constexpr (std::is_same<T, int32_t>())
                        {
#ifdef TPA_X86_64
                            if (tpa::hasAVX512)
                            {
                                const __m512i _Val = _mm512_set1_epi32(static_cast<int32_t>(val));

                                for (; (i + 16uz) < end; i += 16uz)
                                {
                                    _mm512_store_si512(&arr[i], _Val);
                                }//End for
                            }//End if hasAVX512
                            if (tpa::hasAVX2)
                            {
                                const __m256i _Val = _mm256_set1_epi32(static_cast<int32_t>(val));

                                for (; (i + 8uz) < end; i += 8uz)
                                {
                                    _mm256_store_si256((__m256i*) & arr[i], _Val);
                                }//End for
                            }//End if hasAVX2
                            else if (tpa::has_SSE2)
                            {
                                const __m128i _Val = _mm_set1_epi32(static_cast<int32_t>(val));

                                for (; (i + 4uz) < end; i += 4uz)
                                {
                                    _mm_store_si128((__m128i*) & arr[i], _Val);
                                }//End for
                            }//End if hasSSE
#endif
                        //Finish leftovers
                            for (; i != end; ++i)
                            {
                                arr[i] = val;
                            }//End for
                        }//End if
#pragma endregion
#pragma region unsigned int
                        else if constexpr (std::is_same<T, uint32_t>())
                        {
#ifdef _MSC_VER
                            __stosd(reinterpret_cast<unsigned long*>(&arr[i]), val, end - i);
#else
#ifdef TPA_X86_64
                            if (tpa::hasAVX512)
                            {
                                const __m512i _Val = _mm512_set1_epi32(static_cast<uint32_t>(val));

                                for (; (i + 16uz) < end; i += 16uz)
                                {
                                    _mm512_store_si512(&arr[i], _Val);
                                }//End for
                            }//End if hasAVX512
                            else if (tpa::hasAVX2)
                            {
                                const __m256i _Val = _mm256_set1_epi32(static_cast<uint32_t>(val));

                                for (; (i + 8uz) < end; i += 8uz)
                                {
                                    _mm256_store_si256((__m256i*) & arr[i], _Val);
                                }//End for
                            }//End if haxAVX2
                            else if (tpa::has_SSE2)
                            {
                                const __m128i _Val = _mm_set1_epi32(static_cast<uint32_t>(val));

                                for (; (i + 4uz) < end; i += 4uz)
                                {
                                    _mm_store_si128((__m128i*) & arr[i], _Val);
                                }//End for
                            }//End if hasSSE
#endif
                            //Finish leftovers
                            for (; i != end; ++i)
                            {
                                arr[i] = val;
                            }//End for
#endif
                        }//End if
#pragma endregion
#pragma region long
                        else if constexpr (std::is_same<T, int64_t>())
                        {
#ifdef TPA_X86_64
                            if (tpa::hasAVX512)
                            {
                                const __m512i _Val = _mm512_set1_epi64(static_cast<int64_t>(val));

                                for (; (i + 8uz) < end; i += 8uz)
                                {
                                    _mm512_store_si512(&arr[i], _Val);
                                }//End for
                            }//End if hasAVX512
                            else if (tpa::hasAVX2)
                            {
                                const __m256i _Val = _mm256_set1_epi64x(static_cast<int64_t>(val));

                                for (; (i + 4uz) < end; i += 4uz)
                                {
                                    _mm256_store_si256((__m256i*) & arr[i], _Val);
                                }//End for
                            }//End if hasAVX2
                            else if (tpa::has_SSE2)
                            {
                                const __m128i _Val = _mm_set1_epi64x(static_cast<int64_t>(val));

                                for (; (i + 2uz) < end; i += 2uz)
                                {
                                    _mm_store_si128((__m128i*) & arr[i], _Val);
                                }//End for
                            }//End if hasSSE
#endif 
                        //Finish leftovers
                            for (; i != end; ++i)
                            {
                                arr[i] = val;
                            }//End for
                        }//End if
#pragma endregion
#pragma region unsigned long
                        else if constexpr (std::is_same<T, uint64_t>())
                        {
#ifdef _MSC_VER
                            __stosq(reinterpret_cast<uint64_t*>(&arr[i]), val, end - i);
#else
#ifdef TPA_X86_64
                            if (tpa::hasAVX512)
                            {
                                const __m512i _Val = _mm512_set1_epi64(static_cast<uint64_t>(val));

                                for (; (i + 8uz) < end; i += 8uz)
                                {
                                    _mm512_store_si512(&arr[i], _Val);
                                }//End for
                            }//End if hasAVX512
                            else if (tpa::hasAVX2)
                            {
                                const __m256i _Val = _mm256_set1_epi64x(static_cast<uint64_t>(val));

                                for (; (i + 4uz) < end; i += 4uz)
                                {
                                    _mm256_store_si256((__m256i*) & arr[i], _Val);
                                }//End for
                            }//End if hasAVX2
                            else if (tpa::has_SSE2)
                            {
                                const __m128i _Val = _mm_set1_epi64x(static_cast<uint64_t>(val));

                                for (; (i + 2uz) < end; i += 2uz)
                                {
                                    _mm_store_si128((__m128i*) & arr[i], _Val);
                                }//End for
                            }//End if hasSSE
#endif
                        //Finish leftovers
                            for (; i != end; ++i)
                            {
                                arr[i] = val;
                            }//End for
#endif
                        }//End if
#pragma endregion
#pragma region float
                        else if constexpr (std::is_same < T, float >())
                        {
#ifdef TPA_X86_64
                            if (tpa::hasAVX512)
                            {
                                const __m512 _Val = _mm512_set1_ps(static_cast<float>(val));

                                for (; (i + 16uz) < end; i += 16uz)
                                {
                                    _mm512_store_ps(&arr[i], _Val);
                                }//End for
                            }//End if hasAVX512
                            else if (tpa::hasAVX)
                            {
                                const __m256 _Val = _mm256_set1_ps(static_cast<float>(val));

                                for (; (i + 8uz) < end; i += 8uz)
                                {
                                    _mm256_store_ps(&arr[i], _Val);
                                }//End for
                            }//End if
                            else if (tpa::has_SSE)
                            {
                                const __m128 _Val = _mm_set1_ps(static_cast<float>(val));

                                for (; (i + 4uz) < end; i += 4uz)
                                {
                                    _mm_store_ps(&arr[i], _Val);
                                }//End for
                            }//End if hasSSE
#endif
                        //Finish leftovers
                            for (; i != end; ++i)
                            {
                                arr[i] = val;
                            }//End for
                        }//End if
#pragma endregion
#pragma region double
                        else if constexpr (std::is_same <T, double>())
                        {
#ifdef TPA_X86_64
                            if (tpa::hasAVX512)
                            {
                                const __m512d _Val = _mm512_set1_pd(static_cast<double>(val));

                                for (; (i + 8uz) < end; i += 8uz)
                                {
                                    _mm512_store_pd(&arr[i], _Val);
                                }//End for
                            }//End if hasAVX512
                            else if (tpa::hasAVX)
                            {
                                const __m256d _Val = _mm256_set1_pd(static_cast<double>(val));

                                for (; (i + 4uz) < end; i += 4uz)
                                {
                                    _mm256_store_pd(&arr[i], _Val);
                                }//End for
                            }//End if hasAVX2
                            else if (tpa::has_SSE2)
                            {
                                const __m128d _Val = _mm_set1_pd(static_cast<double> (val));

                                for (; (i + 2uz) < end; i += 2uz)
                                {
                                    _mm_store_pd(&arr[i], _Val);
                                }//End for
                            }//End if hasSSE2
#endif
                        //Finish leftovers
                            for (; i != end; ++i)
                            {
                                arr[i] = val;
                            }//End for
                        }//End if
#pragma endregion
#pragma region generic      
                        else
                        {
                            for (; i != end; ++i)
                            {
                                arr[i] = val;
                            }//End for     
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
            if (complete != tpa::nThreads)
            {
                throw tpa::exceptions::NotAllThreadsCompleted(complete);
            }//End if

        }//End try
        catch (const std::future_error& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::fill<T>(): " << ex.code()
                << " " << ex.what() << "\n";
        }//End catch
        catch (const std::bad_alloc& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::fill<T>(): " << ex.what() << "\n";
        }//End catch
        catch (const std::exception& ex)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::fill<T>(): " << ex.what() << "\n";
        }//End catch
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
            std::cerr << "Exception thrown in tpa::fill<T>(): unknown!\n";
        }//End catch
    }//End of fill
#pragma endregion
}//End of namespace