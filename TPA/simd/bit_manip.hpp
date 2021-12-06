#pragma once
/*
*	SIMD Bit Manipulation functions for TPA Library
*	By: David Aaron Braun
*	2021-10-08
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include <type_traits>
#include <concepts>
#include <limits>
#include <utility>
#include <mutex>
#include <future>
#include <iostream>
#include <functional>
#include <bitset>
#include <bit>

#ifdef _M_AMD64
#include <immintrin.h>
#elif defined (_M_ARM64)
#ifdef _WIN32
#include "arm64_neon.h"
#else
#include "arm_neon.h"
#endif
#endif

#include "../_util.hpp"
#include "../ThreadPool.hpp"
#include "../excepts.hpp"
#include "../size_t_lit.hpp"

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>Bit Manipulation Functions.</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa::bit_manip
{
	/// <summary>
	/// <para>Sets a bit to 1 at the specified position</para>
	/// <para>The bit to be set must be within the bounds of 'x'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="pos"></param>
	/// <returns></returns>
	template<typename T>
	inline void set(T& x, const uint64_t pos)
	{
		try
		{
			//Check Bounds
			if (pos < 0ull || pos > static_cast<uint64_t>((sizeof(T) * CHAR_BIT)-1))
			{
				throw std::out_of_range("Position must be within the bounds of T");
			}//End if

			if constexpr (std::is_integral<T>())
			{
				x = (1ull << pos) | x;
			}//End if
			else if constexpr (std::is_same<T, float>())
			{
				int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

				x_as_int = (1ull << pos) | x_as_int;

				x = *reinterpret_cast<float*>(&x_as_int);
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

				x_as_int = (1ull << pos) | x_as_int;

				x = *reinterpret_cast<double*>(&x_as_int);
			}//End if
			else
			{
				[] <bool flag = false>()
				{
					static_assert(flag, "Non-standard types are not supported.");
				}();
			}//End else
		}//End of try
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::set: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::set: unknown!\n";
		}//End catch
	}//End of set

	/// <summary>
	/// <para>Sets a bit to 0 at the specified position</para>
	/// <para>The bit to be cleared must be within the bounds of 'x'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="pos"></param>
	/// <returns></returns>
	template<typename T>
	inline void clear(T& x, const uint64_t pos)
	{
		try
		{
			//Check Bounds
			if (pos < 0ull || pos > static_cast<uint64_t>((sizeof(T) * CHAR_BIT) - 1))
			{
				throw std::out_of_range("Position must be within the bounds of T");
			}//End if

			if constexpr (std::is_integral<T>())
			{
				x = ~(1ull << pos) & x;
			}//End if
			else if constexpr (std::is_same<T, float>())
			{
				int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

				x_as_int = ~(1ull << pos) & x_as_int;

				x = *reinterpret_cast<float*>(&x_as_int);
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

				x_as_int = ~(1ull << pos) & x_as_int;

				x = *reinterpret_cast<double*>(&x_as_int);
			}//End if
			else
			{
				[] <bool flag = false>()
				{
					static_assert(flag, "Non-standard types are not supported.");
				}();
			}//End else
		}//End of try
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::clear: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::clear: unknown!\n";
		}//End catch
	}//End of bit_clear

	/// <summary>
	/// <para>Toggles (flips) a bit at the specified position</para>
	/// <para>The bit to be toggled must be within the bounds of 'x'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="pos"></param>
	/// <returns></returns>
	template<typename T>
	inline void toggle(T& x, const uint64_t pos)
	{
		try
		{
			//Check Bounds
			if (pos < 0ull || pos > static_cast<uint64_t>((sizeof(T) * CHAR_BIT) - 1))
			{
				throw std::out_of_range("Position must be within the bounds of T");
			}//End if

			if constexpr (std::is_integral<T>())
			{
				x = (1ull << pos) ^ x;
			}//End if
			else if constexpr (std::is_same<T, float>())
			{
				int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

				x_as_int = (1ull << pos) ^ x_as_int;

				x = *reinterpret_cast<float*>(&x_as_int);
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

				x_as_int = (1ull << pos) ^ x_as_int;

				x = *reinterpret_cast<double*>(&x_as_int);
			}//End if
			else
			{
				[] <bool flag = false>()
				{
					static_assert(flag, "Non-standard types are not supported.");
				}();
			}//End else
		}//End of try
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::toggle: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::toggle: unknown!\n";
		}//End catch
	}//End of toggle

	/// <summary>
	/// <para>Sets all trailing zeros (0) to one (1)</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline void set_trailing_zeros(T& x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			x = (x - 1) | x;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

			x_as_int = (x_as_int - 1) | x_as_int;

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			x_as_int = (x_as_int - 1) | x_as_int;

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else		
	}//End of set_trailing_zeros

	/// <summary>
	/// <para>Sets all trailing ones (1) to zero (0)</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline void clear_trailing_ones(T& x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			x = (x + 1) & x;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

			x_as_int = (x_as_int + 1) & x_as_int;

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			x_as_int = (x_as_int + 1) & x_as_int;

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else		
	}//End of set_trailing_zeros

	/// <summary>
	/// <para>Sets all leading zeros (0) to one (1)</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline void set_leading_zeros(T& x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			int32_t count = 0;
			int32_t copy = x;

			while (!(copy & (~INT_MAX)))
			{
				count++;
				copy <<= 1;
			}//End while

			for (int32_t i = 0; i < count; ++i)
			{
				tpa::bit_manip::set(x, (((sizeof(T) * CHAR_BIT)-1) - i));
			}//End for

			x = *reinterpret_cast<float*>(&x);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

			int32_t count = 0;
			int32_t copy = x_as_int;

			while (!(copy & (~INT_MAX)))
			{
				count++;
				copy <<= 1;
			}//End while

			for (int32_t i = 0; i < count; ++i)
			{
				tpa::bit_manip::set(x_as_int, (((sizeof(float) * CHAR_BIT)-1) -i));
			}//End for

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			int64_t count = 0;
			int64_t copy = x_as_int;

			while (!(copy & (~INT_MAX)))
			{
				count++;
				copy <<= 1;
			}//End while

			for (int64_t i = 0; i < count; ++i)
			{
				tpa::bit_manip::set(x_as_int, (((sizeof(double)*CHAR_BIT)-1) - i));
			}//End for

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else		
	}//End of set_leading_zeros

	/// <summary>
	/// <para>Sets all leading ones (1) to zero (0)</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline void clear_leading_ones(T& x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			x = -x & x;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

			x_as_int = -x_as_int & (x_as_int);

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			x_as_int = -x_as_int & x_as_int;

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else		
	}//End of clear_leading_ones

	/// <summary>
	/// <para>Extracts the lowest set 1 bit</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline T extract_lsb(T& x) noexcept
	{
		T ret = {};

		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>() ||
			std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			ret = x & -x;
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
#ifdef _M_AMD64
			if (tpa::hasBMI1)
			{
				ret = _blsi_u32(x);
			}//End if
			else
			{
				ret = x & -x;
			}//End else
#else
			ret = x & -x;
#endif
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
#ifdef _M_AMD64
			if (tpa::hasBMI1)
			{
				ret = _blsi_u64(x);
			}//End if
			else
			{
				ret = x & -x;
			}//End else
#else
			ret = x & -x;
#endif
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

#ifdef _M_AMD64
			if (tpa::hasBMI1)
			{
				x_as_int = _blsi_u32(x_as_int);
			}//End if
			else
			{
				x_as_int = x_as_int & -x_as_int;
			}//End else
#else
			x_as_int = x_as_int & -x_as_int;
#endif

			ret = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

#ifdef _M_AMD64
			if (tpa::hasBMI1)
			{
				x_as_int = _blsi_u64(x_as_int);
			}//End if
			else
			{
				x_as_int = x_as_int & -x_as_int;
			}//End else
#else
			x_as_int = x_as_int & -x_as_int;
#endif

			ret = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else

		return ret;
	}//End of extract_lsb

	/// <summary>
	/// <para>Extracts the highest (most significant) set 1 bit</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline T extract_msb(T& x) noexcept
	{
		T ret = {};

		if constexpr (std::is_integral<T>())
		{
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);
			x |= (x >> 8);
			x |= (x >> 16);
			ret = x - (x >> 1);

		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			uint32_t x_as_int = *reinterpret_cast<uint32_t*>(&x);
			
			x_as_int |= (x_as_int >> 1);
			x_as_int |= (x_as_int >> 2);
			x_as_int |= (x_as_int >> 4);
			x_as_int |= (x_as_int >> 8);
			x_as_int |= (x_as_int >> 16);
			x_as_int = x_as_int - (x_as_int >> 1);

			ret = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			x_as_int |= (x_as_int >> 1);
			x_as_int |= (x_as_int >> 2);
			x_as_int |= (x_as_int >> 4);
			x_as_int |= (x_as_int >> 8);
			x_as_int |= (x_as_int >> 16);
			x_as_int |= (x_as_int >> 32);
			x_as_int = x_as_int - (x_as_int >> 1);

			ret = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else

		return ret;
	}//End of extract_msb

	/// <summary>
	/// <para>Returns the number of set one (1) bits in 'x'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr uint32_t pop_count(T& x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
#ifdef _M_AMD64
			if (tpa::hasPOPCNT)
			{
				if constexpr (std::is_same<T, uint64_t>() || std::is_same<T, int64_t>())
				{
					return _mm_popcnt_u64(x);
				}//End if
				else
				{
					return _mm_popcnt_u32(x);
				}//End else
			}//End if
			else
			{
				uint32_t c = 0;
				for (; x != 0; ++c)
				{
					x = x & (x - 1);
				}//End for

				return c;
#else
			uint32_t c = 0;
			for (; x != 0; ++c)
			{
				x = x & (x - 1);
			}//End for

			return c;
#endif
#ifdef _M_AMD64
			}//End else
#endif
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			uint32_t temp = *reinterpret_cast<uint32_t*>(&x);
#ifdef _M_AMD64
			if (tpa::hasPOPCNT)
			{
				return _mm_popcnt_u32(temp);
			}//End if
			else
			{
				uint32_t c = 0;
				for (; temp != 0; ++c)
				{
					temp = temp & (temp - 1);
				}//End for

				return c;
			}//End else
#else
			uint32_t c = 0;
			for (; temp != 0; ++c)
			{
				temp = temp & (temp - 1);
			}//End for

			return c;
#endif
#ifdef _M_AMD64
		}//End if
#endif
		else if constexpr (std::is_same<T, double>())
		{
			uint64_t temp = *reinterpret_cast<uint64_t*>(&x);
#ifdef _M_AMD64
			if (tpa::hasPOPCNT)
			{
				return _mm_popcnt_u64(temp);
			}//End if
			else
			{
				uint32_t c = 0;
				for (; temp != 0; ++c)
				{
					temp = temp & (temp - 1);
				}//End for

				return c;
#else
			uint32_t c = 0;
			for (; temp != 0; ++c)
			{
				temp = temp & (temp - 1);
			}//End for

			return c;
#endif
#ifdef _M_AMD64
			}//End else
#endif
		}//End if
	}//End of pop_count

	/// <summary>
	/// <para>Returns the number of clear zero (0) bits in 'x'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr uint32_t zero_count(T& x) noexcept
	{
		return static_cast<uint32_t>(sizeof(T) * CHAR_BIT) - tpa::bit_manip::pop_count(x);
	}//End of zero_count
}//End of namespace

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>SIMD Matrix Math Functions.</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa::simd
{
#pragma region generic
	/// <summary>
	/// <para>Performs bitwise operations on 2 aligned containers storing the result in a 3rd aligned container.</para> 
	/// <para> Containers of different types are allowed but not recomended.</para>
	/// <para> Containers of different value types are NOT allowed</para>
	/// <para> Containers do not have to be a particular size</para>
	/// <para> If passing 2 containers of different sizes, values will only be calculated up to the container with the smallest size, the destination container must be at least this large.</para> 
	/// 
	/// <para>Will work with floats and doubles but requires at least the SSE2 instruction set at runtime. AVX vastly prefered.</para>
	/// <para>Non-standard floating point types are not supported and will cause a compile error.</para>
	/// <para>Templated predicate takes 1 of these predicates: tpa::bit</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::bit::AND</para>			
	/// <para>tpa::bit::OR</para>
	/// <para>tpa::bit::XOR</para>
	/// <para>tpa::bit::AND_NOT</para>
	/// </summary>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::bit INSTR, typename CONTAINER_A, typename CONTAINER_B, typename CONTAINER_C>
	inline constexpr void bitwise(
		const CONTAINER_A& source1,
		const CONTAINER_B& source2,
		CONTAINER_C& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>&&
		tpa::util::contiguous_seqeunce<CONTAINER_C>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, CONTAINER_B::value_type>() &&
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>(),
				"Compile Error! The source and destination containers must be of the same value type!");

			using T = CONTAINER_A::value_type;

			//Determin the smallest container
			smallest = tpa::util::min(source1.size(), source2.size());

			if (dest.size() < smallest)
			{
				throw tpa::exceptions::ArrayTooSmall();
			}//End if
		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &source2, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;
#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+64) < end; i += 64)
								{
									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi8((__m512i*)&source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_si512(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m512i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) &source1[i]);
									_Bi = _mm_load_si128((__m128i*) &source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif
						}//End if

#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+64) < end; i += 64)
								{
									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi8((__m512i*)&source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_si512(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m512i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif				
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_si512(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_si512(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif 							
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_epi32(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_epi32(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_epi64(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_epi64(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; (i+4) < end; i += 4)
								{			
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _Bi, _DESTi;

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);
									_Bi = _mm_load_si128((__m128i*) & source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_DWQW)
							{
								__m512 _Ai, _Bi, _DESTi;

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);
									_Bi = _mm512_load_ps(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_ps(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm512_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);
									_Bi = _mm256_load_ps(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_ps(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm256_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE)
							{
								__m128 _Ai, _Bi, _DESTi;

								for (; (i+4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_ps(&source1[i]);
									_Bi = _mm_load_ps(&source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_ps(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if has_SSE
#endif						
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_DWQW)
							{
								__m512d _Ai, _Bi, _DESTi;

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);
									_Bi = _mm512_load_pd(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_pd(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm512_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _Bi, _DESTi;

								for (; (i+4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);
									_Bi = _mm256_load_pd(&source2[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_pd(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm256_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE2)
							{
								__m128d _Ai, _Bi, _DESTi;

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_pd(&source1[i]);
									_Bi = _mm_load_pd(&source2[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_pd(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif						
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Calc
							if constexpr (INSTR == tpa::bit::AND)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = tpa::util::fp_bitwise<tpa::bit::AND>(source1[i], source2[i]);
								}//End if
								else
								{
									dest[i] = source1[i] & source2[i];
								}//End else
							}//End if
							else if constexpr (INSTR == tpa::bit::OR)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = tpa::util::fp_bitwise<tpa::bit::OR>(source1[i], source2[i]);
								}//End if
								else
								{
									dest[i] = source1[i] | source2[i];
								}//End else
							}//End if
							else if constexpr (INSTR == tpa::bit::XOR)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = tpa::util::fp_bitwise<tpa::bit::XOR>(source1[i], source2[i]);
								}//End if
								else
								{
									dest[i] = source1[i] ^ source2[i];
								}//End else
							}//End if
							else if constexpr (INSTR == tpa::bit::AND_NOT)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = tpa::util::fp_bitwise<tpa::bit::AND_NOT>(source1[i], source2[i]);
								}//End if
								else
								{
									dest[i] = ~source1[i] & source2[i];
								}//End else
							}//End if							
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
								}();
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
			if (complete != nThreads)
			{
				throw tpa::exceptions::NotAllThreadsCompleted(complete);
			}//End if

		}//End try
		catch (const tpa::exceptions::ArrayTooSmall& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise: " << ex.what() << "\n";
			std::cerr << "tpa::simd::bitwise will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise: unknown!\n";
		}//End catch
	}//End of bitwise()

	/// <summary>
	/// <para>Performs bitwise operations on 1 aligned container and a constant value storing the result in a 2nd aligned container.</para> 
	/// <para> Containers of different types are allowed but not recomended.</para>
	/// <para> Containers of different value types are NOT allowed</para>
	/// <para> Containers do not have to be a particular size</para>
	/// <para> If passing 2 containers of different sizes, values will only be calculated up to the container with the smallest size, the destination container must be at least this large.</para> 
	/// 
	/// <para>Will work with floats and doubles but requires at least the SSE2 instruction set at runtime. AVX vastly prefered.</para>
	/// <para>Non-standard floating point types are not supported and will cause a compile error.</para>
	/// <para>Templated predicate takes 1 of these predicates: tpa::bit</para>
	/// <para>---------------------------------------------------</para>
	/// <para>tpa::bit::AND</para>			
	/// <para>tpa::bit::OR</para>
	/// <para>tpa::bit::XOR</para>
	/// <para>tpa::bit::AND_NOT</para>
	/// </summary>
	/// <param name="source1"></param>
	/// <param name="val"></param>
	/// <param name="dest"></param>
	template<tpa::bit INSTR, typename CONTAINER_A, typename T, typename CONTAINER_C>
	inline constexpr void bitwise_const(
		const CONTAINER_A& source1,
		const T val,
		CONTAINER_C& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_C>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, T>() &&
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>(),
				"Compile Error! The source, destination containers and value must be of the same value type!");

			//Determin the smallest container
			smallest = source1.size();

			if (dest.size() < smallest)
			{
				throw tpa::exceptions::ArrayTooSmall();
			}//End if
		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &val, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;
#pragma region byte
						if constexpr (std::is_same<T, int8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi8(val);

								for (; (i+64) < end; i += 64)
								{
									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_si512(_Ai, _Bi);
									}//End if
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi8(val);

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _DESTi;
								const __m128i _Bi = _mm_set1_epi8(val);

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) &source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif
						}//End if

#pragma endregion
#pragma region unsigned byte
						else if constexpr (std::is_same<T, uint8_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi8(val);

								for (; (i+64) < end; i += 64)
								{
									//Set Values
									_Ai = _mm512_loadu_epi8((__m512i*)&source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_si512(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi8((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi8(val);

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if has AVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _DESTi;
								const __m128i _Bi = _mm_set1_epi8(val);

								for (; (i + 16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint8_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif				
						}//End if
#pragma endregion
#pragma region short
						else if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi16(val);

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_si512(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi16(val);

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _DESTi;
								const __m128i _Bi = _mm_set1_epi16(val);

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi16(val);

								for (; (i+32) < end; i += 32)
								{
									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_si512(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_si512(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi16(val);

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _DESTi;
								const __m128i _Bi = _mm_set1_epi16(val);

								for (; (i + 8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint16_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif 							
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi32(val);

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_epi32(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi32(val);

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _DESTi;
								const __m128i _Bi = _mm_set1_epi32(val);

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi32(val);

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_epi32(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_epi32(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi32(val);

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _DESTi;
								const __m128i _Bi = _mm_set1_epi32(val);

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint32_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi64(val);

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_epi64(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi64x(val);

								for (; (i+4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _DESTi;
								const __m128i _Bi = _mm_set1_epi64(val);

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<int64_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;
								const __m512i _Bi = _mm512_set1_epi64(val);

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_epi64(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_epi64(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;
								const __m256i _Bi = _mm256_set1_epi64x(val);

								for (; (i+4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_si256(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_si256(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128i _Ai, _DESTi;
								const __m128i _Bi = _mm_set1_epi64(val);

								for (; (i + 2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_si128((__m128i*) & source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_si128(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_si128(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<uint64_t>).");
										}();
									}//End else

									//Store Result
									_mm_store_si128((__m128i*) & dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif							
						}//End if
#pragma endregion
#pragma region float
						else if constexpr (std::is_same<T, float>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_DWQW)
							{
								__m512 _Ai, _DESTi;
								const __m512 _Bi = _mm512_set1_ps(val);

								for (; (i+16) < end; i += 16)
								{
									//Set Values
									_Ai = _mm512_load_ps(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_ps(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm512_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256 _Ai, _DESTi;
								const __m256 _Bi = _mm256_set1_ps(val);

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm256_load_ps(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_ps(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm256_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX
							else if (tpa::has_SSE)
							{
								__m128 _Ai, _DESTi;
								const __m128 _Bi = _mm_set1_ps(val);

								for (; (i + 4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm_load_ps(&source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_ps(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_ps(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<float>).");
										}();
									}//End else

									//Store Result
									_mm_store_ps(&dest[i], _DESTi);
								}//End for
							}//End if has_SSE
#endif						
						}//End if
#pragma endregion
#pragma region double
						else if constexpr (std::is_same<T, double>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_DWQW)
							{
								__m512d _Ai, _DESTi;
								const __m512d _Bi = _mm512_set1_pd(val);

								for (; (i+8) < end; i += 8)
								{
									//Set Values
									_Ai = _mm512_load_pd(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm512_and_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm512_or_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm512_xor_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm512_andnot_pd(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm512_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX)
							{
								__m256d _Ai, _DESTi;
								const __m256d _Bi = _mm256_set1_pd(val);

								for (; (i+4) < end; i += 4)
								{
									//Set Values
									_Ai = _mm256_load_pd(&source1[i]);

									//Calc
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm256_and_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm256_or_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm256_xor_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm256_andnot_pd(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm256_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
							else if (tpa::has_SSE2)
							{
								__m128d _Ai, _DESTi;
								const __m128d _Bi = _mm_set1_pd(val);

								for (; (i+2) < end; i += 2)
								{
									//Set Values
									_Ai = _mm_load_pd(&source1[i]);

									//Calc									
									if constexpr (INSTR == tpa::bit::AND)
									{
										_DESTi = _mm_and_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::OR)
									{
										_DESTi = _mm_or_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::XOR)
									{
										_DESTi = _mm_xor_pd(_Ai, _Bi);
									}//End if
									else if constexpr (INSTR == tpa::bit::AND_NOT)
									{
										_DESTi = _mm_andnot_pd(_Ai, _Bi);
									}//End if									
									else
									{
										[] <bool flag = false>()
										{
											static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise<__UNDEFINED_PREDICATE__>(CONTAINER<double>).");
										}();
									}//End else

									//Store Result
									_mm_store_pd(&dest[i], _DESTi);
								}//End for
							}//End if has_SSE2
#endif						
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Calc
							if constexpr (INSTR == tpa::bit::AND)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = tpa::util::fp_bitwise<tpa::bit::AND>(source1[i], val);
								}//End if
								else
								{
									dest[i] = source1[i] & val;
								}//End else
							}//End if
							else if constexpr (INSTR == tpa::bit::OR)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = tpa::util::fp_bitwise<tpa::bit::OR>(source1[i], val);
								}//End if
								else
								{
									dest[i] = source1[i] | val;
								}//End else
							}//End if
							else if constexpr (INSTR == tpa::bit::XOR)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = tpa::util::fp_bitwise<tpa::bit::XOR>(source1[i], val);
								}//End if
								else
								{
									dest[i] = source1[i] ^ val;
								}//End else
							}//End if
							else if constexpr (INSTR == tpa::bit::AND_NOT)
							{
								if constexpr (std::is_floating_point<T>())
								{
									dest[i] = tpa::util::fp_bitwise<tpa::bit::AND_NOT>(source1[i], val);
								}//End if
								else
								{
									dest[i] = ~source1[i] & val;
								}//End else
							}//End if							
							else
							{
								[] <bool flag = false>()
								{
									static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bitwise_const<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
								}();
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
			if (complete != nThreads)
			{
				throw tpa::exceptions::NotAllThreadsCompleted(complete);
			}//End if

		}//End try
		catch (const tpa::exceptions::ArrayTooSmall& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise_const: " << ex.what() << "\n";
			std::cerr << "tpa::simd::bitwise_const will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise()_const: " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise_const: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise_const: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise_const: unknown!\n";
		}//End catch
	}//End of bitwise_const()

	/// <summary>
	/// <para>Shifts the element's bits in a container left by the number specified in the second container, storing the results in a 3rd container.</para>
	/// <para>Containers may be of differnt types, but not recommended.</para>
	/// <para>Containers MUST have the same value_type.</para>
	/// <para>Containers do not have to be a particular size, of passing containers of different sizes, will only execute upto the size of the smallest container, the destination container must be at least this size.</para>
	/// <para>Non-standard integers will work. Floats are not supported and will generate a compile error.</para>
	/// <para>This implementation uses SIMD (if avaialble for type) and Multi-Threading.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<typename CONTAINER_A, typename CONTAINER_B, typename CONTAINER_C>
	inline constexpr void bit_shift_left(
		const CONTAINER_A& source1,
		const CONTAINER_B& source2,
		CONTAINER_C& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>&&
		tpa::util::contiguous_seqeunce<CONTAINER_C>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, CONTAINER_B::value_type>() &&
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>(),
				"Compile Error! The source and destination containers must be of the same value type!");

			using T = CONTAINER_A::value_type;

			static_assert(!std::is_floating_point<T>(),"Compile Error! You cannot bitshift a floating point type!");

			//Determin the smallest container
			smallest = tpa::util::min(source1.size(), source2.size());

			if (dest.size() < smallest)
			{
				throw tpa::exceptions::ArrayTooSmall();
			}//End if
		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &source2, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region short
					if constexpr (std::is_same<T, int16_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; i != end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
								_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

								//Shift Left
								_DESTi = _mm512_sllv_epi16(_Ai, _Bi);

								//Store Result
								_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
#endif							
					}//End if
#pragma endregion
#pragma region unsigned short
					else if constexpr (std::is_same<T, uint16_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; i != end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
								_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

								//Shift Left
								_DESTi = _mm512_sllv_epi16(_Ai, _Bi);

								//Store Result
								_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
							}//End for
						}//End if							
#endif 							
					}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

									//Shift Left
									_DESTi = _mm512_sllv_epi32(_Ai, _Bi);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Shift Left
									_DESTi = _mm256_sllv_epi32(_Ai, _Bi);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

									//Shift Left
									_DESTi = _mm512_sllv_epi32(_Ai, _Bi);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Shift Left
									_DESTi = _mm256_sllv_epi32(_Ai, _Bi);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

									//Shift Left
									_DESTi = _mm512_sllv_epi64(_Ai, _Bi);

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Shift Left
									_DESTi = _mm256_sllv_epi64(_Ai, _Bi);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

									//Shift Left
									_DESTi = _mm512_sllv_epi64(_Ai, _Bi);

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Shift Left
									_DESTi = _mm256_sllv_epi64(_Ai, _Bi);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region generic
					for (; i != end; ++i)
					{
						//Shift Left
						dest[i] = source1[i] << source2[i];
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
			if (complete != nThreads)
			{
				throw tpa::exceptions::NotAllThreadsCompleted(complete);
			}//End if

		}//End try
		catch (const tpa::exceptions::ArrayTooSmall& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_left: " << ex.what() << "\n";
			std::cerr << "tpa::simd::bit_shift_left will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_left(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_left: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_left: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_left: unknown!\n";
		}//End catch
	}//End of bit_shift_left()

	/// <summary>
	/// <para>Shifts the element's bits in a container left by the number specified in 'amount', storing the results in a 3rd container.</para>
	/// <para>Containers may be of differnt types, but not recommended.</para>
	/// <para>Containers MUST have the same value_type.</para>
	/// <para>Containers do not have to be a particular size, of passing containers of different sizes, will only execute upto the size of the smallest container, the destination container must be at least this size.</para>
	/// <para>Non-standard integers will work. Floats are not supported and will generate a compile error.</para>
	/// <para>This implementation uses SIMD (if avaialble for type) and Multi-Threading.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <param name="source1"></param>
	/// <param name="amount"></param>
	/// <param name="dest"></param>
	template<typename CONTAINER_A, typename CONTAINER_C>
	inline constexpr void bit_shift_left(
		const CONTAINER_A& source1,
		const uint8_t amount,
		CONTAINER_C& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_C>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>(),
				"Compile Error! The source and destination containers must be of the same value type!");

			using T = CONTAINER_A::value_type;

			static_assert(!std::is_floating_point<T>(), "Compile Error! You cannot bitshift a floating point type!");

			//Determin the smallest container
			smallest = source1.size();

			if (dest.size() < smallest)
			{
				throw tpa::exceptions::ArrayTooSmall();
			}//End if
		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &amount, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region short
						if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Shift Left
									_DESTi = _mm512_slli_epi16(_Ai, amount);

									//Store Result
									_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512_BW
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Left
									_DESTi = _mm256_slli_epi16(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Shift Left
									_DESTi = _mm512_slli_epi16(_Ai, amount);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if has AVX512_BW				
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Left
									_DESTi = _mm256_slli_epi16(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif 							
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);

									//Shift Left
									_DESTi = _mm512_slli_epi32(_Ai, amount);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Left
									_DESTi = _mm256_slli_epi32(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);

									//Shift Left
									_DESTi = _mm512_slli_epi32(_Ai, amount);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Left
									_DESTi = _mm256_slli_epi32(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);

									//Shift Left
									_DESTi = _mm512_slli_epi64(_Ai, amount);

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Left
									_DESTi = _mm256_slli_epi64(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);

									//Shift Left
									_DESTi = _mm512_slli_epi64(_Ai, amount);

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Left
									_DESTi = _mm256_slli_epi64(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Shift Left
							dest[i] = source1[i] << amount;
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
			if (complete != nThreads)
			{
				throw tpa::exceptions::NotAllThreadsCompleted(complete);
			}//End if

		}//End try
		catch (const tpa::exceptions::ArrayTooSmall& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_left: " << ex.what() << "\n";
			std::cerr << "tpa::simd::bit_shift_left will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_left(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_left: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_left: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_left: unknown!\n";
		}//End catch
	}//End of bit_shift_left()

		/// <summary>
	/// <para>Shifts the element's bits in a container right by the number specified in the second container, storing the results in a 3rd container.</para>
	/// <para>Containers may be of differnt types, but not recommended.</para>
	/// <para>Containers MUST have the same value_type.</para>
	/// <para>Containers do not have to be a particular size, of passing containers of different sizes, will only execute upto the size of the smallest container, the destination container must be at least this size.</para>
	/// <para>Non-standard integers will work. Floats are not supported and will generate a compile error.</para>
	/// <para>This implementation uses SIMD (if avaialble for type) and Multi-Threading.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<typename CONTAINER_A, typename CONTAINER_B, typename CONTAINER_C>
	inline constexpr void bit_shift_right(
		const CONTAINER_A& source1,
		const CONTAINER_B& source2,
		CONTAINER_C& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>&&
		tpa::util::contiguous_seqeunce<CONTAINER_C>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, CONTAINER_B::value_type>() &&
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>(),
				"Compile Error! The source and destination containers must be of the same value type!");

			using T = CONTAINER_A::value_type;

			static_assert(!std::is_floating_point<T>(), "Compile Error! You cannot bitshift a floating point type!");

			//Determin the smallest container
			smallest = tpa::util::min(source1.size(), source2.size());

			if (dest.size() < smallest)
			{
				throw tpa::exceptions::ArrayTooSmall();
			}//End if
		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &source2, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region short
						if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

									//Shift Right
									_DESTi = _mm512_srlv_epi16(_Ai, _Bi);

									//Store Result
									_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
#endif							
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
									_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

									//Shift Right
									_DESTi = _mm512_srlv_epi16(_Ai, _Bi);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if							
#endif 							
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

									//Shift Right
									_DESTi = _mm512_srav_epi32(_Ai, _Bi);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Shift Right
									_DESTi = _mm256_srav_epi32(_Ai, _Bi);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi32((__m512i*)&source2[i]);

									//Shift Right
									_DESTi = _mm512_srav_epi32(_Ai, _Bi);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _Bi, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);
									_Bi = _mm256_load_si256((__m256i*) & source2[i]);

									//Shift Right
									_DESTi = _mm256_srav_epi32(_Ai, _Bi);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

									//Shift Right
									_DESTi = _mm512_srav_epi64(_Ai, _Bi);

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _Bi, _DESTi;

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);
									_Bi = _mm512_load_epi64((__m512i*)&source2[i]);

									//Shift Right
									_DESTi = _mm512_srav_epi64(_Ai, _Bi);

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
#endif							
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Shift Right
							dest[i] = source1[i] >> source2[i];
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
			if (complete != nThreads)
			{
				throw tpa::exceptions::NotAllThreadsCompleted(complete);
			}//End if

		}//End try
		catch (const tpa::exceptions::ArrayTooSmall& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_right: " << ex.what() << "\n";
			std::cerr << "tpa::simd::bit_shift_right will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_right(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_right: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_right: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_right: unknown!\n";
		}//End catch
	}//End of bit_shift_right()

		/// <summary>
	/// <para>Shifts the element's bits in a container right by the number specified in 'amount', storing the results in a 3rd container.</para>
	/// <para>Containers may be of differnt types, but not recommended.</para>
	/// <para>Containers MUST have the same value_type.</para>
	/// <para>Containers do not have to be a particular size, of passing containers of different sizes, will only execute upto the size of the smallest container, the destination container must be at least this size.</para>
	/// <para>Non-standard integers will work. Floats are not supported and will generate a compile error.</para>
	/// <para>This implementation uses SIMD (if avaialble for type) and Multi-Threading.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <param name="source1"></param>
	/// <param name="amount"></param>
	/// <param name="dest"></param>
	template<typename CONTAINER_A, typename CONTAINER_C>
	inline constexpr void bit_shift_right(
		const CONTAINER_A& source1,
		const uint8_t amount,
		CONTAINER_C& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_C>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, CONTAINER_C::value_type>(),
				"Compile Error! The source and destination containers must be of the same value type!");

			using T = CONTAINER_A::value_type;

			static_assert(!std::is_floating_point<T>(), "Compile Error! You cannot bitshift a floating point type!");

			//Determin the smallest container
			smallest = source1.size();

			if (dest.size() < smallest)
			{
				throw tpa::exceptions::ArrayTooSmall();
			}//End if
		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source1, &amount, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region short
						if constexpr (std::is_same<T, int16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Shift Right
									_DESTi = _mm512_srli_epi16(_Ai, amount);

									//Store Result
									_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512_BW
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Right
									_DESTi = _mm256_srli_epi16(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned short
						else if constexpr (std::is_same<T, uint16_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512_ByteWord)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 32)
								{
									if ((i + 32) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);

									//Shift Right
									_DESTi = _mm512_srli_epi16(_Ai, amount);

									//Store Result
									_mm512_storeu_epi16((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if has AVX512_BW				
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Right
									_DESTi = _mm256_srli_epi16(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif 							
						}//End if
#pragma endregion
#pragma region int
						else if constexpr (std::is_same<T, int32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);

									//Shift Right
									_DESTi = _mm512_srli_epi32(_Ai, amount);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Right
									_DESTi = _mm256_srli_epi32(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned int
						else if constexpr (std::is_same<T, uint32_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 16)
								{
									if ((i + 16) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi32((__m512i*)&source1[i]);

									//Shift Right
									_DESTi = _mm512_srli_epi32(_Ai, amount);

									//Store Result
									_mm512_store_epi32((__m512i*)&dest[i], _DESTi);
								}//End for
							}//End if
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Right
									_DESTi = _mm256_srli_epi32(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region long
						else if constexpr (std::is_same<T, int64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 8)
								{
									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);

									//Shift Right
									_DESTi = _mm512_srli_epi64(_Ai, amount);

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 4)
								{
									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Left
									_DESTi = _mm256_srli_epi64(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region unsigned long
						else if constexpr (std::is_same<T, uint64_t>() == true)
						{
#ifdef _M_AMD64
							if (tpa::hasAVX512)
							{
								__m512i _Ai, _DESTi;

								for (; i != end; i += 8)
								{

									if ((i + 8) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm512_load_epi64((__m512i*)&source1[i]);

									//Shift Right
									_DESTi = _mm512_srli_epi64(_Ai, amount);

									//Store Result
									_mm512_store_epi64((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX512
							else if (tpa::hasAVX2)
							{
								__m256i _Ai, _DESTi;

								for (; i != end; i += 4)
								{

									if ((i + 4) > end) [[unlikely]]
									{
										break;
									}//End if

									//Set Values
									_Ai = _mm256_load_si256((__m256i*) & source1[i]);

									//Shift Left
									_DESTi = _mm256_srli_epi64(_Ai, amount);

									//Store Result
									_mm256_store_si256((__m256i*) & dest[i], _DESTi);
								}//End for
							}//End if hasAVX2
#endif							
						}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Shift Left
							dest[i] = source1[i] << amount;
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
			if (complete != nThreads)
			{
				throw tpa::exceptions::NotAllThreadsCompleted(complete);
			}//End if

		}//End try
		catch (const tpa::exceptions::ArrayTooSmall& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_right: " << ex.what() << "\n";
			std::cerr << "tpa::simd::bit_shift_right will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_right(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_right: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_right: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_shift_right: unknown!\n";
		}//End catch
	}//End of bit_shift_right()

	/// <summary>
	/// <para>Invert the bits of the elements in the source container and store the result in the destination container.</para>
	/// <para>Containers of different types are allowed.</para>
	/// <para>Containers MUST be of the same value_type.</para>
	/// <para>Will work with floats and doubles but requires at least the SSE2 instruction set at runtime. AVX vastly prefered.</para>
	/// <para>Non-standard floating point types are not supported and will cause a compile error.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <param name="source1"></param>
	/// <param name="dest"></param>
	template<typename CONTAINER_A, typename CONTAINER_B>
	inline constexpr void bitwise_not(
		const CONTAINER_A& source,
		CONTAINER_B& dest)
		requires tpa::util::contiguous_seqeunce<CONTAINER_A>&&
		tpa::util::contiguous_seqeunce<CONTAINER_B>
	{
		size_t smallest = 0;
		uint32_t complete = 0;
		try
		{
			static_assert(
				std::is_same<CONTAINER_A::value_type, CONTAINER_B::value_type>(),
				"Compile Error! The source and destination containers must be of the same value type!");

			using T = CONTAINER_A::value_type;

			//Determin the smallest container
			smallest = source.size();

			if (dest.size() < smallest)
			{
				throw tpa::exceptions::ArrayTooSmall();
			}//End if
		recover:

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, smallest);

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source, &dest, &sec]()
					{
						const size_t beg = sec.first;
						const size_t end = sec.second;
						size_t i = beg;

#pragma region byte
					if constexpr (std::is_same<T, int8_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _DESTi;
							const __m512i _max = _mm512_set1_epi8(static_cast<int8_t>(std::numeric_limits<uint8_t>::max()));

							for (; i != end; i += 64)
							{
								if ((i + 64) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm512_loadu_epi8(&source[i]);

								//Bit Not
								_DESTi = _mm512_xor_si512(_Ai, _max);

								//Store Result
								_mm512_storeu_epi8(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX512_BW
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _DESTi;
							const __m256i _max = _mm256_set1_epi8(static_cast<int8_t>(std::numeric_limits<uint8_t>::max()));

							for (; i != end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm256_load_si256((__m256i*) &source[i]);

								//Bit Not
								_DESTi = _mm256_xor_si256(_Ai, _max);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX2
#endif							
					}//End if
#pragma endregion
#pragma region unsigned byte
					else if constexpr (std::is_same<T, uint8_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _DESTi;
							const __m512i _max = _mm512_set1_epi8(std::numeric_limits<uint8_t>::max());

							for (; i != end; i += 64)
							{
								if ((i + 64) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm512_loadu_epi8(&source[i]);

								//Bit Not
								_DESTi = _mm512_xor_si512(_Ai, _max);

								//Store Result
								_mm512_storeu_epi8(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX512_BW
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _DESTi;
							const __m256i _max = _mm256_set1_epi8(std::numeric_limits<uint8_t>::max());

							for (; i != end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source[i]);

								//Bit Not
								_DESTi = _mm256_xor_si256(_Ai, _max);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX2
#endif							
					}//End if
#pragma endregion
#pragma region short
					else if constexpr (std::is_same<T, int16_t>() == true)
					{
#ifdef _M_AMD64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _DESTi;
							const __m512i _max = _mm512_set1_epi16(static_cast<int16_t>(std::numeric_limits<uint16_t>::max()));

							for (; i != end; i += 32)
							{
								if ((i + 32) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm512_loadu_epi16(&source[i]);

								//Bit Not
								_DESTi = _mm512_xor_si512(_Ai, _max);

								//Store Result
								_mm512_storeu_epi16(&dest[i], _DESTi);
							}//End for
						}//End if hasAVX512_BW
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _DESTi;
							const __m256i _max = _mm256_set1_epi16(static_cast<int16_t>(std::numeric_limits<uint16_t>::max()));

							for (; i != end; i += 16)
							{
								if ((i + 16) > end) [[unlikely]]
								{
									break;
								}//End if

								//Set Values
								_Ai = _mm256_load_si256((__m256i*) & source[i]);

								//Bit Not
								_DESTi = _mm256_xor_si256(_Ai, _max);

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX2
#endif							
					}//End if
#pragma endregion
#pragma region unsigned short
					else if constexpr (std::is_same<T, uint16_t>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512_ByteWord)
					{
						__m512i _Ai, _DESTi;
						const __m512i _max = _mm512_set1_epi16(std::numeric_limits<uint16_t>::max());

						for (; i != end; i += 32)
						{
							if ((i + 32) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_loadu_epi16(&source[i]);

							//Bit Not
							_DESTi = _mm512_xor_si512(_Ai, _max);

							//Store Result
							_mm512_storeu_epi16(&dest[i], _DESTi);
						}//End for
					}//End if hasAVX512_BW
					else if (tpa::hasAVX2)
					{
						__m256i _Ai, _DESTi;
						const __m256i _max = _mm256_set1_epi16(std::numeric_limits<uint16_t>::max());

						for (; i != end; i += 16)
						{
							if ((i + 16) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_si256((__m256i*) & source[i]);

							//Bit Not
							_DESTi = _mm256_xor_si256(_Ai, _max);

							//Store Result
							_mm256_store_si256((__m256i*) & dest[i], _DESTi);
						}//End for
					}//End if hasAVX2
#endif							
					}//End if
#pragma endregion
#pragma region int
					else if constexpr (std::is_same<T, int32_t>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512i _Ai, _DESTi;
						const __m512i _max = _mm512_set1_epi32(static_cast<int32_t>(std::numeric_limits<uint32_t>::max()));

						for (; i != end; i += 16)
						{
							if ((i + 16) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_loadu_epi32(&source[i]);

							//Bit Not
							_DESTi = _mm512_xor_epi32(_Ai, _max);

							//Store Result
							_mm512_storeu_epi32(&dest[i], _DESTi);
						}//End for
					}//End if hasAVX512_BW
					else if (tpa::hasAVX2)
					{
						__m256i _Ai, _DESTi;
						const __m256i _max = _mm256_set1_epi32(static_cast<int32_t>(std::numeric_limits<uint32_t>::max()));

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_si256((__m256i*) & source[i]);

							//Bit Not
							_DESTi = _mm256_xor_si256(_Ai, _max);

							//Store Result
							_mm256_store_si256((__m256i*) & dest[i], _DESTi);
						}//End for
					}//End if hasAVX2
#endif							
					}//End if
#pragma endregion
#pragma region unsigned int
					else if constexpr (std::is_same<T, uint32_t>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512i _Ai, _DESTi;
						const __m512i _max = _mm512_set1_epi32(std::numeric_limits<uint32_t>::max());

						for (; i != end; i += 16)
						{
							if ((i + 16) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_loadu_epi32(&source[i]);

							//Bit Not
							_DESTi = _mm512_xor_epi32(_Ai, _max);

							//Store Result
							_mm512_storeu_epi32(&dest[i], _DESTi);
						}//End for
					}//End if hasAVX512_BW
					else if (tpa::hasAVX2)
					{
						__m256i _Ai, _DESTi;
						const __m256i _max = _mm256_set1_epi32(std::numeric_limits<uint32_t>::max());

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_si256((__m256i*) & source[i]);

							//Bit Not
							_DESTi = _mm256_xor_si256(_Ai, _max);

							//Store Result
							_mm256_store_si256((__m256i*) & dest[i], _DESTi);
						}//End for
					}//End if hasAVX2
#endif							
					}//End if
#pragma endregion
#pragma region long
					else if constexpr (std::is_same<T, int64_t>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512i _Ai, _DESTi;
						const __m512i _max = _mm512_set1_epi64(static_cast<int64_t>(std::numeric_limits<uint64_t>::max()));

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_loadu_epi64(&source[i]);

							//Bit Not
							_DESTi = _mm512_xor_epi64(_Ai, _max);

							//Store Result
							_mm512_storeu_epi64(&dest[i], _DESTi);
						}//End for
					}//End if hasAVX512_BW
					else if (tpa::hasAVX2)
					{
						__m256i _Ai, _DESTi;
						const __m256i _max = _mm256_set1_epi64x(static_cast<int64_t>(std::numeric_limits<uint64_t>::max()));

						for (; i != end; i += 4)
						{
							if ((i + 4) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_si256((__m256i*) & source[i]);

							//Bit Not
							_DESTi = _mm256_xor_si256(_Ai, _max);

							//Store Result
							_mm256_store_si256((__m256i*) & dest[i], _DESTi);
						}//End for
					}//End if hasAVX2
#endif							
					}//End if
#pragma endregion
#pragma region unsigned long
					else if constexpr (std::is_same<T, uint64_t>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512)
					{
						__m512i _Ai, _DESTi;
						const __m512i _max = _mm512_set1_epi64(std::numeric_limits<uint64_t>::max());

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_loadu_epi64(&source[i]);

							//Bit Not
							_DESTi = _mm512_xor_epi64(_Ai, _max);

							//Store Result
							_mm512_storeu_epi64(&dest[i], _DESTi);
						}//End for
					}//End if hasAVX512_BW
					else if (tpa::hasAVX2)
					{
						__m256i _Ai, _DESTi;
						const __m256i _max = _mm256_set1_epi64x(std::numeric_limits<uint64_t>::max());

						for (; i != end; i += 4)
						{
							if ((i + 4) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_si256((__m256i*) & source[i]);

							//Bit Not
							_DESTi = _mm256_xor_si256(_Ai, _max);

							//Store Result
							_mm256_store_si256((__m256i*) & dest[i], _DESTi);
						}//End for
					}//End if hasAVX2
#endif							
					}//End if
#pragma endregion
#pragma region float
					else if constexpr (std::is_same<T, float>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512_DWQW)
					{
						__m512 _Ai, _DESTi;
						const __m512 _max = _mm512_set1_ps(std::numeric_limits<float>::max());

						for (; i != end; i += 16)
						{
							if ((i + 16) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_load_ps(&source[i]);

							//Bit Not
							_DESTi = _mm512_xor_ps(_Ai, _max);

							//Store Result
							_mm512_store_ps(&dest[i], _DESTi);
						}//End for
					}//End if hasAVX512_DWQW
					else if (tpa::hasAVX)
					{
						__m256 _Ai, _DESTi;
						const __m256 _max = _mm256_set1_ps(std::numeric_limits<float>::max());

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_ps(&source[i]);

							//Bit Not
							_DESTi = _mm256_xor_ps(_Ai, _max);

							//Store Result
							_mm256_store_ps(&dest[i], _DESTi);
						}//End for
					}//End if hasAVX
#endif							
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>() == true)
					{
#ifdef _M_AMD64
					if (tpa::hasAVX512_DWQW)
					{
						__m512d _Ai, _DESTi;
						const __m512d _max = _mm512_set1_pd(std::numeric_limits<double>::max());

						for (; i != end; i += 8)
						{
							if ((i + 8) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm512_load_pd(&source[i]);

							//Bit Not
							_DESTi = _mm512_xor_pd(_Ai, _max);

							//Store Result
							_mm512_store_pd(&dest[i], _DESTi);
						}//End for
					}//End if hasAVX512_DWQW
					else if (tpa::hasAVX)
					{
						__m256d _Ai, _DESTi;
						const __m256d _max = _mm256_set1_pd(std::numeric_limits<double>::max());

						for (; i != end; i += 4)
						{
							if ((i + 4) > end) [[unlikely]]
							{
								break;
							}//End if

							//Set Values
							_Ai = _mm256_load_pd(&source[i]);

							//Bit Not
							_DESTi = _mm256_xor_pd(_Ai, _max);

							//Store Result
							_mm256_store_pd(&dest[i], _DESTi);
						}//End for
					}//End if hasAVX
#endif							
					}//End if
#pragma endregion
#pragma region generic
						for (; i != end; ++i)
						{
							//Bitwise Not
							if constexpr (std::is_floating_point<T>())
							{
								dest[i] = tpa::util::fp_bitwise_not(source[i]);
							}//End if
							else
							{
								dest[i] = ~source[i];
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
			if (complete != nThreads)
			{
				throw tpa::exceptions::NotAllThreadsCompleted(complete);
			}//End if

		}//End try
		catch (const tpa::exceptions::ArrayTooSmall& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise_not: " << ex.what() << "\n";
			std::cerr << "tpa::simd::bitwise_not will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise_not(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise_not: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise_not: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bitwise_not: unknown!\n";
		}//End catch
	}//End of bitwise_not()

#pragma endregion
}//End of namespace
