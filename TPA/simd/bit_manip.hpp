#pragma once
/*
*	SIMD Bit Manipulation functions for TPA Library
*	By: David Aaron Braun
*	2021-12-24
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

#include "../_util.hpp"
#include "../ThreadPool.hpp"
#include "../excepts.hpp"
#include "../size_t_lit.hpp"
#include "../_util.hpp"
#include "../tpa_macros.hpp"
#include "../predicates.hpp"
#include "../tpa_concepts.hpp"
#include "../InstructionSet.hpp"
#include "simd.hpp"

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
	inline constexpr void set(T& x, const uint64_t pos)
	{
		try
		{
			//Check Bounds
			if (pos < 0ull || pos > static_cast<uint64_t>((sizeof(T) * static_cast<uint64_t>(CHAR_BIT))-1ull))
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
	inline constexpr void clear(T& x, const uint64_t pos)
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
	/// <para>Reverses the order of all the bits in a primitive numeric type</para>
	/// <para>Can be undone by calling the 'reverse' again.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void reverse(T& x)
	{
		try
		{
			if constexpr (std::is_integral<T>())
			{
				T rev = 0;
				size_t s = sizeof(T) * CHAR_BIT;

				while(s > 0)
				{
					rev = (rev << 1) | (x & 0x01);
					x >>= 1;
					s -= 1uz;
				}//End while

				x = rev;
			}//End if
			else if constexpr (std::is_same<T, float>())
			{
				int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

				int32_t rev = 0;
				size_t s = sizeof(T) * CHAR_BIT;

				while(s > 0)
				{
					rev = (rev << 1) | (x_as_int & 1);
					x_as_int >>= 1;
					s -= 1uz;
				}//End while

				x = *reinterpret_cast<float*>(&rev);
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

				int64_t rev = 0ll;
				size_t s = sizeof(T) * CHAR_BIT;

				while(s > 0)
				{
					rev = (rev << 1ll) | (x_as_int & 1ll);
					x_as_int >>= 1ll;
					s -= 1uz;
				}//End for

				x = *reinterpret_cast<double*>(&rev);
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
			std::cerr << "Exception thrown in tpa::bit_manip::reverse: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::reverse: unknown!\n";
		}//End catch
	}//End of reverse

	/// <summary>
	/// <para>Toggles (flips) a bit at the specified position</para>
	/// <para>The bit to be toggled must be within the bounds of 'x'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="pos"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void toggle(T& x, const uint64_t pos)
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
	/// <para>Toggles (flips) all the bits in x</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void toggle_all(T& x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			x = ~x;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

			x_as_int = ~x_as_int;

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			x_as_int = ~x_as_int;

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else
	}//End of toggle_all

	/// <summary>
	/// <para>Sets all trailing zeros (0) to one (1)</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void set_trailing_zeros(T& x) noexcept
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
	inline constexpr void clear_trailing_ones(T& x) noexcept
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
	inline constexpr void set_leading_zeros(T& x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			T m = x | (x >> 1);
			m |= m >> 2;
			m |= m >> 4;
			m |= m >> 8;
			m |= m >> 16;
			m |= m >> 32;

			x = (x | ~m);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

			int32_t m = x_as_int | (x_as_int >> 1);
			m |= m >> 2;
			m |= m >> 4;
			m |= m >> 8;
			m |= m >> 16;

			x_as_int = (x_as_int | ~m);

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			int64_t m = x_as_int | (x_as_int >> 1ll);
			m |= m >> 2ll;
			m |= m >> 4ll;
			m |= m >> 8ll;
			m |= m >> 16ll;
			m |= m >> 32ll;

			x_as_int = (x_as_int | ~m);

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
	inline constexpr void clear_leading_ones(T& x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			constexpr T one = static_cast<T>(1);

			T count = 0;
			T copy = ~x;

			while (!(copy & (~std::numeric_limits<T>::max())))
			{
				count++;
				copy <<= one;
			}//End while

			for (T i = 0; i < count; ++i)
			{
				x = ~(one << (((sizeof(T) * CHAR_BIT) - one) - i)) & x;
			}//End for
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

			int32_t count = 0;
			int32_t copy = ~x_as_int;

			while (!(copy & (~std::numeric_limits<int32_t>::max())))
			{
				count++;
				copy <<= 1;
			}//End while

			for (int32_t i = 0; i < count; ++i)
			{
				x_as_int = ~(1 << (((sizeof(T) * CHAR_BIT) - 1) - i)) & x_as_int;
			}//End for

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			int64_t count = 0;
			int64_t copy = ~x_as_int;

			while (!(copy & (~std::numeric_limits<int64_t>::max())))
			{
				count++;
				copy <<= 1;
			}//End while

			for (int64_t i = 0; i < count; ++i)
			{
				x_as_int = ~(1 << (((sizeof(T) * CHAR_BIT) - 1) - i)) & x_as_int;
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
	}//End of clear_leading_ones

	/// <summary>
	/// <para>Extracts the lowest set one (1) bit</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T extract_lsb(T& x) noexcept
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
	/// <para>Clears the lowest set 1 bit to 0</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void clear_lowest_set(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>() ||
			std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x = x ^ (x & -x);
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
#ifdef _M_AMD64
			if (tpa::hasBMI1)
			{
				x = x ^ _blsi_u32(x);
			}//End if
			else
			{
				x = x ^ (x & -x);
			}//End else
#else
			x = x ^ (x & -x);
#endif
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
#ifdef _M_AMD64
			if (tpa::hasBMI1)
			{
				x = x ^ _blsi_u64(x);
			}//End if
			else
			{
				x = x ^ (x & -x);
			}//End else
#else
			x = x ^ (x & -x);
#endif
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

#ifdef _M_AMD64
			if (tpa::hasBMI1)
			{
				x_as_int = x_as_int ^ _blsi_u32(x_as_int);
			}//End if
			else
			{
				x_as_int = x_as_int & -x_as_int;
			}//End else
#else
			x_as_int = x_as_int ^ (x_as_int & -x_as_int);
#endif

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

#ifdef _M_AMD64
			if (tpa::hasBMI1)
			{
				x_as_int = x_as_int ^ _blsi_u64(x_as_int);
			}//End if
			else
			{
				x_as_int = x_as_int ^ (x_as_int & -x_as_int);
			}//End else
#else
			x_as_int = x_as_int ^ (x_as_int & -x_as_int);
#endif

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else
	}//End of clear_lowest_set

	/// <summary>
	/// <para>Extracts the highest (most significant) set 1 bit</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T extract_msb(T& x) noexcept
	{
		T ret = {};

		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			ret = x | (x >> 1);
			ret = ret | (ret >> 2);
			ret = ret | (ret >> 4);
			ret = ((ret + 1) >> 1) | (ret & (1 << ((static_cast<T>(sizeof(ret)) * static_cast<T>(CHAR_BIT)) - 1)));

		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			ret = x | (x >> 1);
			ret = ret |(ret >> 2);
			ret = ret |(ret >> 4);
			ret = ret |(ret >> 8);
			ret = ((ret + 1) >> 1) | (ret & (1 << ((static_cast<T>(sizeof(ret)) * static_cast<T>(CHAR_BIT)) - 1)));
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			ret = x | (x >> 1);
			ret = ret | (ret >> 2);
			ret = ret | (ret >> 4);
			ret = ret | (ret >> 8);
			ret = ret | (ret >> 16);
			ret = ((ret + 1) >> 1) | (ret & (1 << ((static_cast<T>(sizeof(ret)) * static_cast<T>(CHAR_BIT)) - 1)));
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			ret = x | (x >> 1);
			ret = ret | (ret >> 2);
			ret = ret | (ret >> 4);
			ret = ret | (ret >> 8);
			ret = ret | (ret >> 16);
			ret = ret | (ret >> 32);
			ret = ((ret + 1) >> 1) | (ret & (1 << ((static_cast<T>(sizeof(ret)) * static_cast<T>(CHAR_BIT)) - 1)));
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			uint32_t x_as_int = *reinterpret_cast<uint32_t*>(&x);
			
			x_as_int = x_as_int | (x_as_int >> 1);
			x_as_int = x_as_int | (x_as_int >> 2);
			x_as_int = x_as_int | (x_as_int >> 4);
			x_as_int = x_as_int | (x_as_int >> 8);
			x_as_int = x_as_int | (x_as_int >> 16);
			x_as_int = ((x_as_int + 1) >> 1) | (x_as_int & (1 << ((static_cast<T>(sizeof(x_as_int)) * static_cast<T>(CHAR_BIT)) - 1)));

			ret = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			x_as_int = x_as_int | (x_as_int >> 1);
			x_as_int = x_as_int | x_as_int | (x_as_int >> 2);
			x_as_int = x_as_int | x_as_int | (x_as_int >> 4);
			x_as_int = x_as_int | x_as_int | (x_as_int >> 8);
			x_as_int = x_as_int | x_as_int | (x_as_int >> 16);
			x_as_int = x_as_int | (x_as_int >> 32);
			x_as_int = ((x_as_int + 1) >> 1) | (x_as_int & (1 << ((static_cast<T>(sizeof(x_as_int)) * static_cast<T>(CHAR_BIT)) - 1)));

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
	/// <para>Clears the highest set 1 bit to 0</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void clear_highest_set(T& x) noexcept
	{
		T orignal = x;

		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);

			//Prevent overflow
			x = ((x + 1) >> 1) | (x & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x = orignal ^ x;
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);
			x |= (x >> 8);

			//Prevent overflow
			x = ((x + 1) >> 1) | (x & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x = orignal ^ x;
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);
			x |= (x >> 8);
			x |= (x >> 16);
			
			//Prevent overflow
			x = ((x + 1) >> 1) | (x & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x = orignal ^ x;
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);
			x |= (x >> 8);
			x |= (x >> 16);
			x |= (x >> 32);
			
			//Prevent overflow
			x = ((x + 1) >> 1) | (x & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x = orignal ^ x;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			uint32_t x_as_int = *reinterpret_cast<uint32_t*>(&x);
			uint32_t orignal_as_int = *reinterpret_cast<uint32_t*>(&orignal);

			x_as_int |= (x_as_int >> 1);
			x_as_int |= (x_as_int >> 2);
			x_as_int |= (x_as_int >> 4);
			x_as_int |= (x_as_int >> 8);
			x_as_int |= (x_as_int >> 16);
			
			//Prevent overflow
			x_as_int = ((x_as_int + 1) >> 1) | (x_as_int & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x_as_int = orignal_as_int ^ x_as_int;

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);
			int64_t orignal_as_int = *reinterpret_cast<int64_t*>(&orignal);

			x_as_int |= (x_as_int >> 1);
			x_as_int |= (x_as_int >> 2);
			x_as_int |= (x_as_int >> 4);
			x_as_int |= (x_as_int >> 8);
			x_as_int |= (x_as_int >> 16);
			x_as_int |= (x_as_int >> 32);
			//Prevent overflow
			x_as_int = ((x_as_int + 1) >> 1) | (x_as_int & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x_as_int = orignal_as_int ^ x_as_int;

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else
	}//End of clear_highest_set

	/// <summary>
	/// <para>Sets the bit before the current lowest set to 1</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void set_lowest_clear(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>() ||
			std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			T lowest = (x & -x);
			lowest = lowest >> (static_cast<T>(static_cast<uint32_t>(lowest >> 1u) > 0u));

			x = x | lowest;
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
#ifdef TPA_X86_64
			if (tpa::hasBMI1)
			{
				T lowest = _blsi_u32(x);
				lowest = lowest >> (static_cast<T>(static_cast<uint32_t>(lowest >> 1u) > 0u));

				x = x | lowest;
			}//End if
			else
			{
				T lowest (x & -x);
				lowest = lowest >> (static_cast<T>(static_cast<uint32_t>(lowest >> 1u) > 0u));

				x = x | lowest;
			}//End else
#else
			T lowest = (x & -x);
			lowest = lowest >> (static_cast<T>(static_cast<uint32_t>(lowest >> 1u) > 0u));

			x = x | lowest;
#endif
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
#ifdef TPA_X86_64
			if (tpa::hasBMI1)
			{
				T lowest = _blsi_u64(x);
				lowest = lowest >> (static_cast<T>(static_cast<uint64_t>(lowest >> 1ull) > 0ull));

				x = x | lowest;
			}//End if
			else
			{
				T lowest = (x & -x);
				lowest = lowest >> (static_cast<T>(static_cast<uint64_t>(lowest >> 1ull) > 0ull));

				x = x | lowest;
			}//End else
#else
			T lowest = (x & -x);
			lowest = lowest >> (static_cast<T>(static_cast<uint64_t>(lowest >> 1ull) > 0ull));

			x = x | lowest;
#endif
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			uint32_t lowest = 0;
			uint32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

#ifdef TPA_X86_64
			if (tpa::hasBMI1)
			{
				lowest = _blsi_u32(x_as_int);
				lowest = lowest >> (static_cast<uint32_t>(static_cast<uint32_t>(lowest >> 1u) > 0u));

				x_as_int = x_as_int | lowest;
			}//End if
			else
			{
				lowest = x_as_int & -x_as_int;
				lowest = lowest >> (static_cast<uint32_t>(static_cast<uint32_t>(lowest >> 1u) > 0u));

				x_as_int = x_as_int | lowest;
			}//End else
#else
			lowest = x_as_int & -x_as_int;
			lowest = lowest >> (static_cast<uint32_t>(static_cast<uint32_t>(lowest >> 1u) > 0u));

			x_as_int = x_as_int | lowest;
#endif

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t lowest = 0;
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

#ifdef TPA_X86_64
			if (tpa::hasBMI1)
			{
				lowest = _blsi_u64(x_as_int);
				lowest = lowest >> (static_cast<uint64_t>(static_cast<uint64_t>(lowest >> 1ull) > 0ull));

				x_as_int = x_as_int | lowest;
			}//End if
			else
			{
				lowest = x_as_int & -x_as_int;
				lowest = lowest >> (static_cast<uint64_t>(static_cast<uint64_t>(lowest >> 1ull) > 0ull));

				x_as_int = x_as_int | lowest;
		}//End else
#else
			lowest = x_as_int & -x_as_int;
			lowest = lowest >> (static_cast<uint64_t>(static_cast<uint64_t>(lowest >> 1ull) > 0ull));

			x_as_int = x_as_int | lowest;
#endif

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else
	}//End of set_lowest_clear

	/// <summary>
	/// <para>Sets the bit after the current highest set to 1</para>
	/// <para>If the leftmost bit is already set this function makes no changes.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void set_highest_clear(T& x) noexcept
	{
		T orignal = x;

		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);

			//Prevent overflow
			x = ((x + 1) >> 1) | (x & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x = x << static_cast<T>(static_cast<uint32_t>(x << 1u) < static_cast<uint32_t>(std::numeric_limits<uint8_t>::max()));

			x = orignal | x;
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);
			x |= (x >> 8);

			//Prevent overflow
			x = ((x + 1) >> 1) | (x & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x = x << static_cast<T>(static_cast<uint32_t>(x << 1u) < static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()));

			x = orignal | x;
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);
			x |= (x >> 8);
			x |= (x >> 16);

			//Prevent overflow
			x = ((x + 1) >> 1) | (x & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x = x << static_cast<T>(static_cast<uint64_t>(x << 1u) < static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));

			x = orignal | x;
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			x |= (x >> 1);
			x |= (x >> 2);
			x |= (x >> 4);
			x |= (x >> 8);
			x |= (x >> 16);
			x |= (x >> 32);

			//Prevent overflow
			x = ((x + 1) >> 1) | (x & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x = x << static_cast<T>(static_cast<uint64_t>(x << 1ull) < static_cast<uint64_t>(std::numeric_limits<uint64_t>::max()));

			x = orignal | x;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			uint32_t x_as_int = *reinterpret_cast<uint32_t*>(&x);
			uint32_t orignal_as_int = *reinterpret_cast<uint32_t*>(&orignal);

			x_as_int |= (x_as_int >> 1);
			x_as_int |= (x_as_int >> 2);
			x_as_int |= (x_as_int >> 4);
			x_as_int |= (x_as_int >> 8);
			x_as_int |= (x_as_int >> 16);

			//Prevent overflow
			x_as_int = ((x_as_int + 1) >> 1) | (x_as_int & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x_as_int = x_as_int << static_cast<T>(static_cast<uint64_t>(x << 1u) < static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));

			x_as_int = orignal_as_int | x_as_int;

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);
			int64_t orignal_as_int = *reinterpret_cast<int64_t*>(&orignal);

			x_as_int |= (x_as_int >> 1);
			x_as_int |= (x_as_int >> 2);
			x_as_int |= (x_as_int >> 4);
			x_as_int |= (x_as_int >> 8);
			x_as_int |= (x_as_int >> 16);
			x_as_int |= (x_as_int >> 32);
			//Prevent overflow
			x_as_int = ((x_as_int + 1) >> 1) | (x_as_int & (1 << ((static_cast<T>(sizeof(x)) * static_cast<T>(CHAR_BIT)) - 1)));

			x_as_int = x_as_int << static_cast<T>(static_cast<uint64_t>(x << 1ull) < static_cast<uint64_t>(std::numeric_limits<uint64_t>::max()));

			x_as_int = orignal_as_int | x_as_int;

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, "Non-standard types are not supported.");
			}();
		}//End else
	}//End of set_highest_clear

	/// <summary>
	/// <para>Returns the number of set one (1) bits in 'x'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr uint64_t pop_count(T x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
#ifdef TPA_X86_64
			if (tpa::hasPOPCNT)
			{
				if constexpr (std::is_same<T, uint64_t>() || std::is_same<T, int64_t>())
				{
					return static_cast<uint64_t>(_mm_popcnt_u64(x));
				}//End if
				else if constexpr (std::is_same<T, uint32_t>() || std::is_same<T, int32_t>())
				{
					return static_cast<uint64_t>(_mm_popcnt_u32(x));
				}//End else
				else
				{
					uint64_t c = 0ull;
					for (; x != 0; ++c)
					{
						x = x & (x - 1);
					}//End for

					return c;
				}//End else
			}//End if
			else
			{
				uint64_t c = 0ull;
				for (; x != 0; ++c)
				{
					x = x & (x - 1);
				}//End for

				return c;
#else
			uint64_t c = 0ull;
			for (; x != 0; ++c)
			{
				x = x & (x - 1);
			}//End for

			return c;
#endif
#ifdef TPA_X86_64
			}//End else
#endif
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
		uint32_t temp = *reinterpret_cast<uint32_t*>(&x);
#ifdef TPA_X86_64
		if (tpa::hasPOPCNT)
		{
			return static_cast<uint64_t>(_mm_popcnt_u32(temp));
		}//End if
		else
		{
			uint32_t c = 0u;
			for (; temp != 0; ++c)
			{
				temp = temp & (temp - 1);
			}//End for

			return static_cast<uint64_t>(c);
		}//End else
#else
		uint64_t c = 0ull;
		for (; temp != 0; ++c)
		{
			temp = temp & (temp - 1);
		}//End for

		return c;
		}
#endif
#ifdef TPA_X86_64
		}//End if
#endif
		else if constexpr (std::is_same<T, double>())
		{
			uint64_t temp = *reinterpret_cast<uint64_t*>(&x);
#ifdef TPA_X86_64
			if (tpa::hasPOPCNT)
			{
				return _mm_popcnt_u64(temp);
			}//End if
			else
			{
				uint64_t c = 0ull;
				for (; temp != 0; ++c)
				{
					temp = temp & (temp - 1);
				}//End for

				return c;
#else
			uint64_t c = 0ull;
			for (; temp != 0; ++c)
			{
				temp = temp & (temp - 1);
			}//End for

			return c;
#endif
#ifdef TPA_X86_64
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
	[[nodiscard]] inline constexpr uint64_t zero_count(T& x) noexcept
	{
		return static_cast<uint64_t>(sizeof(T) * CHAR_BIT) - tpa::bit_manip::pop_count(x);
	}//End of zero_count

	/// <summary>
	/// <para>Returns the number of leading zero (0) bits in 'x'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr uint64_t leading_zero_count(T x) noexcept
	{
#ifdef TPA_X86_64
		if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			if (tpa::hasLZCNT)
			{
				return _lzcnt_u64(x);
			}//End if
			{
				uint64_t y = 0ull;
				uint64_t n = 64ull;

				y = x >> 32; if (y != 0) { n = n - 32; x = y; }
				y = x >> 16; if (y != 0) { n = n - 16; x = y; }
				y = x >> 8; if (y != 0) { n = n - 8; x = y; }
				y = x >> 4; if (y != 0) { n = n - 4; x = y; }
				y = x >> 2; if (y != 0) { n = n - 2; x = y; }
				y = x >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

				return static_cast<uint64_t>(n - x);
			}//End else
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			if (tpa::hasLZCNT)
			{
				return static_cast<uint64_t>(_lzcnt_u32(x));
			}//End if
			else
			{
				uint32_t y = 0;
				uint32_t n = 32;

				y = x >> 16; if (y != 0) { n = n - 16; x = y; }
				y = x >> 8; if (y != 0) { n = n - 8; x = y; }
				y = x >> 4; if (y != 0) { n = n - 4; x = y; }
				y = x >> 2; if (y != 0) { n = n - 2; x = y; }
				y = x >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

				return static_cast<uint64_t>(n - x);
			}//End else
		}//End else
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			uint16_t y = 0;
			uint16_t n = 16;

			y = x >> 8; if (y != 0) { n = n - 8; x = y; }
			y = x >> 4; if (y != 0) { n = n - 4; x = y; }
			y = x >> 2; if (y != 0) { n = n - 2; x = y; }
			y = x >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

			return static_cast<uint64_t>(n - x);
		}//End if
		else if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			uint8_t y = 0;
			uint8_t n = 8;

			y = x >> 4; if (y != 0) { n = n - 4; x = y; }
			y = x >> 2; if (y != 0) { n = n - 2; x = y; }
			y = x >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

			return static_cast<uint64_t>(n - x);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t temp = *reinterpret_cast<int64_t*>(&x);

			if (tpa::hasLZCNT)
			{
				return _lzcnt_u64(temp);
			}//End if
			else
			{
				uint64_t y = 0ull;
				uint64_t n = 64ull;

				y = temp >> 32; if (y != 0) { n = n - 32; temp = y; }
				y = temp >> 16; if (y != 0) { n = n - 16; temp = y; }
				y = temp >> 8; if (y != 0) { n = n - 8; temp = y; }
				y = temp >> 4; if (y != 0) { n = n - 4; temp = y; }
				y = temp >> 2; if (y != 0) { n = n - 2; temp = y; }
				y = temp >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

				return static_cast<uint64_t>(n - temp);
			}//End else
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t temp = *reinterpret_cast<int32_t*>(&x);
			if (tpa::hasLZCNT)
			{
				return static_cast<uint64_t>(_lzcnt_u32(temp));
			}//End if
			else
			{
				uint32_t y = 0;
				uint32_t n = 32;

				y = temp >> 16; if (y != 0) { n = n - 16; temp = y; }
				y = temp >> 8; if (y != 0) { n = n - 8; temp = y; }
				y = temp >> 4; if (y != 0) { n = n - 4; temp = y; }
				y = temp >> 2; if (y != 0) { n = n - 2; temp = y; }
				y = temp >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

				return static_cast<uint64_t>(n - temp);
			}//End else
		}//End else
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::leading_zero_count() This is not supported.");
			}();
		}//End else
#else
	if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
	{	
		uint64_t y = 0ull;
		uint64_t n = 64ull;

		y = x >> 32; if (y != 0) { n = n - 32; x = y; }
		y = x >> 16; if (y != 0) { n = n - 16; x = y; }
		y = x >> 8; if (y != 0) { n = n - 8; x = y; }
		y = x >> 4; if (y != 0) { n = n - 4; x = y; }
		y = x >> 2; if (y != 0) { n = n - 2; x = y; }
		y = x >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

		return static_cast<uint64_t>(n - x);
	}//End if
	else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
	{		
		uint32_t y = 0;
		uint32_t n = 32;

		y = x >> 16; if (y != 0) { n = n - 16; x = y; }
		y = x >> 8; if (y != 0) { n = n - 8; x = y; }
		y = x >> 4; if (y != 0) { n = n - 4; x = y; }
		y = x >> 2; if (y != 0) { n = n - 2; x = y; }
		y = x >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

		return static_cast<uint64_t>(n - x);
	}//End else
	else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
	{
		uint16_t y = 0;
		uint16_t n = 16;

		y = x >> 8; if (y != 0) { n = n - 8; x = y; }
		y = x >> 4; if (y != 0) { n = n - 4; x = y; }
		y = x >> 2; if (y != 0) { n = n - 2; x = y; }
		y = x >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

		return static_cast<uint64_t>(n - x);
	}//End if
	else if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
	{
		uint8_t y = 0;
		uint8_t n = 8;

		y = x >> 4; if (y != 0) { n = n - 4; x = y; }
		y = x >> 2; if (y != 0) { n = n - 2; x = y; }
		y = x >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

		return static_cast<uint64_t>(n - x);
	}//End if
	else if constexpr (std::is_same<T, double>())
	{
		int64_t temp = *reinterpret_cast<int64_t*>(&x);

		uint64_t y = 0ull;
		uint64_t n = 64ull;

		y = temp >> 32; if (y != 0) { n = n - 32; temp = y; }
		y = temp >> 16; if (y != 0) { n = n - 16; temp = y; }
		y = temp >> 8; if (y != 0) { n = n - 8; temp = y; }
		y = temp >> 4; if (y != 0) { n = n - 4; temp = y; }
		y = temp >> 2; if (y != 0) { n = n - 2; temp = y; }
		y = temp >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

		return static_cast<uint64_t>(n - temp);
	}//End if
	else if constexpr (std::is_same<T, float>())
	{
		int32_t temp = *reinterpret_cast<int32_t*>(&x);
		
		uint32_t y = 0;
		uint32_t n = 32;

		y = temp >> 16; if (y != 0) { n = n - 16; temp = y; }
		y = temp >> 8; if (y != 0) { n = n - 8; temp = y; }
		y = temp >> 4; if (y != 0) { n = n - 4; temp = y; }
		y = temp >> 2; if (y != 0) { n = n - 2; temp = y; }
		y = temp >> 1; if (y != 0) return static_cast<uint64_t>(n - 2);

		return static_cast<uint64_t>(n - temp);
	}//End else
	else
	{
		[] <bool flag = false>()
		{
			static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::leading_zero_count() This is not supported.");
		}();
	}//End else
#endif
	}//End of leading_zero_count

	/// <summary>
	/// <para>Returns the number of trailing zero (0) bits in 'x'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr uint64_t trailing_zero_count(T x) noexcept
	{
#ifdef TPA_X86_64
		if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			if (tpa::hasBMI1)
			{
				return _tzcnt_u64(x);
			}//End if
			{
				uint64_t count = 0ull;

				while (x != 0ull) 
				{
					if ((x & 1ull) == 1ull) 
					{
						break;
					}//End if
					else 
					{
						count++;
						x = (x >> 1ull);
					}//End else
				}//End while

				return count;
			}//End else
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			if (tpa::hasBMI1)
			{
				return static_cast<uint64_t>(_tzcnt_u32(x));
			}//End if
			else
			{
				uint32_t count = 0ul;

				while (x != 0ul)
				{
					if ((x & 1ul) == 1ul)
					{
						break;
					}//End if
					else
					{
						count++;
						x = (x >> 1ul);
					}//End else
				}//End while

				return static_cast<uint64_t>(count);
			}//End else
		}//End else
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			uint16_t count = 0;

			while (x != 0)
			{
				if ((x & 1) == 1)
				{
					break;
				}//End if
				else
				{
					count++;
					x = (x >> 1);
				}//End else
			}//End while

			return static_cast<uint64_t>(count);
		}//End if
		else if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			uint8_t count = 0;

			while (x != 0)
			{
				if ((x & 1) == 1)
				{
					break;
				}//End if
				else
				{
					count++;
					x = (x >> 1);
				}//End else
			}//End while

			return static_cast<uint64_t>(count);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t temp = *reinterpret_cast<int64_t*>(&x);

			if (tpa::hasBMI1)
			{
				return _tzcnt_u64(temp);
			}//End if
			else
			{
				uint64_t count = 0ull;

				while (temp != 0ull)
				{
					if ((temp & 1ull) == 1ull)
					{
						break;
					}//End if
					else
					{
						count++;
						temp = (temp >> 1ull);
					}//End else
				}//End while

				return static_cast<uint64_t>(count);
			}//End else
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t temp = *reinterpret_cast<int32_t*>(&x);

			if (tpa::hasBMI1)
			{
				return static_cast<uint64_t>(_tzcnt_u32(temp));
			}//End if
			else
			{
				uint32_t count = 0ul;

				while (temp != 0ul)
				{
					if ((temp & 1ul) == 1ul)
					{
						break;
					}//End if
					else
					{
						count++;
						temp = (temp >> 1ul);
					}//End else
				}//End while

				return static_cast<uint64_t>(count);
			}//End else
		}//End else
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::trailing_zero_count() This is not supported.");
			}();
		}//End else
#else
	if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
	{		
		uint64_t count = 0ull;

		while (x != 0ull)
		{
			if ((x & 1ull) == 1ull)
			{
				break;
			}//End if
			else
			{
				count++;
				x = (x >> 1ull);
			}//End else
		}//End while

		return count;
	}//End if
	else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
	{		
		uint32_t count = 0ul;

		while (x != 0ul)
		{
			if ((x & 1ul) == 1ul)
			{
				break;
			}//End if
			else
			{
				count++;
				x = (x >> 1ul);
			}//End else
		}//End while

		return static_cast<uint64_t>(count);
	}//End else
	else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
	{
		uint16_t count = 0;

		while (x != 0)
		{
			if ((x & 1) == 1)
			{
				break;
			}//End if
			else
			{
				count++;
				x = (x >> 1);
			}//End else
		}//End while

		return static_cast<uint64_t>(count);
	}//End if
	else if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
	{
		uint8_t count = 0;

		while (x != 0)
		{
			if ((x & 1) == 1)
			{
				break;
			}//End if
			else
			{
				count++;
				x = (x >> 1);
			}//End else
				}//End while

		return static_cast<uint64_t>(count);
			}//End if
	else if constexpr (std::is_same<T, double>())
	{
		int64_t temp = *reinterpret_cast<int64_t*>(&x);
				
		uint64_t count = 0ull;

		while (temp != 0ull)
		{
			if ((temp & 1ull) == 1ull)
			{
				break;
			}//End if
			else
			{
				count++;
				temp = (temp >> 1ull);
			}//End else
		}//End while

		return static_cast<uint64_t>(count);
	}//End if
	else if constexpr (std::is_same<T, float>())
	{
		int32_t temp = *reinterpret_cast<int32_t*>(&x);
				
		uint32_t count = 0ul;

		while (temp != 0ul)
		{
			if ((temp & 1ul) == 1ul)
			{
				break;
			}//End if
			else
			{
				count++;
				temp = (temp >> 1ul);
			}//End else
		}//End while

		return static_cast<uint64_t>(count);
	}//End else
	else
	{
		[] <bool flag = false>()
		{
			static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::trailing_zero_count() This is not supported.");
		}();
	}//End else
#endif
	}//End of trailing_zero_count

	/// <summary>
	/// <para>Returns the number of leading one (1) bits</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr uint64_t leading_one_count(T x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			return tpa::bit_manip::leading_zero_count(~x);
		}//End if
		else if constexpr (std::is_floating_point<T>())
		{
			return tpa::bit_manip::leading_zero_count(tpa::simd::fp_bitwise_not(x));
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::leading_one_count() This is not supported.");
			}();
		}//End else
	}//End of leading_one_count

	/// <summary>
	/// <para>Returns the number of trailing one (1) bits</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr uint64_t trailing_one_count(T x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			return tpa::bit_manip::trailing_zero_count(~x);
		}//End if
		else if constexpr (std::is_floating_point<T>())
		{
			return tpa::bit_manip::trailing_zero_count(tpa::simd::fp_bitwise_not(x));
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::trailing_one_count() This is not supported.");
			}();
		}//End else
	}//End of leading_one_count

	/// <summary>
	/// <para>Returns the number of bit islands (groups/blocks of set one (1) bits) in 'x'</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr uint64_t bit_island_count(T x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			return static_cast<uint64_t>((x & 1) + tpa::bit_manip::pop_count((x^(x>>1))) /2);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t temp = *reinterpret_cast<int32_t*>(&x);

			return static_cast<uint64_t>((temp & 1) + tpa::bit_manip::pop_count((temp ^ (temp >> 1))) / 2);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t temp = *reinterpret_cast<int64_t*>(&x);

			return static_cast<uint64_t>((temp & 1) + tpa::bit_manip::pop_count((temp ^ (temp >> 1))) / 2);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::bit_island_count() This is not supported.");
			}();
		}//End else
	}//End of bit_island_count

	/*
	* Hold the error results for bit_scan_forward if a char pointer is not passed.
	* 1 = no error, 0 = no bits were set
	*/
	unsigned char bsf_not_set = static_cast<unsigned char>(1);

	/*
	* Hold the error results for bit_scan_reverse if a char pointer is not passed.
	* 1 = no error, 0 = no bits were set
	*/
	unsigned char bsr_not_set = static_cast<unsigned char>(1);

	/// <summary>
	/// <para>Returns the index of the lowest set one (1) bit in 'x'</para>
	/// <para>If no bits in 'x' are set the return of this function will be zero '0'</para>
	/// <para>In the case that the bit at index zero could be set you can 
	/// optinally pass a char* which will be filled with a non-zero answer if bit 0 is set,</para>
	/// <para>This functionality is part of the bsf instruction and there is nothing that can be done about it.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="not_set"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr unsigned long bit_scan_forward(T x, unsigned char* not_set = &bsf_not_set) noexcept
	{
		unsigned long index = 0;

#if defined(TPA_X86_64)
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			if (x == 0)
			{
				*not_set = '\0';
				return 0;
			}//End if

			x = x & -x;

			if ((x & 0xf0f0f0f0) != 0) index += 4;
			if ((x & 0xcccccccc) != 0) index += 2;
			if ((x & 0xaaaaaaaa) != 0) index += 1;

			return index;
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			if (x == 0)
			{
				*not_set = '\0';
				return 0;
			}//End if

			x = x & -x;

			if ((x & 0xff00ff00) != 0) index += 8;
			if ((x & 0xf0f0f0f0) != 0) index += 4;
			if ((x & 0xcccccccc) != 0) index += 2;
			if ((x & 0xaaaaaaaa) != 0) index += 1;

			return index;
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			if (not_set == nullptr)
			{
				_BitScanForward(&index, x);
			}//End if
			else
			{
				*not_set = _BitScanForward(&index, x);
			}//End else

			return index;
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			if (not_set == nullptr)
			{
				_BitScanForward64(&index, x);
			}//End if
			else
			{
				*not_set = _BitScanForward64(&index, x);
			}//End else

			return index;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			unsigned long temp = *reinterpret_cast<int32_t*>(&x);

			if (not_set == nullptr)
			{
				_BitScanForward(&index, temp);
			}
			else
			{
				*not_set = _BitScanForward(&index, temp);
			}//End else

			return index;
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			unsigned long temp = *reinterpret_cast<int64_t*>(&x);

			if (not_set == nullptr)
			{
				_BitScanForward64(&index, temp);
			}//End if
			else
			{
				*not_set = _BitScanForward64(&index, temp);
			}//End else

			return index;
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::bit_scan_forward() This is not supported.");
			}();
		}//End else
#else
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			if (x == 0)
			{
				*not_set = '\0';
				return 0;
			}//End if

			x = x & -x;

			if ((x & 0xf0f0f0f0) != 0) index += 4;
			if ((x & 0xcccccccc) != 0) index += 2;
			if ((x & 0xaaaaaaaa) != 0) index += 1;

			return index;
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			if (x == 0)
			{
				*not_set = '\0';
				return 0;
			}//End if

			x = x & -x;

			if ((x & 0xff00ff00) != 0) index += 8;
			if ((x & 0xf0f0f0f0) != 0) index += 4;
			if ((x & 0xcccccccc) != 0) index += 2;
			if ((x & 0xaaaaaaaa) != 0) index += 1;

			return index;
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			if (x == 0)
			{
				*not_set = '\0';
				return 0;
			}//End if

			x = x & -x;

			if ((x & 0xffff0000) != 0) index += 16;
			if ((x & 0xff00ff00) != 0) index += 8;
			if ((x & 0xf0f0f0f0) != 0) index += 4;
			if ((x & 0xcccccccc) != 0) index += 2;
			if ((x & 0xaaaaaaaa) != 0) index += 1;

			return index;
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			if (x == 0)
			{
				*not_set = '\0';
				return 0;
			}//End if

			x = x & -x;

			if ((x & 0xffffffffll) != 0ll) index += 32ll;
			if ((x & 0xffff0000ll) != 0ll) index += 16ll;
			if ((x & 0xff00ff00ll) != 0ll) index += 8ll;
			if ((x & 0xf0f0f0f0ll) != 0ll) index += 4ll;
			if ((x & 0xccccccccll) != 0ll) index += 2ll;
			if ((x & 0xaaaaaaaall) != 0ll) index += 1ll;

			return index;
		}//End if
		else if constexpr (std::is_same<T, float>() || std::is_same<T, float>())
		{
			if (x == 0.0f)
			{
				*not_set = '\0';
				return 0;
			}//End if
			
			int32_t temp = *reinterpret_cast<int32_t*>(&x);		

			temp = temp & -temp;

			if ((temp & 0xffff0000) != 0) index += 16;
			if ((temp & 0xff00ff00) != 0) index += 8;
			if ((temp & 0xf0f0f0f0) != 0) index += 4;
			if ((temp & 0xcccccccc) != 0) index += 2;
			if ((temp & 0xaaaaaaaa) != 0) index += 1;

			return index;
		}//End if
		else if constexpr (std::is_same<T, double>() || std::is_same<T, double>())
		{
			if (x == 0.0)
			{
				*not_set = '\0';
				return 0;
			}//End if

			int64_t temp = *reinterpret_cast<int64_t*>(&x);

			temp = temp & -temp;

			if ((temp & 0xffffffffll) != 0ll) index += 32ll;
			if ((temp & 0xffff0000ll) != 0ll) index += 1ll6;
			if ((temp & 0xff00ff00ll) != 0ll) index += 8ll;
			if ((temp & 0xf0f0f0f0ll) != 0ll) index += 4ll;
			if ((temp & 0xccccccccll) != 0ll) index += 2ll;
			if ((temp & 0xaaaaaaaall) != 0ll) index += 1ll;

			return index;
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::bit_scan_forward() This is not supported.");
			}();
		}//End else
#endif
	}//End of bit_scan_forward

	/// <summary>
	/// <para>Sets all the even-numbered bits in a primitive type to 1</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void set_even(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			x = x | static_cast<T>(0b0101'0101);
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x = x | static_cast<T>(0b0101'0101'0101'0101);
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			x = x | static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101);
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			x = x | static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			x = *reinterpret_cast<float*>(*reinterpret_cast<int32_t*>(&x)) | static_cast<int32_t>(0b0101'0101'0101'0101'0101'0101'0101'0101);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			x = *reinterpret_cast<double*>(*reinterpret_cast<int64_t*>(&x)) | static_cast<int64_t>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::set_even() This is not supported.");
			}();
		}//End else
	}//End of set_even

	/// <summary>
	/// <para>Clears all the even-numbered bits in a primitive type to 0</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void clear_even(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			x = x & static_cast<T>(0b0101'0101);
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x = x & static_cast<T>(0b0101'0101'0101'0101);
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			x = x & static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101);
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			x = x & static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			x = *reinterpret_cast<float*>(*reinterpret_cast<int32_t*>(&x)) & static_cast<int32_t>(0b0101'0101'0101'0101'0101'0101'0101'0101);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			x = *reinterpret_cast<double*>(*reinterpret_cast<int64_t*>(&x)) & static_cast<int64_t>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::set_even() This is not supported.");
			}();
		}//End else
	}//End of clear_even

	/// <summary>
	/// <para>Toggles all the even-numbered bits in a primitive type</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void toggle_even(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			constexpr int8_t one = static_cast<int8_t>(1);

			for (int8_t i = static_cast<int8_t>(0); i != static_cast<int8_t>(8); i+=static_cast<int8_t>(2))
			{
				x = x ^ (one << i);
			}//End for
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			int16_t one = static_cast<int16_t>(1);

			for (int16_t i = static_cast<int16_t>(0); i != static_cast<int16_t>(16); i += static_cast<int16_t>(2))
			{
				x = x ^ (one << i);
			}//End for
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			for (int32_t i = 0; i != 32; i += 2)
			{
				x = x ^ (1 << i);
			}//End for
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			for (int64_t i = 0ll; i != 64ll; i += 2ll)
			{
				x = x ^ (1ll << i);
			}//End for
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

			for (int32_t i = 0; i != 32; i += 2)
			{
				x_as_int = x_as_int ^ (1 << i);
			}//End for

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			for (int64_t i = 0ll; i != 64ll; i += 2ll)
			{
				x_as_int = x_as_int ^ (1ll << i);
			}//End for

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::toggle_even() This is not supported.");
			}();
		}//End else
	}//End of set_even

	/// <summary>
	/// <para>Toggles all the odd-numbered bits in a primitive type</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void toggle_odd(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			constexpr int8_t one = static_cast<int8_t>(1);

			for (int8_t i = static_cast<int8_t>(1); i != static_cast<int8_t>(9); i += static_cast<int8_t>(2))
			{
				x = x ^ (one << i);
			}//End for
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			int16_t one = static_cast<int16_t>(1);

			for (int16_t i = static_cast<int16_t>(1); i != static_cast<int16_t>(17); i += static_cast<int16_t>(2))
			{
				x = x ^ (one << i);
			}//End for
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			for (int32_t i = 1; i != 33; i += 2)
			{
				x = x ^ (1 << i);
			}//End for
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			for (int64_t i = 1ll; i != 65ll; i += 2ll)
			{
				x = x ^ (1ll << i);
			}//End for
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

			for (int32_t i = 1; i != 33; i += 2)
			{
				x_as_int = x_as_int ^ (1 << i);
			}//End for

			x = *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			for (int64_t i = 1ll; i != 65ll; i += 2ll)
			{
				x_as_int = x_as_int ^ (1ll << i);
			}//End for

			x = *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::toggle_odd() This is not supported.");
			}();
		}//End else
	}//End of toggle_odd

	/// <summary>
	/// <para>Sets all the odd-numbered bits in a primitive type to 1</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="not_set"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void set_odd(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			x = x | static_cast<T>(0b1010'1010);
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x = x | static_cast<T>(0b1010'1010'1010'1010);
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			x = x | static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010);
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			x = x | static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			x = *reinterpret_cast<float*>(*reinterpret_cast<int32_t*>(&x)) | static_cast<int32_t>(0b1010'1010'1010'1010'1010'1010'1010'1010);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			x = *reinterpret_cast<double*>(*reinterpret_cast<int64_t*>(&x)) | static_cast<int64_t>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::set_even() This is not supported.");
			}();
		}//End else
	}//End of set_odd

	/// <summary>
	/// <para>Clears all the odd-numbered bits in a primitive type to 0</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="not_set"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void clear_odd(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			x = x & static_cast<T>(0b1010'1010);
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x = x & static_cast<T>(0b1010'1010'1010'1010);
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			x = x & static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010);
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			x = x & static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			x = *reinterpret_cast<float*>(*reinterpret_cast<int32_t*>(&x)) & static_cast<int32_t>(0b1010'1010'1010'1010'1010'1010'1010'1010);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			x = *reinterpret_cast<double*>(*reinterpret_cast<int64_t*>(&x)) & static_cast<int64_t>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::set_even() This is not supported.");
			}();
		}//End else
	}//End of clear_odd

	/// <summary>
	/// <para>Clears the most significant bit (leftmost bit) to 0</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void clear_msb(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			x = static_cast<T>(0b0111'1111) & x;
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x = static_cast<T>(0b0111'1111'1111'1111) & x;
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			x = static_cast<T>(0b0111'1111'1111'1111'1111'1111'1111'1111) & x;
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			x = static_cast<T>(0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111) & x;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			x = *reinterpret_cast<float*>(*reinterpret_cast<int32_t*>(&x)) & static_cast<int32_t>(0b0111'1111'1111'1111'1111'1111'1111'1111);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			x = *reinterpret_cast<double*>(*reinterpret_cast<int64_t*>(&x)) & static_cast<T>(0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::clear_msb() This is not supported.");
			}();
		}//End else
	}//End of clear_msb

	/// <summary>
	/// <para>Clears the least significant bit (rightmost bit) to 0</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void clear_lsb(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			x = static_cast<T>(0b1111'1110) & x;
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x = static_cast<T>(0b1111'1111'1111'1110) & x;
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			x = static_cast<T>(0b1111'1111'1111'1111'1111'1111'1111'1110) & x;
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			x = static_cast<T>(0b1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1110) & x;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			x = *reinterpret_cast<float*>(*reinterpret_cast<int32_t*>(&x)) & static_cast<int32_t>(0b1111'1111'1111'1111'1111'1111'1111'1110);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			x = *reinterpret_cast<double*>(*reinterpret_cast<int64_t*>(&x)) & static_cast<T>(0b1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1110);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::clear_lsb() This is not supported.");
			}();
		}//End else
	}//End of clear_lsb

	/// <summary>
	/// <para>Sets the most significant bit (leftmost bit) to 1</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void set_msb(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			x = x | static_cast<T>(0b1000'0000);
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x = x | static_cast<T>(0b1000'0000'0000'0000);
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			x = x | static_cast<T>(0b1000'0000'0000'0000'0000'0000'0000'0000);
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			x = x | static_cast<T>(0b1000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			x = *reinterpret_cast<float*>(*reinterpret_cast<int32_t*>(&x)) | static_cast<T>(0b1000'0000'0000'0000'0000'0000'0000'0000);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			x = *reinterpret_cast<double*>(*reinterpret_cast<int64_t*>(&x)) | static_cast<T>(0b1000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::set_msb() This is not supported.");
			}();
		}//End else
	}//End of set_msb

	/// <summary>
	/// <para>Sets the least significant bit (rightmost bit) to 1</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void set_lsb(T& x) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			x = x | static_cast<T>(0b0000'0001);
		}//End if
		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			x = x | static_cast<T>(0b0000'0000'0000'0001);
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
			x = x | static_cast<T>(0b0000'0000'0000'0000'0000'0000'0000'0001);
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			x = x | static_cast<T>(0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			x = *reinterpret_cast<float*>(*reinterpret_cast<int32_t*>(&x)) | static_cast<T>(0b0000'0000'0000'0000'0000'0000'0000'0001);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			x = *reinterpret_cast<double*>(*reinterpret_cast<int64_t*>(&x)) | static_cast<T>(0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::set_lsb() This is not supported.");
			}();
		}//End else
	}//End of set_lsb

	/// <summary>
	/// <para>Returns the index of the highest set one (1) bit in 'x'</para>
	/// <para>If no bits in 'x' are set the return of this function will be zero '0'</para>
	/// <para>In the case that the bit at index zero could be set you can 
	/// optinally pass a char* which will be filled with a non-zero answer if bit 0 is set,</para>
	/// <para>This functionality is part of the bsf instruction and there is nothing that can be done about it.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="not_set"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr unsigned long bit_scan_reverse(T x, unsigned char* not_set = &bsr_not_set) noexcept
	{
		unsigned long index = {};

#if defined(TPA_X86_64)	
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>() ||
			std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
			index = (sizeof(x) * CHAR_BIT) - tpa::bit_manip::leading_zero_count(x) - 1;

			return index;
		}//End if
		else if constexpr( std::is_same<T,int32_t>() || std::is_same<T,uint32_t>())
		{
			if (not_set == nullptr)
			{
				_BitScanReverse(&index, x);
			}//End if
			else
			{
				*not_set = _BitScanReverse(&index, x);
			}//End else

			return index;
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
			if (not_set == nullptr)
			{
				_BitScanReverse64(&index, x);
			}//End if
			else
			{
				*not_set = _BitScanReverse64(&index, x);
			}//End else

			return index;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			unsigned long temp = *reinterpret_cast<int32_t*>(&x);

			if (not_set == nullptr)
			{
				_BitScanReverse(&index, temp);
			}
			else
			{
				*not_set = _BitScanReverse(&index, temp);
			}//End else

			return index;
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			unsigned long temp = *reinterpret_cast<int64_t*>(&x);

			if (not_set == nullptr)
			{
				_BitScanReverse64(&index, temp);
			}//End if
			else
			{
				*not_set = _BitScanReverse64(&index, temp);
			}//End else

			return index;
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::bit_scan_forward() This is not supported.");
			}();
		}//End else
#else
		if constexpr (std::is_integral<T>())
		{
			index = (sizeof(x) * CHAR_BIT) - tpa::bit_manip::leading_zero_count(x) - 1;

			return index;
		}//End if
		else if constexpr (std::is_same<T,float>())
		{
			int32_t temp = *reinterpret_cast<int32_t*>(&x);

			index = (sizeof(temp) * CHAR_BIT) - tpa::bit_manip::leading_zero_count(temp) - 1;

			return index;
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t temp = *reinterpret_cast<int64_t*>(&x);

			index = (sizeof(temp) * CHAR_BIT) - tpa::bit_manip::leading_zero_count(temp) - 1;

			return index;
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::bit_scan_forward() This is not supported.");
			}();
		}//End else
#endif
	}//End of bit_scan_reverse

	/// <summary>
	/// <para>Returns the next value which can be represented within the bounds of 'x' with the same number of one (1) bits set.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T next_lexicographic_permutation(T x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			T one = static_cast<T>(1);

			T temp = x | (x - one);
			x = (temp + one) | (((~temp & -(~temp)) - one) >> (tpa::bit_manip::bit_scan_forward(x) + one));

			return x;
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

			int32_t temp = x_as_int | (x_as_int - 1);
			x_as_int = (temp + 1) | ((~temp & -(~temp)) - 1) >> (tpa::bit_manip::bit_scan_forward(x_as_int) + 1);

			return *reinterpret_cast<float*>(&x_as_int);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

			int64_t temp = x_as_int | (x_as_int - 1ll);
			x_as_int = (temp + 1ll) | ((~temp & -(~temp)) - 1ll) >> (tpa::bit_manip::bit_scan_forward(x_as_int) + 1ll);

			return *reinterpret_cast<double*>(&x_as_int);
		}//End if
		else
		{
			[] <bool flag = false>()
			{
				static_assert(flag, " You have passed a non-standard type in tpa::bitmanip::next_lexicographic_permutation() This is not supported.");
			}();
		}//End else
	}//End of next_lexicographic_permutation

	/// <summary>
	/// <para>Returns true if the bit specidied by 'pos' in 'x' is set to one (1)</para>
	/// <para>If 'pos' is outside the bounds of 'x' will throw an std::out_of_range exception.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="pos"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr bool is_set(T x, uint64_t pos)
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
				return (x & (1 << pos));
			}//End if 
			else if constexpr (std::is_same<T, float>())
			{
				int32_t temp = *reinterpret_cast<int32_t*>(&x);

				return (temp & (1 << pos));
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				int64_t temp = *reinterpret_cast<int64_t*>(&x);

				return (temp & (1ll << pos));
			}//End if
			else
			{
				return (x & (1 << pos));
			}//End else
		}//End try
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::is_set: " << ex.what() << "\n";
			return false;
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::is_set: unknown!\n";
			return false;
		}//End catch
	}//End of is_set

	/// <summary>
	/// <para>Returns true if the bit specidied by 'pos' in 'x' is set to zero (0)</para>
	/// <para>If 'pos' is outside the bounds of 'x' will throw an std::out_of_range exception.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="pos"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr bool is_clear(T x, uint64_t pos)
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
				return !(x & (1 << pos));
			}//End if 
			else if constexpr (std::is_same<T, float>())
			{
				int32_t temp = *reinterpret_cast<int32_t*>(&x);

				return !(temp & (1 << pos));
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				int64_t temp = *reinterpret_cast<int64_t*>(&x);

				return !(temp & (1ll << pos));
			}//End if
			else
			{
				return !(x & (1 << pos));
			}//End else
		}//End try
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::is_clear: " << ex.what() << "\n";
			return false;
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::is_clear: unknown!\n";
			return false;
		}//End catch
	}//End of is_clear

	/// <summary>
	/// <para>Sets all the bits in 'x' to one (1)</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void set_all(T& x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			x = ~(x & 0ll);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t temp = *reinterpret_cast<int32_t*>(&x);

			temp = ~(temp & 0);

			x = *reinterpret_cast<float*>(&temp);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t temp = *reinterpret_cast<int64_t*>(&x);

			temp = ~(temp & 0ll);

			x = *reinterpret_cast<double*>(&temp);
		}//End if
		else
		{
			x = ~(x & 0ll);
		}//End else
	}//End of set_all

	/// <summary>
	/// <para>Sets all the bits in 'x' to zero (0)</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <returns></returns>
	template<typename T>
	inline constexpr void clear_all(T& x) noexcept
	{
		if constexpr (std::is_integral<T>())
		{
			x = (x & 0ll);
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
			int32_t temp = *reinterpret_cast<int32_t*>(&x);

			temp = (temp & 0);

			x = *reinterpret_cast<float*>(&temp);
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
			int64_t temp = *reinterpret_cast<int64_t*>(&x);

			temp = (temp & 0ll);

			x = *reinterpret_cast<double*>(&temp);
		}//End if
		else
		{
			x = (x & 0ll);
		}//End else
	}//End of clear_all

	/// <summary>
	/// <para>Returns a type T which has had its bits set to the same as bits specificed in 'x' starting from 'start' and ending at 'start' + 'len'</para>
	/// <para>Warning: Currently only works as expected when start is set to bit# 0, bug fix coming.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="start"></param>
	/// <param name="len"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T extract(T x, uint64_t start, uint64_t len)
	{
		try
		{
			//Check Bounds
			if (start < 0ull || start > static_cast<uint64_t>((sizeof(T) * CHAR_BIT) - 1))
			{
				throw std::out_of_range("'start' must be within the bounds of T");
			}//End if

			if ((start + len) < 0ull || (start + len) > static_cast<uint64_t>((sizeof(T) * CHAR_BIT) - 1))
			{
				throw std::out_of_range("'start + len' must be within the bounds of T");
			}//End if

#ifdef TPA_X86_64
			if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>() || std::is_same<T, int16_t>() ||
				std::is_same<T, uint16_t>())
			{
				if (tpa::hasBMI1)
				{
					uint32_t temp = 0u;
					std::memmove(&temp, &x, sizeof(T));
					return static_cast<T>(_bextr_u32(temp, static_cast<uint32_t>(start), static_cast<uint32_t>(len)));
				}//End if
				else
				{
					x >>= start;
					const T mask = (1 << len) - 1;
					return x & mask;
				}//End else
			}//End if
			else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
			{
				if (tpa::hasBMI1)
				{
					return static_cast<T>(_bextr_u32(x, static_cast<uint32_t>(start), static_cast<uint32_t>(len)));
				}//End if
				else
				{
					x >>= start;
					const T mask = (1 << len) - 1;
					return x & mask;
				}//End else
			}//End else
			else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
			{
				if (tpa::hasBMI1)
				{
					return static_cast<T>(_bextr_u64(x, static_cast<uint32_t>(start), static_cast<uint32_t>(len)));
				}//End if
				else
				{
					x >>= start;
					const T mask = (1ll << len) - 1ll;
					return x & mask;
				}//End else
			}//End if
			else if constexpr (std::is_same<T, float>())
			{
				uint32_t temp = *reinterpret_cast<uint32_t*>(&x);

				if (tpa::hasBMI1)
				{
					temp = static_cast<T>(_bextr_u32(temp, static_cast<uint32_t>(start), static_cast<uint32_t>(len)));
					return *reinterpret_cast<float*>(&temp);
				}//End if
				else
				{
					temp >>= start;
					const uint32_t mask = (1u << len) - 1u;
					temp = temp & mask;

					return *reinterpret_cast<float*>(&temp);
				}//End else
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				uint64_t temp = *reinterpret_cast<uint64_t*>(&x);

				if (tpa::hasBMI1)
				{
					temp = static_cast<T>(_bextr_u64(temp, static_cast<uint32_t>(start), static_cast<uint32_t>(len)));
					return *reinterpret_cast<double*>(&temp);
				}//End if
				else
				{
					temp >>= start;
					const uint64_t mask = (1ull << len) - 1ull;
					temp = temp & mask;

					return *reinterpret_cast<double*>(&temp);
				}//End else
			}//End if
			else
			{
				x >>= start;
				const T mask = (1ull << len) - 1ull;
				return x & mask;
			}//End else

#elif defined(TPA_ARM) & !defined(_MSC_VER)
			if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>() || std::is_same<T, int16_t>() ||
				std::is_same<T, uint16_t>())
			{
				uint32_t temp = 0u;
				std::memmove(&temp, &x, sizeof(T));
				return static_cast<T>(_arm_ubfx(temp, static_cast<uint32_t>(start), static_cast<uint32_t>(len)));
			}//End if
			else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
			{
				return static_cast<T>(_arm_ubfx(x, static_cast<uint32_t>(start), static_cast<uint32_t>(len)));
			}//End else
			else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
			{
				x >>= start;
				const T mask = (1ll << len) - 1ll;
				return x & mask;
			}//End if
			else if constexpr (std::is_same<T, float>())
			{
				uint32_t temp = *reinterpret_cast<uint32_t*>(&x);

				temp = static_cast<T>(_arm_ubfx(temp, static_cast<uint32_t>(start), static_cast<uint32_t>(len)));
				return *reinterpret_cast<float*>(&temp);
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				uint64_t temp = *reinterpret_cast<uint64_t*>(&x);

				temp >>= start;
				const uint64_t mask = (1ull << len) - 1ull;
				temp = temp & mask;

				return *reinterpret_cast<double*>(&temp);
			}//End if
			else
			{
				x >>= start;
				const T mask = (1ull << len) - 1ull;
				return x & mask;
			}//End else
#else
			if constexpr (std::is_integral<T>())
			{
				x >>= start;
				const T mask = (1ull << len) - 1ull;
				return x & mask;
			}//End if
			else if constexpr (std::is_same<T, float>())
			{
				uint32_t temp = *reinterpret_cast<uint32_t*>(&x);

				temp >>= start;
				const uint32_t mask = (1u << len) - 1u;
				temp = temp & mask;

				return *reinterpret_cast<float*>(&temp);
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				uint64_t temp = *reinterpret_cast<uint64_t*>(&x);

				temp >>= start;
				const uint64_t mask = (1ull << len) - 1ull;
				temp = temp & mask;

				return *reinterpret_cast<double*>(&temp);
			}//End if
			else
			{
				x >>= start;
				const T mask = (1ull << len) - 1ull;
				return x & mask;
			}//End else
#endif

		}//End try
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::extract: " << ex.what() << "\n";
			return static_cast<T>(0);
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::extract: unknown!\n";
			return static_cast<T>(0);
		}//End catch
	}//End of extract

	/// <summary>
	/// <para>Copy bits from 'b' into 'a' where the corresponding bit in 'mask' is set a one (1)</para>
	/// <para>Different types are allowed but must be intentically sized.</para>
	/// </summary>
	/// <typeparam name="T1"></typeparam>
	/// <typeparam name="T2"></typeparam>
	/// <typeparam name="MASK"></typeparam>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <param name="mask"></param>
	/// <returns></returns>
	template<typename T1, typename T2, typename MASK>
	inline constexpr void masked_copy(T1& a, T2& b, MASK& mask) noexcept
	{
		static_assert(sizeof(T1) == sizeof(T2) && sizeof(T1) == sizeof(MASK), "'a', 'b', and 'mask' must be equally sized.");

		if constexpr (std::is_integral<T1>() && std::is_integral<T2>() && std::is_integral<MASK>())
		{
			a = ((b & mask) | (a & ~mask));
		}//End if
		else if constexpr (std::is_same<T1, float>() || std::is_same<T2, float>() || std::is_same<MASK, float>())
		{
			int32_t temp_a = *reinterpret_cast<int32_t*>(&a);
			int32_t temp_b = *reinterpret_cast<int32_t*>(&b);
			int32_t temp_mask = *reinterpret_cast<int32_t*>(&mask);
			int32_t res = 0;

			res = ((temp_b & temp_mask) | (temp_a & ~temp_mask));

			a = *reinterpret_cast<T1*>(&res);
		}//End if
		else if constexpr (std::is_same<T1, double>() || std::is_same<T2, double>() || std::is_same<MASK, double>())
		{
			int64_t temp_a = *reinterpret_cast<int64_t*>(&a);
			int64_t temp_b = *reinterpret_cast<int64_t*>(&b);
			int64_t temp_mask = *reinterpret_cast<int64_t*>(&mask);
			int64_t res = 0;

			res = ((temp_b & temp_mask) | (temp_a & ~temp_mask));

			a = *reinterpret_cast<T1*>(&res);
		}//End if
		else
		{
			a = ((b & mask) | (a & ~mask));
		}//End else
	}//End of masked_copy

	/// <summary>
	/// <para>Swaps the bits in 'x' at indices 'a' and 'b'</para>
	/// <para>'a' and 'b' must be within the bounds of 'x' or will 
	/// throw an std::out_of_range exception</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="a"></param>
	/// <param name="b"></param>
	template<typename T>
	inline constexpr void bit_swap(T& x, uint64_t a, uint64_t b)
	{
		try
		{
			//Check Bounds
			if (a < 0ull || a > static_cast<uint64_t>((sizeof(T) * CHAR_BIT) - 1))
			{
				throw std::out_of_range("'a' must be within the bounds of T");
			}//End if

			if (b < 0ull || b > static_cast<uint64_t>((sizeof(T) * CHAR_BIT) - 1))
			{
				throw std::out_of_range("'b' must be within the bounds of T");
			}//End if

			if constexpr (std::is_integral<T>())
			{
				x ^= (1 << a);
				x ^= (1 << b);
			}//End if
			else if constexpr (std::is_same<T, float>())
			{
				int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);

				x_as_int ^= (1 << a);
				x_as_int ^= (1 << b);

				x = *reinterpret_cast<float*>(&x_as_int);
			}//End if
			else if constexpr (std::is_same<T, double>())
			{
				int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);

				x_as_int ^= (1ll << a);
				x_as_int ^= (1ll << b);

				x = *reinterpret_cast<double*>(&x_as_int);
			}//End if
			else
			{
				x ^= (1 << a);
				x ^= (1 << b);
			}//End else
		}//End try
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::bit_swap: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::bit_swap: unknown!\n";
		}//End catch
	}//End of bit_swap

	/// <summary>
	/// <para>Rotates bits left in 'x' by 'amount'</para>
	/// <para>Uses hardware instructions where available.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="amount"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T rotate_left(const T& x, const T& amount) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			return (x << amount) | (x >> (CHAR_BIT - amount));
		}//End if

		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
#if defined(__INTEL_COMPILER) && defined(TPA_X86_64)
			return static_cast<T>(_rotwl(static_cast<uint16_t>(x), static_cast<int32_t>(amount)));
#else
			return (x << amount) | (x >> ((sizeof(int16_t)*CHAR_BIT) - amount));
#endif
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
#if defined(TPA_X86_64)
			return _rotl(static_cast<uint32_t>(x), static_cast<int32_t>(amount));
#else
			return (x << amount) | (x >> ((sizeof(int32_t) * CHAR_BIT) - amount));
#endif
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
#if defined(TPA_X86_64)
			return _rotl64(static_cast<uint64_t>(x), static_cast<int32_t>(amount));
#else
			return (x << amount) | (x >> ((sizeof(int64_t) * CHAR_BIT) - amount));
#endif
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
#if defined(TPA_X86_64)
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);
			int32_t a_as_int = *reinterpret_cast<int32_t*>(&amount);
			int32_t result = 0;

			result = _rotl(static_cast<uint32_t>(x_as_int), static_cast<int32_t>(a_as_int));
			return *reinterpret_cast<float*>(&result);
#else
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);
			int32_t a_as_int = *reinterpret_cast<int32_t*>(&amount);
			int32_t result = 0;

			result = (x_as_int << a_as_int) | (x_as_int >> ((sizeof(int32_t) * CHAR_BIT) - a_as_int));
			return *reinterpret_cast<float*>(&result);
#endif
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
#if defined(TPA_X86_64)
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);
			int64_t a_as_int = *reinterpret_cast<int64_t*>(&amount);
			int64_t result = 0;

			result = _rotl64(static_cast<uint64_t>(x_as_int), static_cast<int32_t>(a_as_int));
			return *reinterpret_cast<double*>(&result);
#else
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);
			int64_t a_as_int = *reinterpret_cast<int64_t*>(&amount);
			int64_t result = 0ll;

			result = (x_as_int << a_as_int) | (x_as_int >> ((sizeof(int64_t) * CHAR_BIT) - a_as_int));
			return *reinterpret_cast<double*>(&result);
#endif
		}//End if
		else
		{
			return (x << amount) | (x >> ((sizeof(T) * CHAR_BIT) - amount));
		}//End else
	}//End of rotate_left

	/// <summary>
	/// <para>Rotates bits right in 'x' by 'amount'</para>
	/// <para>Uses hardware instructions where available.</para>
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="x"></param>
	/// <param name="amount"></param>
	/// <returns></returns>
	template<typename T>
	[[nodiscard]] inline constexpr T rotate_right(const T& x, const T& amount) noexcept
	{
		if constexpr (std::is_same<T, int8_t>() || std::is_same<T, uint8_t>())
		{
			return (x >> amount) | (x << (CHAR_BIT - amount));
		}//End if

		else if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
		{
#if defined(__INTEL_COMPILER) && defined(TPA_X86_64)
			return static_cast<T>(_rotwr(static_cast<uint16_t>(x), static_cast<int32_t>(amount)));
#else
			return (x >> amount) | (x << ((sizeof(int16_t) * CHAR_BIT) - amount));
#endif
		}//End if
		else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
		{
#if defined(TPA_X86_64)
			return _rotr(static_cast<uint32_t>(x), static_cast<int32_t>(amount));
#else
			return (x >> amount) | (x << ((sizeof(int32_t) * CHAR_BIT) - amount));
#endif
		}//End if
		else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
		{
#if defined(TPA_X86_64)
			return _rotr64(static_cast<uint64_t>(x), static_cast<int32_t>(amount));
#else
			return (x >> amount) | (x << ((sizeof(int64_t) * CHAR_BIT) - amount));
#endif
		}//End if
		else if constexpr (std::is_same<T, float>())
		{
#if defined(TPA_X86_64)
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);
			int32_t a_as_int = *reinterpret_cast<int32_t*>(&amount);
			int32_t result = 0;

			result = _rotr(static_cast<uint32_t>(x_as_int), static_cast<int32_t>(a_as_int));
			return *reinterpret_cast<float*>(&result);
#else
			int32_t x_as_int = *reinterpret_cast<int32_t*>(&x);
			int32_t a_as_int = *reinterpret_cast<int32_t*>(&amount);
			int32_t result = 0;

			result = (x_as_int >> a_as_int) | (x_as_int << ((sizeof(int32_t) * CHAR_BIT) - a_as_int));
			return *reinterpret_cast<float*>(&result);
#endif
		}//End if
		else if constexpr (std::is_same<T, double>())
		{
#if defined(TPA_X86_64)
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);
			int64_t a_as_int = *reinterpret_cast<int64_t*>(&amount);
			int64_t result = 0;

			result = _rotr64(static_cast<uint64_t>(x_as_int), static_cast<int32_t>(a_as_int));
			return *reinterpret_cast<double*>(&result);
#else
			int64_t x_as_int = *reinterpret_cast<int64_t*>(&x);
			int64_t a_as_int = *reinterpret_cast<int64_t*>(&amount);
			int64_t result = 0ll;

			result = (x_as_int >> a_as_int) | (x_as_int << ((sizeof(int64_t) * CHAR_BIT) - a_as_int));
			return *reinterpret_cast<double*>(&result);
#endif
		}//End if
		else
		{
			return (x >> amount) | (x << ((sizeof(T) * CHAR_BIT) - amount));
		}//End else
	}//End of rotate_right

}//End of namespace

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>SIMD Matrix Math Functions.</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa{

namespace bit_manip 
{
	/// <summary>
	/// <para>Modifies the bits in a numeric type according to the specified instruction at the specified position.</para>
	/// <para>The position must be with in the bounds of a type. EX: Bounds of int32_t = Bit 0 to Bit 31.</para>
	/// <para>Note: tpa::fill may perform better than tpa::bit_modify with tpa::bit_mod::SET_ALL and tpa:bit_mod::CLEAR_ALL</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <param name="x"></param>
	/// <param name="pos"></param>
	/// <returns></returns>
	template<tpa::bit_mod INSTR, typename CONTAINER_A>
	inline constexpr void bit_modify(CONTAINER_A& source, const uint64_t pos = 0) requires tpa::util::contiguous_seqeunce<CONTAINER_A>
	{
		uint32_t complete = 0u;
		using T = CONTAINER_A::value_type;

		try
		{
			//Check Bounds
			if (pos < 0ull || pos > static_cast<uint64_t>((static_cast<uint64_t>(sizeof(T)) * static_cast<uint64_t>(CHAR_BIT)) - 1ull))
			{
				throw std::out_of_range("Position must be within the bounds of T");
			}//End if

			std::vector<std::pair<size_t, size_t>> sections;
			tpa::util::prepareThreading(sections, source.size());

			std::vector<std::shared_future<uint32_t>> results;
			results.reserve(tpa::nThreads);

			std::shared_future<uint32_t> temp;

			for (const auto& sec : sections)
			{
				//Launch lambda from multiple threads
				temp = tpa::tp->addTask([&source, &pos, &sec]()
				{				
					const size_t beg = sec.first;
					const size_t end = sec.second;
					size_t i = beg;					

#pragma region short
					if constexpr (std::is_same<T, int16_t>() || std::is_same<T, uint16_t>())
					{						
#ifdef TPA_X86_64
						if (tpa::hasAVX512_ByteWord)
						{
							const uint32_t p = static_cast<uint32_t>(pos);

							__m512i _source = _mm512_setzero_si512(); 
							__m512i _DESTi = _mm512_setzero_si512();		
							__m512i _temp = _mm512_setzero_si512();
							const __m512i _zero = _mm512_setzero_si512();
							const __m512i _one = _mm512_set1_epi16(static_cast<int16_t>(1));
							const __m512i _max = _mm512_set1_epi16(std::numeric_limits<uint16_t>::max());
							const __m512i _shifted_left = _mm512_slli_epi16(_one, p);
							const __m512i _not_shifted_left = tpa::simd::_mm512_not_si512(_shifted_left);
							
							for (; (i + 32uz) < end; i += 32uz)
							{
								//Set Values
								_source = _mm512_loadu_epi16(&source[i]);
																
								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm512_or_si512(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									//~(1ull << pos) & source[i]
									_DESTi = _mm512_and_si512(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm512_xor_si512(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm512_not_si512(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while(bits > 0uz)
									{										
										_DESTi = _mm512_or_si512(_mm512_slli_epi16(_DESTi, 1u), _mm512_and_si512(_source, _one));
										_source = _mm512_srli_epi16(_source, 1u);
										bits -= 1uz;
									}//End while
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm512_or_si512(_mm512_sub_epi16(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm512_and_si512(_mm512_add_epi16(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm512_lzset_epi16(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm512_not_si512(tpa::simd::_mm512_lzset_epi16(tpa::simd::_mm512_not_si512(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm512_or_si512(_source, _mm512_sub_epi16(_source, _one));

									__m512i _rhs = _mm512_add_epi16(tpa::simd::_mm512_bsf_epi16(_source), _one);
									   
									__m512i _added = _mm512_add_epi16(_temp, _one);
									__m512i _not_temp = tpa::simd::_mm512_not_si512(_temp);
									__m512i _lhs = _mm512_sub_epi16(_mm512_and_si512(_not_temp, _mm512_sub_epi16(_zero, _not_temp)), _one);
									__m512i _lhs_shifted_right = _mm512_srlv_epi16(_lhs, _rhs);

									_DESTi = _mm512_or_si512(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m512i _EVEN = _mm512_set1_epi16(static_cast<T>(0b0101'0101'0101'0101));

									_DESTi = _mm512_or_si512(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m512i _ODD = _mm512_set1_epi16(static_cast<T>(0b1010'1010'1010'1010));

									_DESTi = _mm512_or_si512(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m512i _EVEN = _mm512_set1_epi16(static_cast<T>(0b0101'0101'0101'0101));

									_DESTi = _mm512_and_si512(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m512i _ODD = _mm512_set1_epi16(static_cast<T>(0b1010'1010'1010'1010));

									_DESTi = _mm512_and_si512(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int16_t x = static_cast<int16_t>(0); x != static_cast<int16_t>(16); x += static_cast<int16_t>(2))
									{
										_DESTi = _mm512_xor_si512(_DESTi, _mm512_slli_epi16(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int16_t x = static_cast<int16_t>(1); x != static_cast<int16_t>(17); x += static_cast<int16_t>(2))
									{
										_DESTi = _mm512_xor_si512(_DESTi, _mm512_slli_epi16(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m512i _set_msb = _mm512_set1_epi16(static_cast<int16_t>(0b1000'0000'0000'0000));
									_DESTi = _mm512_or_si512(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m512i _clear_msb = _mm512_set1_epi16(static_cast<int16_t>(0b0111'1111'1111'1111));
									_DESTi = _mm512_and_si512(_source, _clear_msb);                         
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m512i _clear_lsb = _mm512_set1_epi16(static_cast<int16_t>(0b1111'1111'1111'1110));
									_DESTi = _mm512_and_si512(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m512i _clear_lsb = _mm512_set1_epi16(static_cast<int16_t>(0b0000'0000'0000'0001));
									_DESTi = _mm512_or_si512(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm512_xor_si512(_source, _mm512_and_si512(_source, _mm512_sub_epi16(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm512_xor_si512(_source, tpa::simd::_mm512_exthsb_epi16(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m512i _hsb = tpa::simd::_mm512_exthsb_epi16(_source);
									const __m512i _sl = _mm512_slli_epi16(_hsb, 1u);
									const __mmask32 _mask = _mm512_cmplt_epi16_mask(_sl, _max);

									_DESTi = _mm512_or_si512(_source, _mm512_mask_slli_epi16(_hsb, _mask, _hsb, 1u));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m512i _lowest = _mm512_and_si512(_source, _mm512_sub_epi16(_zero, _source));
									_lowest = _mm512_srli_epi16(_lowest, 1u);
									const __mmask32 _mask = _mm512_cmpgt_epi16_mask(_lowest, _zero);

									_DESTi = _mm512_or_si512(_source, _mm512_mask_slli_epi16(_lowest, _mask, _lowest, 1u));
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t> / uint16_t).");
									}();
								}//End else

								//Store Result
								_mm512_storeu_epi16(&source[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							const int32_t p = static_cast<int32_t>(pos);							

							__m256i _source = _mm256_setzero_si256();
							__m256i _DESTi = _mm256_setzero_si256();
							__m256i _temp = _mm256_setzero_si256();
							const __m256i _zero = _mm256_setzero_si256();
							const __m256i _one = _mm256_set1_epi16(static_cast<int16_t>(1));
							const __m256i _max = _mm256_set1_epi16(std::numeric_limits<uint16_t>::max());
							const __m256i _shifted_left = _mm256_slli_epi16(_one, p);	
							const __m256i _not_shifted_left = tpa::simd::_mm256_not_si256(_shifted_left);

							for (; (i + 16uz) < end; i += 16uz)
							{
								//Set Values
								_source = _mm256_load_si256((__m256i*) &source[i]);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm256_or_si256(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									//~(1ull << pos) & source[i]
									_DESTi = _mm256_and_si256(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm256_xor_si256(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm256_not_si256(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0uz)
									{
										_DESTi = _mm256_or_si256(_mm256_slli_epi16(_DESTi, 1), _mm256_and_si256(_source, _one));
										_source = _mm256_srli_epi16(_source, 1);
										bits -= 1uz;
									}//End while
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm256_or_si256(_mm256_sub_epi16(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm256_and_si256(_mm256_add_epi16(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm256_lzset_epi16(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm256_not_si256(tpa::simd::_mm256_lzset_epi16(tpa::simd::_mm256_not_si256(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm256_or_si256(_source, _mm256_sub_epi16(_source, _one));

									__m256i _rhs = _mm256_add_epi16(tpa::simd::_mm256_bsf_epi16(_source), _one);

									__m256i _added = _mm256_add_epi16(_temp, _one);									
									__m256i _not_temp = tpa::simd::_mm256_not_si256(_temp);
									__m256i _lhs = _mm256_sub_epi16(_mm256_and_si256(_not_temp, _mm256_sub_epi16(_zero, _not_temp)), _one);
									__m256i _lhs_shifted_right = tpa::simd::_mm256_srlv_epi16(_lhs, _rhs);

									_DESTi = _mm256_or_si256(_added, _lhs_shifted_right);	
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m256i _EVEN = _mm256_set1_epi16(static_cast<T>(0b0101'0101'0101'0101));

									_DESTi = _mm256_or_si256(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m256i _ODD = _mm256_set1_epi16(static_cast<T>(0b1010'1010'1010'1010));

									_DESTi = _mm256_or_si256(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m256i _EVEN = _mm256_set1_epi16(static_cast<T>(0b0101'0101'0101'0101));

									_DESTi = _mm256_and_si256(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m256i _ODD = _mm256_set1_epi16(static_cast<T>(0b1010'1010'1010'1010));

									_DESTi = _mm256_and_si256(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int16_t x = static_cast<int16_t>(0); x != static_cast<int16_t>(16); x += static_cast<int16_t>(2))
									{
										_DESTi = _mm256_xor_si256(_DESTi, _mm256_slli_epi16(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int16_t x = static_cast<int16_t>(1); x != static_cast<int16_t>(17); x += static_cast<int16_t>(2))
									{
										_DESTi = _mm256_xor_si256(_DESTi, _mm256_slli_epi16(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m256i _set_msb = _mm256_set1_epi16(static_cast<int16_t>(0b1000'0000'0000'0000));
									_DESTi = _mm256_or_si256(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m256i _clear_msb = _mm256_set1_epi16(static_cast<int16_t>(0b0111'1111'1111'1111));
									_DESTi = _mm256_and_si256(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m256i _clear_lsb = _mm256_set1_epi16(static_cast<int16_t>(0b1111'1111'1111'1110));
									_DESTi = _mm256_and_si256(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m256i _clear_lsb = _mm256_set1_epi16(static_cast<int16_t>(0b0000'0000'0000'0001));
									_DESTi = _mm256_or_si256(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm256_xor_si256(_source, _mm256_and_si256(_source, _mm256_sub_epi16(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm256_xor_si256(_source, tpa::simd::_mm256_exthsb_epi16(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m256i _hsb = tpa::simd::_mm256_exthsb_epi16(_source);
									const __m256i _sl = _mm256_slli_epi16(_hsb, 1);
									__m256i _mask = _mm256_cmpgt_epi16(_max, _sl);
									_mask = _mm256_and_si256(_sl, _mask);

									_DESTi = _mm256_or_si256(_source, _mask);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m256i _lowest = _mm256_and_si256(_source, _mm256_sub_epi16(_zero, _source));
									_lowest = _mm256_srli_epi16(_lowest, 1);

									__m256i _mask = _mm256_cmpgt_epi16(_lowest, _zero);
									_mask = _mm256_and_si256(_lowest, _mask);

									_DESTi = _mm256_or_si256(_source, _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t / uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*)&source[i], _DESTi);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							const int32_t p = static_cast<int32_t>(pos);

							__m128i _source = _mm_setzero_si128();
							__m128i _DESTi = _mm_setzero_si128();
							__m128i _temp = _mm_setzero_si128();
							const __m128i _zero = _mm_setzero_si128();
							const __m128i _one = _mm_set1_epi16(static_cast<int16_t>(1));
							const __m128i _max = _mm_set1_epi16(std::numeric_limits<uint16_t>::max());
							const __m128i _shifted_left = _mm_slli_epi16(_one, p);	
							const __m128i _not_shifted_left = tpa::simd::_mm_not_si128(_shifted_left);

							for (; (i + 8uz) < end; i += 8uz)
							{
								//Set Values
								_source = _mm_load_si128((__m128i*) & source[i]);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm_or_si128(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									//~(1ull << pos) & source[i]
									_DESTi = _mm_and_si128(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm_xor_si128(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm_not_si128(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while(bits > 0)
									{										
										_DESTi = _mm_or_si128(_mm_slli_epi16(_DESTi, 1), _mm_and_si128(_source, _one));
										_source = _mm_srli_epi16(_source, 1);
										bits -= 1;
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm_or_si128(_mm_sub_epi16(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm_and_si128(_mm_add_epi16(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm_lzset_epi16(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm_not_si128(tpa::simd::_mm_lzset_epi16(tpa::simd::_mm_not_si128(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm_or_si128(_source, _mm_sub_epi16(_source, _one));

									__m128i _rhs = _mm_add_epi16(tpa::simd::_mm_bsf_epi16(_source), _one);
									   
									__m128i _added = _mm_add_epi16(_temp, _one);
									__m128i _not_temp = tpa::simd::_mm_not_si128(_temp);
									__m128i _lhs = _mm_sub_epi16(_mm_and_si128(_not_temp, _mm_sub_epi16(_zero, _not_temp)), _one);
									__m128i _lhs_shifted_right = tpa::simd::_mm_srlv_epi16(_lhs, _rhs);

									_DESTi = _mm_or_si128(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m128i _EVEN = _mm_set1_epi16(static_cast<T>(0b0101'0101'0101'0101));

									_DESTi = _mm_or_si128(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m128i _ODD = _mm_set1_epi16(static_cast<T>(0b1010'1010'1010'1010));

									_DESTi = _mm_or_si128(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m128i _EVEN = _mm_set1_epi16(static_cast<T>(0b0101'0101'0101'0101));

									_DESTi = _mm_and_si128(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m128i _ODD = _mm_set1_epi16(static_cast<T>(0b1010'1010'1010'1010));

									_DESTi = _mm_and_si128(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int16_t x = static_cast<int16_t>(0); x != static_cast<int16_t>(16); x += static_cast<int16_t>(2))
									{
										_DESTi = _mm_xor_si128(_DESTi, _mm_slli_epi16(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int16_t x = static_cast<int16_t>(1); x != static_cast<int16_t>(17); x += static_cast<int16_t>(2))
									{
										_DESTi = _mm_xor_si128(_DESTi, _mm_slli_epi16(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m128i _set_msb = _mm_set1_epi16(static_cast<int16_t>(0b1000'0000'0000'0000));
									_DESTi = _mm_or_si128(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m128i _clear_msb = _mm_set1_epi16(static_cast<int16_t>(0b0111'1111'1111'1111));
									_DESTi = _mm_and_si128(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m128i _clear_lsb = _mm_set1_epi16(static_cast<int16_t>(0b1111'1111'1111'1110));
									_DESTi = _mm_and_si128(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m128i _clear_lsb = _mm_set1_epi16(static_cast<int16_t>(0b0000'0000'0000'0001));
									_DESTi = _mm_or_si128(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm_xor_si128(_source, _mm_and_si128(_source, _mm_sub_epi16(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm_xor_si128(_source, tpa::simd::_mm_exthsb_epi16(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m128i _hsb = tpa::simd::_mm_exthsb_epi16(_source);
									const __m128i _sl = _mm_slli_epi16(_hsb, 1);
									__m128i _mask = _mm_cmpgt_epi16(_max, _sl);
									_mask = _mm_and_si128(_sl, _mask);

									_DESTi = _mm_or_si128(_source, _mask);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m128i _lowest = _mm_and_si128(_source, _mm_sub_epi16(_zero, _source));
									_lowest = _mm_srli_epi16(_lowest, 1);

									__m128i _mask = _mm_cmpgt_epi16(_lowest, _zero);
									_mask = _mm_and_si128(_lowest, _mask);

									_DESTi = _mm_or_si128(_source, _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t / uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & source[i], _DESTi);
							}//End for
						}//End if has SSE2
#endif
					}//End if
#pragma endregion
#pragma region int
					else if constexpr (std::is_same<T, int32_t>() || std::is_same<T, uint32_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							const uint32_t p = static_cast<uint32_t>(pos);

							__m512i _source = _mm512_setzero_si512();
							__m512i _DESTi = _mm512_setzero_si512();
							__m512i _temp = _mm512_setzero_si512();
							const __m512i _zero = _mm512_setzero_si512();
							const __m512i _one = _mm512_set1_epi32(1);
							const __m512i _max = _mm512_set1_epi32(std::numeric_limits<uint32_t>::max());
							const __m512i _shifted_left = _mm512_slli_epi32(_one, p);
							const __m512i _not_shifted_left = tpa::simd::_mm512_not_si512(_shifted_left);

							for (; (i + 16uz) < end; i += 16uz)
							{
								//Set Values
								_source = _mm512_load_epi32(&source[i]);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm512_or_si512(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									//~(1ull << pos) & source[i]
									_DESTi = _mm512_and_si512(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm512_xor_si512(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm512_not_si512(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0uz)
									{
										_DESTi = _mm512_or_si512(_mm512_slli_epi32(_DESTi, 1u), _mm512_and_si512(_source, _one));
										_source = _mm512_srli_epi32(_source, 1u);
										bits -= 1uz;
									}//End while
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm512_or_si512(_mm512_sub_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm512_and_si512(_mm512_add_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm512_lzset_epi32(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm512_not_si512(tpa::simd::_mm512_lzset_epi32(tpa::simd::_mm512_not_si512(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm512_or_si512(_source, _mm512_sub_epi32(_source, _one));

									__m512i _rhs = _mm512_add_epi32(tpa::simd::_mm512_bsf_epi32(_source), _one);

									__m512i _added = _mm512_add_epi32(_temp, _one);
									__m512i _not_temp = tpa::simd::_mm512_not_si512(_temp);
									__m512i _lhs = _mm512_sub_epi32(_mm512_and_si512(_not_temp, _mm512_sub_epi32(_zero, _not_temp)), _one);
									__m512i _lhs_shifted_right = _mm512_srlv_epi32(_lhs, _rhs);

									_DESTi = _mm512_or_si512(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m512i _EVEN = _mm512_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm512_or_si512(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m512i _ODD = _mm512_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm512_or_si512(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m512i _EVEN = _mm512_set1_epi32(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm512_and_si512(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m512i _ODD = _mm512_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm512_and_si512(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(0); x != static_cast<int32_t>(32); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm512_xor_si512(_DESTi, _mm512_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(1); x != static_cast<int32_t>(33); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm512_xor_si512(_DESTi, _mm512_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m512i _set_msb = _mm512_set1_epi32(static_cast<int32_t>(0b1000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm512_or_si512(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m512i _clear_msb = _mm512_set1_epi32(static_cast<int32_t>(0b0111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm512_and_si512(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m512i _clear_lsb = _mm512_set1_epi32(static_cast<int32_t>(0b1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm512_and_si512(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m512i _clear_lsb = _mm512_set1_epi32(static_cast<int32_t>(0b0000'0000'0000'0000'0000'0000'0000'0001));
									_DESTi = _mm512_or_si512(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm512_xor_si512(_source, _mm512_and_si512(_source, _mm512_sub_epi32(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm512_xor_si512(_source, tpa::simd::_mm512_exthsb_epi32(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m512i _hsb = tpa::simd::_mm512_exthsb_epi32(_source);
									const __m512i _sl = _mm512_slli_epi32(_hsb, 1u);
									const __mmask16 _mask = _mm512_cmplt_epi16_mask(_sl, _max);

									_DESTi = _mm512_or_si512(_source, _mm512_mask_slli_epi32(_hsb, _mask, _hsb, 1u));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m512i _lowest = _mm512_and_si512(_source, _mm512_sub_epi32(_zero, _source));
									_lowest = _mm512_srli_epi32(_lowest, 1u);
									const __mmask16 _mask = _mm512_cmpgt_epi32_mask(_lowest, _zero);

									_DESTi = _mm512_or_si512(_source, _mm512_mask_slli_epi32(_lowest, _mask, _lowest, 1u));
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t> / uint32_t).");
									}();
								}//End else

								//Store Result
								_mm512_store_epi32(&source[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							const int32_t p = static_cast<int32_t>(pos);

							__m256i _source = _mm256_setzero_si256();
							__m256i _DESTi = _mm256_setzero_si256();
							__m256i _temp = _mm256_setzero_si256();
							const __m256i _zero = _mm256_setzero_si256();
							const __m256i _one = _mm256_set1_epi32(1);
							const __m256i _max = _mm256_set1_epi32(std::numeric_limits<uint32_t>::max());
							const __m256i _shifted_left = _mm256_slli_epi32(_one, p);
							const __m256i _not_shifted_left = tpa::simd::_mm256_not_si256(_shifted_left);

							for (; (i + 8uz) < end; i += 8uz)
							{
								//Set Values
								_source = _mm256_load_si256((__m256i*) & source[i]);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm256_or_si256(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									//~(1ull << pos) & source[i]
									_DESTi = _mm256_and_si256(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm256_xor_si256(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm256_not_si256(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0uz)
									{
										_DESTi = _mm256_or_si256(_mm256_slli_epi32(_DESTi, 1), _mm256_and_si256(_source, _one));
										_source = _mm256_srli_epi32(_source, 1);
										bits -= 1uz;
									}//End while
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm256_or_si256(_mm256_sub_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm256_and_si256(_mm256_add_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm256_lzset_epi32(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm256_not_si256(tpa::simd::_mm256_lzset_epi32(tpa::simd::_mm256_not_si256(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm256_or_si256(_source, _mm256_sub_epi32(_source, _one));

									__m256i _rhs = _mm256_add_epi32(tpa::simd::_mm256_bsf_epi32(_source), _one);

									__m256i _added = _mm256_add_epi32(_temp, _one);
									__m256i _not_temp = tpa::simd::_mm256_not_si256(_temp);
									__m256i _lhs = _mm256_sub_epi32(_mm256_and_si256(_not_temp, _mm256_sub_epi32(_zero, _not_temp)), _one);
									__m256i _lhs_shifted_right = _mm256_srlv_epi32(_lhs, _rhs);

									_DESTi = _mm256_or_si256(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m256i _EVEN = _mm256_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm256_or_si256(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m256i _ODD = _mm256_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm256_or_si256(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m256i _EVEN = _mm256_set1_epi32(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm256_and_si256(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m256i _ODD = _mm256_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm256_and_si256(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(0); x != static_cast<int32_t>(32); x += static_cast<int16_t>(2))
									{
										_DESTi = _mm256_xor_si256(_DESTi, _mm256_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(1); x != static_cast<int32_t>(33); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm256_xor_si256(_DESTi, _mm256_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m256i _set_msb = _mm256_set1_epi32(static_cast<int16_t>(0b1000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm256_or_si256(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m256i _clear_msb = _mm256_set1_epi32(static_cast<int16_t>(0b0111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm256_and_si256(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m256i _clear_lsb = _mm256_set1_epi32(static_cast<int16_t>(0b1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm256_and_si256(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m256i _clear_lsb = _mm256_set1_epi32(static_cast<int16_t>(0b0000'0000'0000'0001'0000'0000'0000'0001));
									_DESTi = _mm256_or_si256(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm256_xor_si256(_source, _mm256_and_si256(_source, _mm256_sub_epi32(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm256_xor_si256(_source, tpa::simd::_mm256_exthsb_epi32(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m256i _hsb = tpa::simd::_mm256_exthsb_epi32(_source);
									const __m256i _sl = _mm256_slli_epi32(_hsb, 1);
									__m256i _mask = _mm256_cmpgt_epi32(_max, _sl);
									_mask = _mm256_and_si256(_sl, _mask);

									_DESTi = _mm256_or_si256(_source, _mask);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m256i _lowest = _mm256_and_si256(_source, _mm256_sub_epi32(_zero, _source));
									_lowest = _mm256_srli_epi32(_lowest, 1);

									__m256i _mask = _mm256_cmpgt_epi32(_lowest, _zero);
									_mask = _mm256_and_si256(_lowest, _mask);

									_DESTi = _mm256_or_si256(_source, _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t / uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*) & source[i], _DESTi);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							const int32_t p = static_cast<int32_t>(pos);

							__m128i _source = _mm_setzero_si128();
							__m128i _DESTi = _mm_setzero_si128();
							__m128i _temp = _mm_setzero_si128();
							const __m128i _zero = _mm_setzero_si128();
							const __m128i _one = _mm_set1_epi32(1);
							const __m128i _max = _mm_set1_epi32(std::numeric_limits<uint32_t>::max());
							const __m128i _shifted_left = _mm_slli_epi32(_one, p);
							const __m128i _not_shifted_left = tpa::simd::_mm_not_si128(_shifted_left);

							for (; (i + 4uz) < end; i += 4uz)
							{
								//Set Values
								_source = _mm_load_si128((__m128i*) & source[i]);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm_or_si128(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									_DESTi = _mm_and_si128(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm_xor_si128(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm_not_si128(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0)
									{
										_DESTi = _mm_or_si128(_mm_slli_epi32(_DESTi, 1), _mm_and_si128(_source, _one));
										_source = _mm_srli_epi32(_source, 1);
										bits -= 1;
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm_or_si128(_mm_sub_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm_and_si128(_mm_add_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm_lzset_epi32(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm_not_si128(tpa::simd::_mm_lzset_epi32(tpa::simd::_mm_not_si128(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm_or_si128(_source, _mm_sub_epi32(_source, _one));

									__m128i _rhs = _mm_add_epi32(tpa::simd::_mm_bsf_epi32(_source), _one);

									__m128i _added = _mm_add_epi32(_temp, _one);
									__m128i _not_temp = tpa::simd::_mm_not_si128(_temp);
									__m128i _lhs = _mm_sub_epi32(_mm_and_si128(_not_temp, _mm_sub_epi32(_zero, _not_temp)), _one);
									__m128i _lhs_shifted_right = _mm_srlv_epi32(_lhs, _rhs);

									_DESTi = _mm_or_si128(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m128i _EVEN = _mm_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm_or_si128(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m128i _ODD = _mm_set1_epi32(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm_or_si128(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m128i _EVEN = _mm_set1_epi32(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm_and_si128(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m128i _ODD = _mm_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm_and_si128(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(0); x != static_cast<int32_t>(32); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm_xor_si128(_DESTi, _mm_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(1); x != static_cast<int32_t>(33); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm_xor_si128(_DESTi, _mm_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m128i _set_msb = _mm_set1_epi32(static_cast<int32_t>(0b1000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm_or_si128(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m128i _clear_msb = _mm_set1_epi32(static_cast<int32_t>(0b0111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm_and_si128(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m128i _clear_lsb = _mm_set1_epi32(static_cast<int32_t>(0b1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm_and_si128(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m128i _clear_lsb = _mm_set1_epi32(static_cast<int32_t>(0b0000'0000'0000'0000'0000'0000'0000'0001));
									_DESTi = _mm_or_si128(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm_xor_si128(_source, _mm_and_si128(_source, _mm_sub_epi32(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm_xor_si128(_source, tpa::simd::_mm_exthsb_epi32(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m128i _hsb = tpa::simd::_mm_exthsb_epi32(_source);
									const __m128i _sl = _mm_slli_epi32(_hsb, 1);
									__m128i _mask = _mm_cmpgt_epi32(_max, _sl);
									_mask = _mm_and_si128(_sl, _mask);

									_DESTi = _mm_or_si128(_source, _mask);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m128i _lowest = _mm_and_si128(_source, _mm_sub_epi32(_zero, _source));
									_lowest = _mm_srli_epi32(_lowest, 1);

									__m128i _mask = _mm_cmpgt_epi32(_lowest, _zero);
									_mask = _mm_and_si128(_lowest, _mask);

									_DESTi = _mm_or_si128(_source, _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t / uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & source[i], _DESTi);
							}//End for
						}//End if has SSE2
#endif
					}//End if
#pragma endregion 
#pragma region long
					else if constexpr (std::is_same<T, int64_t>() || std::is_same<T, uint64_t>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							const uint32_t p = static_cast<uint32_t>(pos);

							__m512i _source = _mm512_setzero_si512();
							__m512i _DESTi = _mm512_setzero_si512();
							__m512i _temp = _mm512_setzero_si512();
							const __m512i _zero = _mm512_setzero_si512();
							const __m512i _one = _mm512_set1_epi64(1ull);
							const __m512i _max = _mm512_set1_epi64(std::numeric_limits<uint64_t>::max());
							const __m512i _shifted_left = _mm512_slli_epi64(_one, p);
							const __m512i _not_shifted_left = tpa::simd::_mm512_not_si512(_shifted_left);

							for (; (i + 8uz) < end; i += 8uz)
							{
								//Set Values
								_source = _mm512_load_epi64(&source[i]);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm512_or_si512(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									_DESTi = _mm512_and_si512(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm512_xor_si512(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm512_not_si512(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0uz)
									{
										_DESTi = _mm512_or_si512(_mm512_slli_epi64(_DESTi, 1u), _mm512_and_si512(_source, _one));
										_source = _mm512_srli_epi64(_source, 1u);
										bits -= 1uz;
									}//End while
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm512_or_si512(_mm512_sub_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm512_and_si512(_mm512_add_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm512_lzset_epi64(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm512_not_si512(tpa::simd::_mm512_lzset_epi64(tpa::simd::_mm512_not_si512(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm512_or_si512(_source, _mm512_sub_epi64(_source, _one));

									__m512i _rhs = _mm512_add_epi64(tpa::simd::_mm512_bsf_epi64(_source), _one);

									__m512i _added = _mm512_add_epi64(_temp, _one);
									__m512i _not_temp = tpa::simd::_mm512_not_si512(_temp);
									__m512i _lhs = _mm512_sub_epi64(_mm512_and_si512(_not_temp, _mm512_sub_epi64(_zero, _not_temp)), _one);
									__m512i _lhs_shifted_right = _mm512_srlv_epi64(_lhs, _rhs);

									_DESTi = _mm512_or_si512(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m512i _EVEN = _mm512_set1_epi64(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm512_or_si512(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m512i _ODD = _mm512_set1_epi64(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm512_or_si512(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m512i _EVEN = _mm512_set1_epi64(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm512_and_si512(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m512i _ODD = _mm512_set1_epi64(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm512_and_si512(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int64_t x = static_cast<int64_t>(0); x != static_cast<int64_t>(64); x += static_cast<int64_t>(2))
									{
										_DESTi = _mm512_xor_si512(_DESTi, _mm512_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int64_t x = static_cast<int64_t>(1); x != static_cast<int64_t>(65); x += static_cast<int64_t>(2))
									{
										_DESTi = _mm512_xor_si512(_DESTi, _mm512_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m512i _set_msb = _mm512_set1_epi64(static_cast<int64_t>(0b1000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm512_or_si512(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m512i _clear_msb = _mm512_set1_epi64(static_cast<int64_t>(0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm512_and_si512(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m512i _clear_lsb = _mm512_set1_epi64(static_cast<int64_t>(0b1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm512_and_si512(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m512i _clear_lsb = _mm512_set1_epi64(static_cast<int64_t>(0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001));
									_DESTi = _mm512_or_si512(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm512_xor_si512(_source, _mm512_and_si512(_source, _mm512_sub_epi64(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm512_xor_si512(_source, tpa::simd::_mm512_exthsb_epi64(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m512i _hsb = tpa::simd::_mm512_exthsb_epi64(_source);
									const __m512i _sl = _mm512_slli_epi64(_hsb, 1u);
									const __mmask8 _mask = _mm512_cmplt_epi64_mask(_sl, _max);

									_DESTi = _mm512_or_si512(_source, _mm512_mask_slli_epi64(_hsb, _mask, _hsb, 1u));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m512i _lowest = _mm512_and_si512(_source, _mm512_sub_epi64(_zero, _source));
									_lowest = _mm512_srli_epi64(_lowest, 1u);
									const __mmask8 _mask = _mm512_cmpgt_epi64_mask(_lowest, _zero);

									_DESTi = _mm512_or_si512(_source, _mm512_mask_slli_epi64(_lowest, _mask, _lowest, 1u));
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t> / uint32_t).");
									}();
								}//End else

								//Store Result
								_mm512_store_epi64(&source[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							const int32_t p = static_cast<int32_t>(pos);

							__m256i _source = _mm256_setzero_si256();
							__m256i _DESTi = _mm256_setzero_si256();
							__m256i _temp = _mm256_setzero_si256();
							const __m256i _zero = _mm256_setzero_si256();
							const __m256i _one = _mm256_set1_epi64x(1ll);
							const __m256i _max = _mm256_set1_epi64x(std::numeric_limits<uint64_t>::max());
							const __m256i _shifted_left = _mm256_slli_epi64(_one, p);
							const __m256i _not_shifted_left = tpa::simd::_mm256_not_si256(_shifted_left);

							for (; (i + 4uz) < end; i += 4uz)
							{
								//Set Values
								_source = _mm256_load_si256((__m256i*) & source[i]);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm256_or_si256(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									_DESTi = _mm256_and_si256(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm256_xor_si256(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm256_not_si256(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0uz)
									{
										_DESTi = _mm256_or_si256(_mm256_slli_epi64(_DESTi, 1), _mm256_and_si256(_source, _one));
										_source = _mm256_srli_epi64(_source, 1);
										bits -= 1uz;
									}//End while
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm256_or_si256(_mm256_sub_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm256_and_si256(_mm256_add_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm256_lzset_epi64(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm256_not_si256(tpa::simd::_mm256_lzset_epi64(tpa::simd::_mm256_not_si256(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm256_or_si256(_source, _mm256_sub_epi64(_source, _one));

									__m256i _rhs = _mm256_add_epi64(tpa::simd::_mm256_bsf_epi64(_source), _one);

									__m256i _added = _mm256_add_epi64(_temp, _one);
									__m256i _not_temp = tpa::simd::_mm256_not_si256(_temp);
									__m256i _lhs = _mm256_sub_epi64(_mm256_and_si256(_not_temp, _mm256_sub_epi64(_zero, _not_temp)), _one);
									__m256i _lhs_shifted_right = _mm256_srlv_epi64(_lhs, _rhs);

									_DESTi = _mm256_or_si256(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m256i _EVEN = _mm256_set1_epi64x(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm256_or_si256(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m256i _ODD = _mm256_set1_epi64x(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm256_or_si256(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m256i _EVEN = _mm256_set1_epi64x(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm256_and_si256(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m256i _ODD = _mm256_set1_epi64x(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm256_and_si256(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int64_t x = static_cast<int32_t>(0); x != static_cast<int32_t>(64); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm256_xor_si256(_DESTi, _mm256_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(1); x != static_cast<int32_t>(65); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm256_xor_si256(_DESTi, _mm256_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m256i _set_msb = _mm256_set1_epi64x(static_cast<int16_t>(0b1000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm256_or_si256(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m256i _clear_msb = _mm256_set1_epi64x(static_cast<int16_t>(0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm256_and_si256(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m256i _clear_lsb = _mm256_set1_epi64x(static_cast<int16_t>(0b1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm256_and_si256(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m256i _clear_lsb = _mm256_set1_epi64x(static_cast<int16_t>(0b0000'0000'0000'0001'0000'0000'0000'0000'0000'0000'0000'0001'0000'0000'0000'0001));
									_DESTi = _mm256_or_si256(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm256_xor_si256(_source, _mm256_and_si256(_source, _mm256_sub_epi64(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm256_xor_si256(_source, tpa::simd::_mm256_exthsb_epi64(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m256i _hsb = tpa::simd::_mm256_exthsb_epi64(_source);
									const __m256i _sl = _mm256_slli_epi64(_hsb, 1);
									__m256i _mask = _mm256_cmpgt_epi64(_max, _sl);
									_mask = _mm256_and_si256(_sl, _mask);

									_DESTi = _mm256_or_si256(_source, _mask);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m256i _lowest = _mm256_and_si256(_source, _mm256_sub_epi64(_zero, _source));
									_lowest = _mm256_srli_epi64(_lowest, 1);

									__m256i _mask = _mm256_cmpgt_epi64(_lowest, _zero);
									_mask = _mm256_and_si256(_lowest, _mask);

									_DESTi = _mm256_or_si256(_source, _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t / uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_si256((__m256i*) & source[i], _DESTi);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE42)
						{
							const int32_t p = static_cast<int32_t>(pos);

							__m128i _source = _mm_setzero_si128();
							__m128i _DESTi = _mm_setzero_si128();
							__m128i _temp = _mm_setzero_si128();
							const __m128i _zero = _mm_setzero_si128();
							const __m128i _one = _mm_set1_epi64x(1ll);
							const __m128i _max = _mm_set1_epi64x(std::numeric_limits<uint64_t>::max());
							const __m128i _shifted_left = _mm_slli_epi64(_one, p);
							const __m128i _not_shifted_left = tpa::simd::_mm_not_si128(_shifted_left);

							for (; (i + 2uz) < end; i += 2uz)
							{
								//Set Values
								_source = _mm_load_si128((__m128i*) & source[i]);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm_or_si128(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									_DESTi = _mm_and_si128(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm_xor_si128(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm_not_si128(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0)
									{
										_DESTi = _mm_or_si128(_mm_slli_epi64(_DESTi, 1), _mm_and_si128(_source, _one));
										_source = _mm_srli_epi64(_source, 1);
										bits -= 1;
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm_or_si128(_mm_sub_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm_and_si128(_mm_add_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm_lzset_epi64(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm_not_si128(tpa::simd::_mm_lzset_epi64(tpa::simd::_mm_not_si128(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm_or_si128(_source, _mm_sub_epi64(_source, _one));

									__m128i _rhs = _mm_add_epi32(tpa::simd::_mm_bsf_epi32(_source), _one);

									__m128i _added = _mm_add_epi64(_temp, _one);
									__m128i _not_temp = tpa::simd::_mm_not_si128(_temp);
									__m128i _lhs = _mm_sub_epi64(_mm_and_si128(_not_temp, _mm_sub_epi64(_zero, _not_temp)), _one);
									__m128i _lhs_shifted_right = _mm_srlv_epi64(_lhs, _rhs);

									_DESTi = _mm_or_si128(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m128i _EVEN = _mm_set1_epi64x(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm_or_si128(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m128i _ODD = _mm_set1_epi64x(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm_or_si128(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m128i _EVEN = _mm_set1_epi64x(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm_and_si128(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m128i _ODD = _mm_set1_epi64x(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm_and_si128(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(0); x != static_cast<int32_t>(64); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm_xor_si128(_DESTi, _mm_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(1); x != static_cast<int32_t>(65); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm_xor_si128(_DESTi, _mm_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m128i _set_msb = _mm_set1_epi64x(static_cast<int64_t>(0b1000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm_or_si128(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m128i _clear_msb = _mm_set1_epi64x(static_cast<int64_t>(0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm_and_si128(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m128i _clear_lsb = _mm_set1_epi64x(static_cast<int64_t>(0b1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm_and_si128(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m128i _clear_lsb = _mm_set1_epi64x(static_cast<int64_t>(0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001));
									_DESTi = _mm_or_si128(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm_xor_si128(_source, _mm_and_si128(_source, _mm_sub_epi64(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm_xor_si128(_source, tpa::simd::_mm_exthsb_epi64(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m128i _hsb = tpa::simd::_mm_exthsb_epi64(_source);
									const __m128i _sl = _mm_slli_epi64(_hsb, 1);
									__m128i _mask = _mm_cmpgt_epi64(_max, _sl);
									_mask = _mm_and_si128(_sl, _mask);

									_DESTi = _mm_or_si128(_source, _mask);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m128i _lowest = _mm_and_si128(_source, _mm_sub_epi64(_zero, _source));
									_lowest = _mm_srli_epi64(_lowest, 1);

									__m128i _mask = _mm_cmpgt_epi64(_lowest, _zero);
									_mask = _mm_and_si128(_lowest, _mask);

									_DESTi = _mm_or_si128(_source, _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t / uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_si128((__m128i*) & source[i], _DESTi);
							}//End for
						}//End if has SSE2
#endif
					}//End if
#pragma endregion
#pragma region float
					else if constexpr (std::is_same<T, float>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							const uint32_t p = static_cast<uint32_t>(pos);

							__m512i _source = _mm512_setzero_si512();
							__m512 _sourcef = _mm512_setzero_si512();
							__m512i _DESTi = _mm512_setzero_si512();
							__m512i _temp = _mm512_setzero_si512();
							const __m512i _zero = _mm512_setzero_si512();
							const __m512i _one = _mm512_set1_epi32(1);
							const __m512i _max = _mm512_set1_epi32(std::numeric_limits<uint32_t>::max());
							const __m512i _shifted_left = _mm512_slli_epi32(_one, p);
							const __m512i _not_shifted_left = tpa::simd::_mm512_not_si512(_shifted_left);

							for (; (i + 16uz) < end; i += 16uz)
							{
								//Set Values
								_sourcef = _mm512_load_ps(&source[i]);
								_source = _mm512_castps_si512(_sourcef);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm512_or_si512(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									//~(1ull << pos) & source[i]
									_DESTi = _mm512_and_si512(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm512_xor_si512(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm512_not_si512(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0uz)
									{
										_DESTi = _mm512_or_si512(_mm512_slli_epi32(_DESTi, 1u), _mm512_and_si512(_source, _one));
										_source = _mm512_srli_epi32(_source, 1u);
										bits -= 1uz;
									}//End while
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm512_or_si512(_mm512_sub_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm512_and_si512(_mm512_add_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm512_lzset_epi32(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm512_not_si512(tpa::simd::_mm512_lzset_epi32(tpa::simd::_mm512_not_si512(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm512_or_si512(_source, _mm512_sub_epi32(_source, _one));

									__m512i _rhs = _mm512_add_epi32(tpa::simd::_mm512_bsf_epi32(_source), _one);

									__m512i _added = _mm512_add_epi32(_temp, _one);
									__m512i _not_temp = tpa::simd::_mm512_not_si512(_temp);
									__m512i _lhs = _mm512_sub_epi32(_mm512_and_si512(_not_temp, _mm512_sub_epi32(_zero, _not_temp)), _one);
									__m512i _lhs_shifted_right = _mm512_srlv_epi32(_lhs, _rhs);

									_DESTi = _mm512_or_si512(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m512i _EVEN = _mm512_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm512_or_si512(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m512i _ODD = _mm512_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm512_or_si512(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m512i _EVEN = _mm512_set1_epi32(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm512_and_si512(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m512i _ODD = _mm512_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm512_and_si512(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(0); x != static_cast<int32_t>(32); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm512_xor_si512(_DESTi, _mm512_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(1); x != static_cast<int32_t>(33); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm512_xor_si512(_DESTi, _mm512_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m512i _set_msb = _mm512_set1_epi32(static_cast<int32_t>(0b1000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm512_or_si512(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m512i _clear_msb = _mm512_set1_epi32(static_cast<int32_t>(0b0111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm512_and_si512(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m512i _clear_lsb = _mm512_set1_epi32(static_cast<int32_t>(0b1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm512_and_si512(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m512i _clear_lsb = _mm512_set1_epi32(static_cast<int32_t>(0b0000'0000'0000'0000'0000'0000'0000'0001));
									_DESTi = _mm512_or_si512(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm512_xor_si512(_source, _mm512_and_si512(_source, _mm512_sub_epi32(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm512_xor_si512(_source, tpa::simd::_mm512_exthsb_epi32(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m512i _hsb = tpa::simd::_mm512_exthsb_epi32(_source);
									const __m512i _sl = _mm512_slli_epi32(_hsb, 1u);
									const __mmask16 _mask = _mm512_cmplt_epi16_mask(_sl, _max);

									_DESTi = _mm512_or_si512(_source, _mm512_mask_slli_epi32(_hsb, _mask, _hsb, 1u));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m512i _lowest = _mm512_and_si512(_source, _mm512_sub_epi32(_zero, _source));
									_lowest = _mm512_srli_epi32(_lowest, 1u);
									const __mmask16 _mask = _mm512_cmpgt_epi32_mask(_lowest, _zero);

									_DESTi = _mm512_or_si512(_source, _mm512_mask_slli_epi32(_lowest, _mask, _lowest, 1u));
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t> / uint32_t).");
									}();
								}//End else

								//Store Result
								_mm512_store_ps(&source[i], _mm512_castsi512_ps(_DESTi));
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							const int32_t p = static_cast<int32_t>(pos);

							__m256i _source = _mm256_setzero_si256();
							__m256 _sourcef = _mm256_setzero_si256();
							__m256i _DESTi = _mm256_setzero_si256();
							__m256i _temp = _mm256_setzero_si256();
							const __m256i _zero = _mm256_setzero_si256();
							const __m256i _one = _mm256_set1_epi32(1);
							const __m256i _max = _mm256_set1_epi32(std::numeric_limits<uint32_t>::max());
							const __m256i _shifted_left = _mm256_slli_epi32(_one, p);
							const __m256i _not_shifted_left = tpa::simd::_mm256_not_si256(_shifted_left);

							for (; (i + 8uz) < end; i += 8uz)
							{
								//Set Values
								_sourcef = _mm256_load_ps(&source[i]);
								_source = _mm256_castps_si256(_sourcef);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm256_or_si256(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									//~(1ull << pos) & source[i]
									_DESTi = _mm256_and_si256(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm256_xor_si256(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm256_not_si256(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0uz)
									{
										_DESTi = _mm256_or_si256(_mm256_slli_epi32(_DESTi, 1), _mm256_and_si256(_source, _one));
										_source = _mm256_srli_epi32(_source, 1);
										bits -= 1uz;
									}//End while
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm256_or_si256(_mm256_sub_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm256_and_si256(_mm256_add_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm256_lzset_epi32(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm256_not_si256(tpa::simd::_mm256_lzset_epi32(tpa::simd::_mm256_not_si256(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm256_or_si256(_source, _mm256_sub_epi32(_source, _one));

									__m256i _rhs = _mm256_add_epi32(tpa::simd::_mm256_bsf_epi32(_source), _one);

									__m256i _added = _mm256_add_epi32(_temp, _one);
									__m256i _not_temp = tpa::simd::_mm256_not_si256(_temp);
									__m256i _lhs = _mm256_sub_epi32(_mm256_and_si256(_not_temp, _mm256_sub_epi32(_zero, _not_temp)), _one);
									__m256i _lhs_shifted_right = _mm256_srlv_epi32(_lhs, _rhs);

									_DESTi = _mm256_or_si256(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m256i _EVEN = _mm256_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm256_or_si256(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m256i _ODD = _mm256_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm256_or_si256(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m256i _EVEN = _mm256_set1_epi32(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm256_and_si256(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m256i _ODD = _mm256_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm256_and_si256(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(0); x != static_cast<int32_t>(32); x += static_cast<int16_t>(2))
									{
										_DESTi = _mm256_xor_si256(_DESTi, _mm256_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(1); x != static_cast<int32_t>(33); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm256_xor_si256(_DESTi, _mm256_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m256i _set_msb = _mm256_set1_epi32(static_cast<int16_t>(0b1000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm256_or_si256(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m256i _clear_msb = _mm256_set1_epi32(static_cast<int16_t>(0b0111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm256_and_si256(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m256i _clear_lsb = _mm256_set1_epi32(static_cast<int16_t>(0b1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm256_and_si256(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m256i _clear_lsb = _mm256_set1_epi32(static_cast<int16_t>(0b0000'0000'0000'0001'0000'0000'0000'0001));
									_DESTi = _mm256_or_si256(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm256_xor_si256(_source, _mm256_and_si256(_source, _mm256_sub_epi32(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm256_xor_si256(_source, tpa::simd::_mm256_exthsb_epi32(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m256i _hsb = tpa::simd::_mm256_exthsb_epi32(_source);
									const __m256i _sl = _mm256_slli_epi32(_hsb, 1);
									__m256i _mask = _mm256_cmpgt_epi32(_max, _sl);
									_mask = _mm256_and_si256(_sl, _mask);

									_DESTi = _mm256_or_si256(_source, _mask);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m256i _lowest = _mm256_and_si256(_source, _mm256_sub_epi32(_zero, _source));
									_lowest = _mm256_srli_epi32(_lowest, 1);

									__m256i _mask = _mm256_cmpgt_epi32(_lowest, _zero);
									_mask = _mm256_and_si256(_lowest, _mask);

									_DESTi = _mm256_or_si256(_source, _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t / uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_ps(&source[i], _mm256_castsi256_ps(_DESTi));
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							const int32_t p = static_cast<int32_t>(pos);

							__m128i _source = _mm_setzero_si128();
							__m128 _sourcef = _mm_setzero_si128();
							__m128i _DESTi = _mm_setzero_si128();
							__m128i _temp = _mm_setzero_si128();
							const __m128i _zero = _mm_setzero_si128();
							const __m128i _one = _mm_set1_epi32(1);
							const __m128i _max = _mm_set1_epi32(std::numeric_limits<uint32_t>::max());
							const __m128i _shifted_left = _mm_slli_epi32(_one, p);
							const __m128i _not_shifted_left = tpa::simd::_mm_not_si128(_shifted_left);

							for (; (i + 4uz) < end; i += 4uz)
							{
								//Set Values
								_sourcef = _mm_load_ps(&source[i]);
								_source = _mm_castps_si128(_sourcef);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm_or_si128(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									_DESTi = _mm_and_si128(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm_xor_si128(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm_not_si128(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0)
									{
										_DESTi = _mm_or_si128(_mm_slli_epi32(_DESTi, 1), _mm_and_si128(_source, _one));
										_source = _mm_srli_epi32(_source, 1);
										bits -= 1;
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm_or_si128(_mm_sub_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm_and_si128(_mm_add_epi32(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm_lzset_epi32(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm_not_si128(tpa::simd::_mm_lzset_epi32(tpa::simd::_mm_not_si128(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm_or_si128(_source, _mm_sub_epi32(_source, _one));

									__m128i _rhs = _mm_add_epi32(tpa::simd::_mm_bsf_epi32(_source), _one);

									__m128i _added = _mm_add_epi32(_temp, _one);
									__m128i _not_temp = tpa::simd::_mm_not_si128(_temp);
									__m128i _lhs = _mm_sub_epi32(_mm_and_si128(_not_temp, _mm_sub_epi32(_zero, _not_temp)), _one);
									__m128i _lhs_shifted_right = _mm_srlv_epi32(_lhs, _rhs);

									_DESTi = _mm_or_si128(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m128i _EVEN = _mm_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm_or_si128(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m128i _ODD = _mm_set1_epi32(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm_or_si128(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m128i _EVEN = _mm_set1_epi32(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm_and_si128(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m128i _ODD = _mm_set1_epi32(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm_and_si128(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(0); x != static_cast<int32_t>(32); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm_xor_si128(_DESTi, _mm_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(1); x != static_cast<int32_t>(33); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm_xor_si128(_DESTi, _mm_slli_epi32(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m128i _set_msb = _mm_set1_epi32(static_cast<int32_t>(0b1000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm_or_si128(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m128i _clear_msb = _mm_set1_epi32(static_cast<int32_t>(0b0111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm_and_si128(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m128i _clear_lsb = _mm_set1_epi32(static_cast<int32_t>(0b1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm_and_si128(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m128i _clear_lsb = _mm_set1_epi32(static_cast<int32_t>(0b0000'0000'0000'0000'0000'0000'0000'0001));
									_DESTi = _mm_or_si128(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm_xor_si128(_source, _mm_and_si128(_source, _mm_sub_epi32(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm_xor_si128(_source, tpa::simd::_mm_exthsb_epi32(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m128i _hsb = tpa::simd::_mm_exthsb_epi32(_source);
									const __m128i _sl = _mm_slli_epi32(_hsb, 1);
									__m128i _mask = _mm_cmpgt_epi32(_max, _sl);
									_mask = _mm_and_si128(_sl, _mask);

									_DESTi = _mm_or_si128(_source, _mask);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m128i _lowest = _mm_and_si128(_source, _mm_sub_epi32(_zero, _source));
									_lowest = _mm_srli_epi32(_lowest, 1);

									__m128i _mask = _mm_cmpgt_epi32(_lowest, _zero);
									_mask = _mm_and_si128(_lowest, _mask);

									_DESTi = _mm_or_si128(_source, _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t / uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_ps(&source[i], _mm_castsi128_ps(_DESTi));
							}//End for
						}//End if has SSE2
#endif
					}//End if
#pragma endregion
#pragma region double
					else if constexpr (std::is_same<T, double>())
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512)
						{
							const uint32_t p = static_cast<uint32_t>(pos);

							__m512i _source = _mm512_setzero_si512();
							__m512 _sourcef = _mm512_setzero_si512();
							__m512i _DESTi = _mm512_setzero_si512();
							__m512i _temp = _mm512_setzero_si512();
							const __m512i _zero = _mm512_setzero_si512();
							const __m512i _one = _mm512_set1_epi64(1ull);
							const __m512i _max = _mm512_set1_epi64(std::numeric_limits<uint64_t>::max());
							const __m512i _shifted_left = _mm512_slli_epi64(_one, p);
							const __m512i _not_shifted_left = tpa::simd::_mm512_not_si512(_shifted_left);

							for (; (i + 8uz) < end; i += 8uz)
							{
								//Set Values
								_sourcef = _mm512_load_pd(&source[i]);
								_source = _mm512_castpd_si512(_sourcef);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm512_or_si512(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									_DESTi = _mm512_and_si512(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm512_xor_si512(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm512_not_si512(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0uz)
									{
										_DESTi = _mm512_or_si512(_mm512_slli_epi64(_DESTi, 1u), _mm512_and_si512(_source, _one));
										_source = _mm512_srli_epi64(_source, 1u);
										bits -= 1uz;
									}//End while
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm512_or_si512(_mm512_sub_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm512_and_si512(_mm512_add_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm512_lzset_epi64(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm512_not_si512(tpa::simd::_mm512_lzset_epi64(tpa::simd::_mm512_not_si512(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm512_or_si512(_source, _mm512_sub_epi64(_source, _one));

									__m512i _rhs = _mm512_add_epi64(tpa::simd::_mm512_bsf_epi64(_source), _one);

									__m512i _added = _mm512_add_epi64(_temp, _one);
									__m512i _not_temp = tpa::simd::_mm512_not_si512(_temp);
									__m512i _lhs = _mm512_sub_epi64(_mm512_and_si512(_not_temp, _mm512_sub_epi64(_zero, _not_temp)), _one);
									__m512i _lhs_shifted_right = _mm512_srlv_epi64(_lhs, _rhs);

									_DESTi = _mm512_or_si512(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m512i _EVEN = _mm512_set1_epi64(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm512_or_si512(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m512i _ODD = _mm512_set1_epi64(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm512_or_si512(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m512i _EVEN = _mm512_set1_epi64(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm512_and_si512(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m512i _ODD = _mm512_set1_epi64(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm512_and_si512(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int64_t x = static_cast<int64_t>(0); x != static_cast<int64_t>(64); x += static_cast<int64_t>(2))
									{
										_DESTi = _mm512_xor_si512(_DESTi, _mm512_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int64_t x = static_cast<int64_t>(1); x != static_cast<int64_t>(65); x += static_cast<int64_t>(2))
									{
										_DESTi = _mm512_xor_si512(_DESTi, _mm512_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m512i _set_msb = _mm512_set1_epi64(static_cast<int64_t>(0b1000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm512_or_si512(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m512i _clear_msb = _mm512_set1_epi64(static_cast<int64_t>(0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm512_and_si512(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m512i _clear_lsb = _mm512_set1_epi64(static_cast<int64_t>(0b1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm512_and_si512(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m512i _clear_lsb = _mm512_set1_epi64(static_cast<int64_t>(0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001));
									_DESTi = _mm512_or_si512(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm512_xor_si512(_source, _mm512_and_si512(_source, _mm512_sub_epi64(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm512_xor_si512(_source, tpa::simd::_mm512_exthsb_epi64(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m512i _hsb = tpa::simd::_mm512_exthsb_epi64(_source);
									const __m512i _sl = _mm512_slli_epi64(_hsb, 1u);
									const __mmask8 _mask = _mm512_cmplt_epi64_mask(_sl, _max);

									_DESTi = _mm512_or_si512(_source, _mm512_mask_slli_epi64(_hsb, _mask, _hsb, 1u));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m512i _lowest = _mm512_and_si512(_source, _mm512_sub_epi64(_zero, _source));
									_lowest = _mm512_srli_epi64(_lowest, 1u);
									const __mmask8 _mask = _mm512_cmpgt_epi64_mask(_lowest, _zero);

									_DESTi = _mm512_or_si512(_source, _mm512_mask_slli_epi64(_lowest, _mask, _lowest, 1u));
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int32_t> / uint32_t).");
									}();
								}//End else

								//Store Result
								_mm512_store_pd(&source[i], _mm512_castsi512_pd(_DESTi));
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							const int32_t p = static_cast<int32_t>(pos);

							__m256i _source = _mm256_setzero_si256();
							__m256 _sourcef = _mm256_setzero_si256();
							__m256i _DESTi = _mm256_setzero_si256();
							__m256i _temp = _mm256_setzero_si256();
							const __m256i _zero = _mm256_setzero_si256();
							const __m256i _one = _mm256_set1_epi64x(1ll);
							const __m256i _max = _mm256_set1_epi64x(std::numeric_limits<uint64_t>::max());
							const __m256i _shifted_left = _mm256_slli_epi64(_one, p);
							const __m256i _not_shifted_left = tpa::simd::_mm256_not_si256(_shifted_left);

							for (; (i + 4uz) < end; i += 4uz)
							{
								//Set Values
								_sourcef = _mm256_load_pd(&source[i]);
								_source = _mm256_castpd_si256(_sourcef);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm256_or_si256(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									_DESTi = _mm256_and_si256(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm256_xor_si256(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm256_not_si256(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0uz)
									{
										_DESTi = _mm256_or_si256(_mm256_slli_epi64(_DESTi, 1), _mm256_and_si256(_source, _one));
										_source = _mm256_srli_epi64(_source, 1);
										bits -= 1uz;
									}//End while
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm256_or_si256(_mm256_sub_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm256_and_si256(_mm256_add_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm256_lzset_epi64(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm256_not_si256(tpa::simd::_mm256_lzset_epi64(tpa::simd::_mm256_not_si256(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm256_or_si256(_source, _mm256_sub_epi64(_source, _one));

									__m256i _rhs = _mm256_add_epi64(tpa::simd::_mm256_bsf_epi64(_source), _one);

									__m256i _added = _mm256_add_epi64(_temp, _one);
									__m256i _not_temp = tpa::simd::_mm256_not_si256(_temp);
									__m256i _lhs = _mm256_sub_epi64(_mm256_and_si256(_not_temp, _mm256_sub_epi64(_zero, _not_temp)), _one);
									__m256i _lhs_shifted_right = _mm256_srlv_epi64(_lhs, _rhs);

									_DESTi = _mm256_or_si256(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m256i _EVEN = _mm256_set1_epi64x(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm256_or_si256(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m256i _ODD = _mm256_set1_epi64x(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm256_or_si256(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m256i _EVEN = _mm256_set1_epi64x(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm256_and_si256(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m256i _ODD = _mm256_set1_epi64x(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm256_and_si256(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int64_t x = static_cast<int32_t>(0); x != static_cast<int32_t>(64); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm256_xor_si256(_DESTi, _mm256_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(1); x != static_cast<int32_t>(65); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm256_xor_si256(_DESTi, _mm256_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m256i _set_msb = _mm256_set1_epi64x(static_cast<int16_t>(0b1000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm256_or_si256(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m256i _clear_msb = _mm256_set1_epi64x(static_cast<int16_t>(0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm256_and_si256(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m256i _clear_lsb = _mm256_set1_epi64x(static_cast<int16_t>(0b1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm256_and_si256(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m256i _clear_lsb = _mm256_set1_epi64x(static_cast<int16_t>(0b0000'0000'0000'0001'0000'0000'0000'0000'0000'0000'0000'0001'0000'0000'0000'0001));
									_DESTi = _mm256_or_si256(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm256_xor_si256(_source, _mm256_and_si256(_source, _mm256_sub_epi64(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm256_xor_si256(_source, tpa::simd::_mm256_exthsb_epi64(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m256i _hsb = tpa::simd::_mm256_exthsb_epi64(_source);
									const __m256i _sl = _mm256_slli_epi64(_hsb, 1);
									__m256i _mask = _mm256_cmpgt_epi64(_max, _sl);
									_mask = _mm256_and_si256(_sl, _mask);

									_DESTi = _mm256_or_si256(_source, _mask);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m256i _lowest = _mm256_and_si256(_source, _mm256_sub_epi64(_zero, _source));
									_lowest = _mm256_srli_epi64(_lowest, 1);

									__m256i _mask = _mm256_cmpgt_epi64(_lowest, _zero);
									_mask = _mm256_and_si256(_lowest, _mask);

									_DESTi = _mm256_or_si256(_source, _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t / uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm256_store_pd(&source[i], _mm256_castsi256_pd(_DESTi));
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE42)
						{
							const int32_t p = static_cast<int32_t>(pos);

							__m128i _source = _mm_setzero_si128();
							__m128 _sourcef = _mm_setzero_si128();
							__m128i _DESTi = _mm_setzero_si128();
							__m128i _temp = _mm_setzero_si128();
							const __m128i _zero = _mm_setzero_si128();
							const __m128i _one = _mm_set1_epi64x(1ll);
							const __m128i _max = _mm_set1_epi64x(std::numeric_limits<uint64_t>::max());
							const __m128i _shifted_left = _mm_slli_epi64(_one, p);
							const __m128i _not_shifted_left = tpa::simd::_mm_not_si128(_shifted_left);

							for (; (i + 2uz) < end; i += 2uz)
							{
								//Set Values
								_sourcef = _mm_load_pd(&source[i]);
								_source = _mm_castpd_si128(_sourcef);

								if constexpr (INSTR == tpa::bit_mod::SET)
								{
									_DESTi = _mm_or_si128(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
								{
									_DESTi = _max;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR)
								{
									_DESTi = _mm_and_si128(_not_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
								{
									_DESTi = _zero;
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
								{
									_DESTi = _mm_xor_si128(_shifted_left, _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
								{
									_DESTi = tpa::simd::_mm_not_si128(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::REVERSE)
								{
									size_t bits = sizeof(T) * CHAR_BIT;
									while (bits > 0)
									{
										_DESTi = _mm_or_si128(_mm_slli_epi64(_DESTi, 1), _mm_and_si128(_source, _one));
										_source = _mm_srli_epi64(_source, 1);
										bits -= 1;
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
								{
									_DESTi = _mm_or_si128(_mm_sub_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
								{
									_DESTi = _mm_and_si128(_mm_add_epi64(_source, _one), _source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
								{
									_DESTi = tpa::simd::_mm_lzset_epi64(_source);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
								{
									_DESTi = tpa::simd::_mm_not_si128(tpa::simd::_mm_lzset_epi64(tpa::simd::_mm_not_si128(_source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
								{
									_temp = _mm_or_si128(_source, _mm_sub_epi64(_source, _one));

									__m128i _rhs = _mm_add_epi32(tpa::simd::_mm_bsf_epi32(_source), _one);

									__m128i _added = _mm_add_epi64(_temp, _one);
									__m128i _not_temp = tpa::simd::_mm_not_si128(_temp);
									__m128i _lhs = _mm_sub_epi64(_mm_and_si128(_not_temp, _mm_sub_epi64(_zero, _not_temp)), _one);
									__m128i _lhs_shifted_right = _mm_srlv_epi64(_lhs, _rhs);

									_DESTi = _mm_or_si128(_added, _lhs_shifted_right);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
								{
									const __m128i _EVEN = _mm_set1_epi64x(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm_or_si128(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
								{
									const __m128i _ODD = _mm_set1_epi64x(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm_or_si128(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
								{
									const __m128i _EVEN = _mm_set1_epi64x(static_cast<T>(0b1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010'1010));

									_DESTi = _mm_and_si128(_source, _EVEN);
								}//End else
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
								{
									const __m128i _ODD = _mm_set1_epi64x(static_cast<T>(0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101));

									_DESTi = _mm_and_si128(_source, _ODD);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(0); x != static_cast<int32_t>(64); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm_xor_si128(_DESTi, _mm_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
								{
									_DESTi = _source;

									for (int32_t x = static_cast<int32_t>(1); x != static_cast<int32_t>(65); x += static_cast<int32_t>(2))
									{
										_DESTi = _mm_xor_si128(_DESTi, _mm_slli_epi64(_one, x));
									}//End for
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
								{
									const __m128i _set_msb = _mm_set1_epi64x(static_cast<int64_t>(0b1000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000));
									_DESTi = _mm_or_si128(_source, _set_msb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
								{
									const __m128i _clear_msb = _mm_set1_epi64x(static_cast<int64_t>(0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111));
									_DESTi = _mm_and_si128(_source, _clear_msb);
								}//End if                                                                   
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
								{
									const __m128i _clear_lsb = _mm_set1_epi64x(static_cast<int64_t>(0b1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1110));
									_DESTi = _mm_and_si128(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
								{
									const __m128i _clear_lsb = _mm_set1_epi64x(static_cast<int64_t>(0b0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0000'0001));
									_DESTi = _mm_or_si128(_source, _clear_lsb);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
								{
									_DESTi = _mm_xor_si128(_source, _mm_and_si128(_source, _mm_sub_epi64(_zero, _source)));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
								{
									_DESTi = _mm_xor_si128(_source, tpa::simd::_mm_exthsb_epi64(_source));
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
								{
									const __m128i _hsb = tpa::simd::_mm_exthsb_epi64(_source);
									const __m128i _sl = _mm_slli_epi64(_hsb, 1);
									__m128i _mask = _mm_cmpgt_epi64(_max, _sl);
									_mask = _mm_and_si128(_sl, _mask);

									_DESTi = _mm_or_si128(_source, _mask);
								}//End if
								else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
								{
									__m128i _lowest = _mm_and_si128(_source, _mm_sub_epi64(_zero, _source));
									_lowest = _mm_srli_epi64(_lowest, 1);

									__m128i _mask = _mm_cmpgt_epi64(_lowest, _zero);
									_mask = _mm_and_si128(_lowest, _mask);

									_DESTi = _mm_or_si128(_source, _mask);
								}//End if
								else
								{
									[] <bool flag = false>()
									{
										static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<int16_t / uint16_t>).");
									}();
								}//End else

								//Store Result
								_mm_store_pd(&source[i], _mm_castsi128_pd(_DESTi));
							}//End for
						}//End if has SSE2
#endif
					}//End if
#pragma endregion
#pragma region generic
					for (; i < end; ++i)
					{
						if constexpr (INSTR == tpa::bit_mod::SET)
						{
							tpa::bit_manip::set(source[i], pos);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::SET_ALL)
						{
							tpa::bit_manip::set_all(source[i]);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::CLEAR)
						{
							tpa::bit_manip::clear(source[i], pos);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::CLEAR_ALL)
						{
							tpa::bit_manip::clear_all(source[i]);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::TOGGLE)
						{
							tpa::bit_manip::toggle(source[i], pos);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ALL)
						{
							tpa::bit_manip::toggle_all(source[i]);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::REVERSE)
						{
							tpa::bit_manip::reverse(source[i]);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::SET_TRAILING_ZEROS)
						{
							tpa::bit_manip::set_trailing_zeros(source[i]);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::CLEAR_TRAILING_ONES)
						{
							tpa::bit_manip::clear_trailing_ones(source[i]);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::SET_LEADING_ZEROS)
						{							
							tpa::bit_manip::set_leading_zeros(source[i]);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::CLEAR_LEADING_ONES)
						{
							tpa::bit_manip::clear_leading_ones(source[i]);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::NEXT_LEXICOGRAPHIC_PERMUTATION)
						{
							source[i] = tpa::bit_manip::next_lexicographic_permutation(source[i]);
						}//End Next Lexographic permutation
						else if constexpr (INSTR == tpa::bit_mod::SET_EVEN)
						{
							tpa::bit_manip::set_even(source[i]);
						}//End Set Even
						else if constexpr (INSTR == tpa::bit_mod::SET_ODD)
						{
							tpa::bit_manip::set_odd(source[i]);
						}//End of Set Odd
						else if constexpr (INSTR == tpa::bit_mod::CLEAR_EVEN)
						{
							tpa::bit_manip::clear_even(source[i]);
						}//Clear Even
						else if constexpr (INSTR == tpa::bit_mod::CLEAR_ODD)
						{
							tpa::bit_manip::clear_odd(source[i]);
						}//Clear Odd
						else if constexpr (INSTR == tpa::bit_mod::TOGGLE_EVEN)
						{
							tpa::bit_manip::toggle_even(source[i]);
						}//End of Toggle Even
						else if constexpr (INSTR == tpa::bit_mod::TOGGLE_ODD)
						{
							tpa::bit_manip::toggle_odd(source[i]);
						}//End of Toggle Odd
						else if constexpr (INSTR == tpa::bit_mod::CLEAR_MSB)
						{
							tpa::bit_manip::clear_msb(source[i]);
						}//End of Clear MSB
						else if constexpr (INSTR == tpa::bit_mod::SET_MSB)
						{
							tpa::bit_manip::set_msb(source[i]);
						}//End of Set MSB
						else if constexpr (INSTR == tpa::bit_mod::CLEAR_LSB)
						{
							tpa::bit_manip::clear_lsb(source[i]);
						}//End of Clear LSB
						else if constexpr (INSTR == tpa::bit_mod::SET_LSB)
						{
							tpa::bit_manip::set_lsb(source[i]);
						}//End of Set LSB
						else if constexpr (INSTR == tpa::bit_mod::CLEAR_LOWEST_SET)
						{
							tpa::bit_manip::clear_lowest_set(source[i]);
						}//End of clear lowest set bit
						else if constexpr (INSTR == tpa::bit_mod::CLEAR_HIGHEST_SET)
						{
							tpa::bit_manip::clear_highest_set(source[i]);
						}//End of clear highest set bit
						else if constexpr (INSTR == tpa::bit_mod::SET_HIGHEST_CLEAR)
						{
							tpa::bit_manip::set_highest_clear(source[i]);
						}//End if
						else if constexpr (INSTR == tpa::bit_mod::SET_LOWEST_CLEAR)
						{
							tpa::bit_manip::set_lowest_clear(source[i]);
						}//End if
						else
						{
							[] <bool flag = false>()
							{
								static_assert(flag, " You have specifed an invalid SIMD instruction in tpa::simd::bit_manip::bit_modify<__UNDEFINED_PREDICATE__>(CONTAINER<T>).");
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
			
		}//End of try
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::bit_modify: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::bit_manip::bit_modify: unknown!\n";
		}//End catch	
	}//End of bit_modify

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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
									dest[i] = tpa::simd::fp_bitwise<tpa::bit::AND>(source1[i], source2[i]);
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
									dest[i] = tpa::simd::fp_bitwise<tpa::bit::OR>(source1[i], source2[i]);
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
									dest[i] = tpa::simd::fp_bitwise<tpa::bit::XOR>(source1[i], source2[i]);
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
									dest[i] = tpa::simd::fp_bitwise<tpa::bit::AND_NOT>(source1[i], source2[i]);
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
									dest[i] = tpa::simd::fp_bitwise<tpa::bit::AND>(source1[i], val);
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
									dest[i] = tpa::simd::fp_bitwise<tpa::bit::OR>(source1[i], val);
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
									dest[i] = tpa::simd::fp_bitwise<tpa::bit::XOR>(source1[i], val);
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
									dest[i] = tpa::simd::fp_bitwise<tpa::bit::AND_NOT>(source1[i], val);
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
	/// <para>Shifts, Rotates or Merges the element's bits in a container the specified direction by the number specified in the second container, storing the results in a 3rd container.</para>
	/// <para>Containers may be of differnt types, but not recommended.</para>
	/// <para>Containers MUST have the same value_type.</para>
	/// <para>Containers do not have to be a particular size, of passing containers of different sizes, will only execute upto the size of the smallest container, the destination container must be at least this size.</para>
	/// <para>Non-standard integers will work.</para>
	/// <para>This implementation uses SIMD (if avaialble for type) and Multi-Threading.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_B"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <param name="source1"></param>
	/// <param name="source2"></param>
	/// <param name="dest"></param>
	template<tpa::bit_move INSTR, typename CONTAINER_A, typename CONTAINER_B, typename CONTAINER_C>
	inline constexpr void bit_move(
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

#pragma region short
					if constexpr ((std::is_same<T, int16_t>()) || (std::is_same<T, uint16_t>()))
					{
#ifdef TPA_X86_64
						if (tpa::hasAVX512_ByteWord)
						{
							__m512i _Ai, _Bi, _DESTi;

							for (; (i + 32uz) < end; i += 32uz)
							{
								//Set Values
								_Ai = _mm512_loadu_epi16((__m512i*)&source1[i]);
								_Bi = _mm512_loadu_epi16((__m512i*)&source2[i]);

								if constexpr (INSTR == tpa::bit_move::SHIFT_LEFT)
								{
									_DESTi = _mm512_sllv_epi16(_Ai, _Bi);
								}//End  if
								else if constexpr (INSTR == tpa::bit_move::SHIFT_LEFT_ARITHMETIC)
								{
									_DESTi = _mm512_sllv_epi16(_Ai, _Bi);
								}//End  if
								else if constexpr (INSTR == tpa::bit_move::SHIFT_RIGHT)
								{
									_DESTi = _mm512_srlv_epi16(_Ai, _Bi);
								}//End  if
								else if constexpr (INSTR == tpa::bit_move::SHIFT_RIGHT_ARITHMETIC)
								{
									_DESTi = _mm512_srav_epi16(_Ai, _Bi);
								}//End  if

								//Store Result
								_mm512_storeu_epi16((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX512
						else if (tpa::hasAVX2)
						{
							__m256i _Ai, _Bi, _DESTi;

							for (; (i + 16uz) < end; i += 16uz)
							{
								//Set Values
								_Ai = _mm256_load_si256((__m256i*)&source1[i]);
								_Bi = _mm256_load_si256((__m256i*)&source2[i]);

								if constexpr (INSTR == tpa::bit_move::SHIFT_LEFT)
								{
									_DESTi = tpa::simd::_mm256_sllv_epi16(_Ai, _Bi);
								}//End  if
								else if constexpr (INSTR == tpa::bit_move::SHIFT_LEFT_ARITHMETIC)
								{
									_DESTi = tpa::simd::_mm256_sllv_epi16(_Ai, _Bi);
								}//End  if
								else if constexpr (INSTR == tpa::bit_move::SHIFT_RIGHT)
								{
									_DESTi = tpa::simd::_mm256_srlv_epi16(_Ai, _Bi);
								}//End  if
								else if constexpr (INSTR == tpa::bit_move::SHIFT_RIGHT_ARITHMETIC)
								{
									//_DESTi = tpa::simd::_mm256_srav_epi16(_Ai, _Bi);
									//TO-DO Implement the above
								}//End  if

								//Store Result
								_mm256_store_si256((__m256i*) & dest[i], _DESTi);
							}//End for
						}//End if hasAVX2
						else if (tpa::has_SSE2)
						{
							__m128i _Ai, _Bi, _DESTi;

							for (; (i + 8uz) < end; i += 8uz)
							{
								//Set Values
								_Ai = _mm_load_si128((__m128i*) & source1[i]);
								_Bi = _mm_load_si128((__m128i*) & source2[i]);

								if constexpr (INSTR == tpa::bit_move::SHIFT_LEFT)
								{
									_DESTi = tpa::simd::_mm_sllv_epi16(_Ai, _Bi);
								}//End  if
								else if constexpr (INSTR == tpa::bit_move::SHIFT_LEFT_ARITHMETIC)
								{
									_DESTi = tpa::simd::_mm_sllv_epi16(_Ai, _Bi);
								}//End  if
								else if constexpr (INSTR == tpa::bit_move::SHIFT_RIGHT)
								{
									_DESTi = tpa::simd::_mm_srlv_epi16(_Ai, _Bi);
								}//End  if
								else if constexpr (INSTR == tpa::bit_move::SHIFT_RIGHT_ARITHMETIC)
								{
									//_DESTi = tpa::simd::_mm_srav_epi16(_Ai, _Bi);
									//TO-DO Implement the above
								}//End  if

								//Store Result
								_mm_store_si128((__m128i*) & dest[i], _DESTi);
							}//End for
						}//End if hasSSE2
#endif							
					}//End if
#pragma endregion
#pragma region generic
					for (; i != end; ++i)
					{
						if constexpr (INSTR == tpa::bit_move::SHIFT_LEFT)
						{
							dest[i] = source1[i] << source2[i];
						}//End if
						else if constexpr (INSTR == tpa::bit_move::SHIFT_LEFT_ARITHMETIC)
						{
							dest[i] = source1[i] << source2[i];
						}//End if
						else if constexpr (INSTR == tpa::bit_move::SHIFT_RIGHT)
						{
							dest[i] = source1[i] >> source2[i];
						}//End if
						else if constexpr (INSTR == tpa::bit_move::SHIFT_RIGHT_ARITHMETIC)
						{
							dest[i] = source1[i] >> source2[i];
						}//End if
						else if constexpr (INSTR == tpa::bit_move::ROTATE_LEFT)
						{
							dest[i] = tpa::bit_manip::rotate_left(source1[i], source2[i]);
						}//End if
						else if constexpr (INSTR == tpa::bit_move::ROTATE_RIGHT)
						{
							dest[i] = tpa::bit_manip::rotate_right(source1[i], source2[i]);
						}//End if
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
			std::cerr << "Exception thrown in tpa::simd::bit_move: " << ex.what() << "\n";
			std::cerr << "tpa::simd::bit_move will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_move(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_move: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_move: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_move: unknown!\n";
		}//End catch
	}//End of bit_move()

	/// <summary>
	/// <para>Shifts, Rotates or Merges the element's bits in a container the specified direction by the number specified in 'amount', storing the results in a 3rd container.</para>
	/// <para>Containers may be of differnt types, but not recommended.</para>
	/// <para>Containers MUST have the same value_type.</para>
	/// <para>Containers do not have to be a particular size, of passing containers of different sizes, will only execute upto the size of the smallest container, the destination container must be at least this size.</para>
	/// <para>Non-standard integers will work.</para>
	/// <para>This implementation uses SIMD (if avaialble for type) and Multi-Threading.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_A"></typeparam>
	/// <typeparam name="CONTAINER_C"></typeparam>
	/// <param name="source1"></param>
	/// <param name="amount"></param>
	/// <param name="dest"></param>
	template<tpa::bit_move INSTR, typename CONTAINER_A, typename CONTAINER_C>
	inline constexpr void bit_move(
		const CONTAINER_A& source1,
		const uint32_t amount,
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
			std::cerr << "Exception thrown in tpa::simd::bit_move: " << ex.what() << "\n";
			std::cerr << "tpa::simd::bit_move will execute upto the current size of the container.";
			smallest = dest.size();
			goto recover;
		}//End catch
		catch (const std::future_error& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_move(): " << ex.code()
				<< " " << ex.what() << "\n";
		}//End catch
		catch (const std::bad_alloc& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_move: " << ex.what() << "\n";
		}//End catch
		catch (const std::exception& ex)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_move: " << ex.what() << "\n";
		}//End catch
		catch (...)
		{
			std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
			std::cerr << "Exception thrown in tpa::simd::bit_move: unknown!\n";
		}//End catch
	}//End of bit_move()

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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
#ifdef TPA_X86_64
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
								dest[i] = tpa::simd::fp_bitwise_not(source[i]);
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

	}//End of namespace bit_manip
}//End of namespace tpa
