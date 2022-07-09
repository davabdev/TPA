#pragma once
/*
* Truely Parallel Algorithms Library - Concepts
* By: David Aaron Braun
* 2022-07-08
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
#include <vector>
#include <forward_list>
#include <thread>
#include <utility>
#include <iostream>
#include <chrono>
#include <mutex>
#include <numbers>
#include <bitset>
#include <bit>
#include <array>

#include "ThreadPool.hpp"
#include "excepts.hpp"
#include "size_t_lit.hpp"
#include "tpa_macros.hpp"
#include "predicates.hpp"

namespace tpa::util {

	template<typename CONT, typename ITER = CONT::iterator, typename VAL = CONT::value_type>
	/// <summary>
	/// <para> concept contiguous_seqeunce requires: </para>
	/// <para> size() function returning an integer convertible to std::size_t </para>
	/// <para> An implementation of the subscript operator[] returning a const T &amp; </para>
	/// <para> Container's iterator must satisfy all the requirements of std::contiguous_iterator </para>
	/// </summary>
	concept contiguous_seqeunce = requires(const CONT & cont, const std::size_t index) {
		{cont.size()} -> std::convertible_to<std::size_t>;
		{cont[index]} -> std::convertible_to<const VAL&>;
		std::contiguous_iterator<ITER>;
	};

	template<typename T>
	/// <summary>
	/// <para>Concept calculatable requires: </para>
	/// <para>operator overload  for = </para>
	/// <para>operator overloads for ==  and != </para>
	/// <para>operator overloads for +, -, *, and / </para>
	/// <para>operator overloads for (POST) ++, -- </para>
	/// <para>operator overloads for (PRE) ++, -- //</para>
	/// <para>operator overloads for +=, -=, *=, and /= </para>
	/// </summary>
	concept calculatable = requires(T t)
	{
		t == t;
		t != t;

		t = t;

		t + t;
		t - t;
		t* t;
		t / t;

		++t;
		t++;

		--t;
		t--;

		t += t;
		t -= t;
		t *= t;
		t /= t;
	};

}//End of namespace
