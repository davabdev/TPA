#pragma once
/*
*	Truly Parallel Algorithms Library - Algorithm - minmax_element function
*	By: David Aaron Braun
*	2021-07-31
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
	/// <para>Returns an std::pair of the smallest and largest elements of a container.</para>
	/// <para>This parallel implentation uses Multi-Threading and SIMD.</para>
	/// <para>The return type is the value_type of the container.</para>
    /// <para>If passing an container containing no elements, will throw an exception and return 0.</para>
	/// </summary>
	/// <typeparam name="CONTAINER_T"></typeparam>
	/// <typeparam name="T"></typeparam>
	/// <param name="arr"></param>
	/// <returns></returns>
	template<class CONTAINER_T, typename T = CONTAINER_T::value_type>
	inline constexpr std::pair<T,T> minmax_element(const CONTAINER_T& arr) 
	requires tpa::util::contiguous_seqeunce<CONTAINER_T>
	{
		return {tpa::min_element(arr), tpa::max_element(arr)};
	}//End of minmax_element

	/// <summary>
	/// <para>Returns an std::pair of the smallest and largest elements of a container.</para>
	/// <para>Requires 2 predicate functions 1 for min and 1 for max.</para>
	/// <para>This parallel implementation uses Multi-Threading only.</para>
	/// <para>The return type is the value_type of the container.</para>
	/// <para>If passing an container containing no elements, will throw an exception and return 0.</para>
	/// <para>IMPORTANT: This implementation is intended to be used with non-numeric custom classes, if your container's value_type is numeric, use the implementation without a predicate for a performance increase!</para>
	/// </summary>
	/// <typeparam name="CONTAINER_T"></typeparam>
	/// <typeparam name="T"></typeparam>
	/// <param name="arr"></param>
	/// <param name="min_p"></param>
	/// <param name="max_p"></param>
	/// <returns></returns>
	template<class CONTAINER_T, typename T = CONTAINER_T::value_type, class PRED, class PRED2>
	inline constexpr std::pair<T,T> minmax_element(const CONTAINER_T& arr, const PRED min_p, const PRED2 max_p)
	requires tpa::util::contiguous_seqeunce<CONTAINER_T>
	{
		return { tpa::min_element(arr, min_p), tpa::max_element(arr, max_p) };
	}//End of minmax_element
#pragma endregion
}//End of namespace
