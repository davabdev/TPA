#pragma once
/*
* Truely Parallel Algorithms Library - Timer Class
* By: David Aaron Braun
* 2022-07-08
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include <iostream>
#include <chrono>

#include "ThreadPool.hpp"
#include "excepts.hpp"
#include "size_t_lit.hpp"
#include "tpa_macros.hpp"
#include "predicates.hpp"
#include "tpa_concepts.hpp"

namespace tpa::util{

	/// <summary>
	/// <para>Provides a scope-based timer (stop watch) class for benchmarking purposes</para>
	/// <para>Outputs time taken to console in nanoseconds.</para>
	/// <para>Requires iostream and chrono.</para>
	/// </summary>
	class Timer
	{
	public:
		Timer()
		{
			start = std::chrono::high_resolution_clock::now();
		}//End of constructor

		Timer(Timer const&) = delete;
		Timer& operator=(Timer const&) = delete;
		Timer(Timer&&) = delete;
		Timer& operator=(Timer&&) = delete;

		~Timer()
		{
			end = std::chrono::high_resolution_clock::now();
			std::chrono::high_resolution_clock::duration d = end - start;
			std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(d).count() << "ns\n";
		}//End of destructor

	private:
		std::chrono::high_resolution_clock::time_point start;
		std::chrono::high_resolution_clock::time_point end;
	};//End of class Timer

};//End of namespace