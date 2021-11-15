/*
* Application for testing of TPA Code
* David Aaron Braun
* 2021-01-24
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/
#include <cstdlib>
#include <crtdbg.h>
#include <iostream>
#include <locale>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <execution>
#include <functional>
#include <limits>
#include <random>
#include <iomanip>

#include <array>
#include <vector>

#include <fenv.h>


//#include <boost/multiprecision/cpp_int.hpp>
//#include <boost/container/static_vector.hpp>

#include "../TPA/tpa_main.hpp"

using numtype = int32_t;//boost::multiprecision::uint128_t;
using returnType = float;//boost::multiprecision::uint128_t; 

std::vector<numtype> vec;
std::vector<returnType> vec2;
std::vector<returnType> vec3;
//std::vector<numtype> vec4;


#pragma region single-threaded-tests

template<typename T>
inline bool mimx(T min, T max)
{
	if (min < max)
	{
		return true;
	}//End if
	else
	{
		return false;
	}//End else
}

uint64_t addTwo(const uint64_t& lhs, const uint64_t& rhs)
{
	uint64_t ans = lhs + rhs;
	std::cout << ans << "\n";
	return ans;
}//End of addTwo

template <typename T>
inline constexpr T fill_less_than()
{
	static T countDown = 50'000;

	return countDown -= 1;
}//End of fill_less_than

template <typename T>
inline constexpr T gen_odd()
{
	static T odds = 0;
	return ((odds +=2) -1);
}//End of gen_odd

template <typename T>
inline T gen_even()
{
	static T counter = 0;
	return (counter += 2);
}//End of gen_even

template<typename T>
inline constexpr T fast_random()
{
	return static_cast<T>(std::rand());
}//End of fast_random

std::random_device rd;
std::mt19937 gen(rd());
std::mt19937_64 gen_64(rd());

std::uniform_int_distribution<uint64_t> distrib(1, 6);
std::uniform_real_distribution<float> f_distrib(1.0f, 6.0f);
std::uniform_real_distribution<double> d_distrib(1.0, 6.0);
template<typename T>
inline constexpr T true_random()
{
	if constexpr (std::is_same<float,T>())
	{
		return f_distrib(gen);
	}//End if
	else if constexpr (std::is_same<double, T>())
	{
		return d_distrib(gen_64);
	}//End if
	else
	{
		return distrib(gen);
	}
}//End of true_random

#pragma endregion

int main()
{
#ifdef _DEBUG
	// Enable CRT memory leak checking
	int dbgFlags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	dbgFlags |= _CRTDBG_CHECK_ALWAYS_DF;
	dbgFlags |= _CRTDBG_DELAY_FREE_MEM_DF;
	dbgFlags |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag(dbgFlags);
#endif

	try
	{
		std::cout.imbue(std::locale(""));
		std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 7);
		std::ios::sync_with_stdio(false);

		vec.resize(1'000'000'133);
	    vec2.resize(1'000'000'133);
	    //vec3.resize(500'000'133);

		tpa::runtime_instruction_set.output_CPU_info();	

		//Generate		
		/*
		std::cout << "STD Generate Single-Threaded: ";
		{
			tpa::util::Timer t;
			std::generate(vec.begin(), vec.end(), true_random<numtype>);
		}
		
		std::cout << "STD Generate Multi-Threaded: ";
		{
			tpa::util::Timer t;
			std::generate(std::execution::par_unseq, vec.begin(), vec.end(), true_random<numtype>);
		}		
		*/
		std::cout << "TPA Generate Multi-Threaded SIMD: ";
		{
			tpa::util::Timer t;
			tpa::generate<tpa::gen::UNIFORM>(vec, 1.0f, 6.0f);
			//tpa::generate<tpa::gen::UNIFORM>(vec2, 1, 500);
		}
		
		std::cout << "STD sine Single Threaded: ";
		{			
			tpa::util::Timer t;
			for (size_t i = 0; i < vec.size(); ++i)
			{
				vec2[i] = std::sin(vec[i]);
			}
		}	

		std::cout << "vec1 \t+\t vec2 \t=\t vec3\n";
		for (size_t i = 0; i != vec.size(); ++i)
		{
			if (i > 101)
			{
				break;
			}

			std::cout << std::left <<
				std::setw(5) << i <<
				std::setw(35) << static_cast<double>(vec[i]) <<
				std::setw(35) << static_cast<double>(vec2[i]) <<
				/*std::setw(35) << static_cast<double>(vec3[i]) <<*/ "\n";
		}//End for

		//std::fill(vec3.begin(), vec3.end(), (numtype)0);

		std::cout << "STD sine transform: ";
		{
			tpa::util::Timer t;
			std::transform(std::execution::par_unseq, vec.cbegin(), vec.cend(), vec2.begin(), [](numtype a) -> returnType { 
				return static_cast<returnType>(std::sin(a));
			});
		}
		
		std::cout << "vec1 \t+\t vec2 \t=\t vec3\n";
		for (size_t i = 0; i != vec.size(); ++i)
		{
			if (i > 101)
			{
				break;
			}

			std::cout << std::left << 
			std::setw(5) << i <<
			std::setw(35) << static_cast<double>(vec[i]) <<
			std::setw(35) << static_cast<double>(vec2[i]) <<
			/*std::setw(35) << static_cast<double>(vec3[i]) <<*/ "\n";
		}//End for

		std::fill(vec3.begin(), vec3.end(), (numtype)0);

		std::cout << "TPA sine Multi-Threaded SIMD: ";
		{
			tpa::util::Timer t;
			tpa::simd::trigonometry<tpa::trig::SINE, tpa::angle::RADIANS>(vec, vec2);
		}

		std::cout << "vec1 \t+\t vec2 \t=\t vec3\n";
		for (size_t i = 0; i != vec.size(); ++i)
		{
			if (i > 101)
			{
				break;
			}

			std::cout << std::left <<
				std::setw(5) << i <<
				std::setw(35) << static_cast<double>(vec[i]) <<
				std::setw(35) << static_cast<double>(vec2[i]) <<
				/*std::setw(35) << static_cast<double>(vec3[i]) <<*/ "\n";
		}//End for
		
		std::cout << "End of Benchmark.\n";

		return EXIT_SUCCESS;
	}//End try
	catch (const std::bad_alloc& ex)
	{
		std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
		std::cerr << "Exception thrown in Testing::main: " << ex.what() << "\n";
		return EXIT_FAILURE;
	}//End catch
	catch (const std::exception& ex)
	{
		std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
		std::cerr << "Exception thrown in Testing::main: " << ex.what() << "\n";
		return EXIT_FAILURE;
	}//End catch
	catch (...)
	{
		std::scoped_lock<std::mutex> lock(tpa::util::consoleMtx);
		std::cerr << "Exception thrown in Testing::main: unknown!\n";
		return EXIT_FAILURE;
	}//End catch
}//End of main method