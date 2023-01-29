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
//#include <boost/multiprecision/cpp_bin_float.hpp>

#include "../TPA/tpa_main.hpp"

using numtype = int32_t;//boost::multiprecision::cpp_bin_float_oct;
using returnType = int32_t;//boost::multiprecision::cpp_bin_float_oct; 

std::vector<numtype> vec;
std::vector<numtype> vec2;
std::vector<returnType> vec3;
//std::vector<numtype> vec4;


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

		vec.resize(1'000'000'000);
	    vec2.resize(1'000'000'000);
	    //vec3.resize(1'000'000'000);

		tpa::runtime_instruction_set.output_CPU_info();	

		//Generate	
		/*
		std::cout << "STD iota Single-Threaded: ";
		{
			tpa::util::Timer t;
			std::iota(vec.begin(), vec.end(), numtype(0));
		}
		
		std::cout << "STD Generate Multi-Threaded: ";
		{
			tpa::util::Timer t;
			std::generate(std::execution::par_unseq, vec.begin(), vec.end(), true_random<numtype>);
		}		
		*/		

		std::cout << "TPA iota Multi-Threaded SIMD: ";
		{
			tpa::util::Timer t;
			tpa::generate<tpa::gen::UNIFORM>(vec, std::numeric_limits<numtype>::min(), std::numeric_limits<numtype>::max());
			tpa::fill(vec2, static_cast<numtype>(0));
		}

		std::cout << "index\t vec1\t| vec2 \t| vec3\n";
		for (size_t i = 0; i != vec.size(); ++i)
		{
			if (i > 101)
			{
				break;
			}

			std::cout << std::left <<
				std::setw(5) << i <<
				std::setw(33) << std::setw(33) << tpa::util::as_bits(vec[i]) <<
				std::setw(33) << std::setw(33) << tpa::util::as_bits(vec2[i]) <<
				//std::setw(10) << static_cast<double>(vec3[i]) <<
				"\n";
		}//End for

		tpa::copy(vec, vec2);

		std::cout << "STD Set Lowest Clear: ";
		{
			tpa::util::Timer t;
			
			for (size_t i = 0uz; i < vec.size(); ++i)
			{
				tpa::bit_manip::set_lowest_clear(vec2[i]);
			}//End for
		}

		std::cout << "index\t vec1\t| vec2 \t| vec3\n";
		for (size_t i = 0; i != vec.size(); ++i)
		{
			if (i > 101)
			{
				break;
			}

			std::cout << std::left <<
				std::setw(5) << i <<
				std::setw(33) << std::setw(33) << tpa::util::as_bits(vec[i]) <<
				std::setw(33) << std::setw(33) << tpa::util::as_bits(vec2[i]) <<
				//std::setw(10) << static_cast<double>(vec3[i]) <<
				"\n";
		}//End for

		tpa::copy(vec, vec2);
		
		std::cout << "STD Set Lowest Clear Multi-Threaded: ";
		{
			tpa::util::Timer t;

			std::for_each(std::execution::par_unseq, vec2.begin(), vec2.end(),
				tpa::bit_manip::set_lowest_clear<numtype>);
		}
		

		std::cout << "index\t vec1\t| vec2 \t| vec3\n";
		for (size_t i = 0; i != vec.size(); ++i)
		{
			if (i > 101)
			{
				break;
			}

			std::cout << std::left <<
				std::setw(5) << i <<
				std::setw(33) << std::setw(33) << tpa::util::as_bits(vec[i]) <<
				std::setw(33) << std::setw(33) << tpa::util::as_bits(vec2[i]) <<
				//std::setw(10) << static_cast<double>(vec3[i]) <<
				"\n";
		}//End for

		tpa::copy(vec, vec2);

		std::cout << "TPA Set Lowest Clear SIMD: ";
		{
			tpa::util::Timer t;
			tpa::bit_manip::bit_modify<tpa::bit_mod::SET_LOWEST_CLEAR>(vec2);
		}

		std::cout << "index\t vec1\t| vec2 \t| vec3\n";
		for (size_t i = 0; i != vec.size(); ++i)
		{
			if (i > 101)
			{
				break;
			}

			std::cout << std::left <<
				std::setw(5) << i <<
				std::setw(33) << std::setw(33) << tpa::util::as_bits(vec[i]) <<
				std::setw(33) << std::setw(33) << tpa::util::as_bits(vec2[i]) <<
				//std::setw(10) << static_cast<double>(vec3[i]) <<
				"\n";
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


