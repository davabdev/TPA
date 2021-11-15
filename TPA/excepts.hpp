#pragma once

/*
* Custom Exceptions for Lib TPA
* David Aaron Braun
* 2021-04-08
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include <exception>
#include <string>

#include <fenv.h>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#include <float.h>
#if defined(_M_FP_PRECISE) || defined(_M_FP_STRICT) 
#pragma fenv_access (on)
#endif
#else
#pragma STDC FENV_ACCESS ON
#endif

/// <summary>
/// Namespace to store TPA error codes
/// </summary>
namespace tpa::error_codes {
	const char* NotInit = "Thread Pool has not been initialized! Call 'tpa::init()' before using any library functions!";
	const char* SizeOfZero = "The size of the passed container was 0!";

	const char* NotArrayLike = "The passed container does not have an implentation for size() or operator[] !";
	const char* InvalidSimdInstruction = "AN INVALID SIMD INSTRUCTION WAS PASSED!";

	const char* SIMD_Unavailable = "SIMD is required for this function but is unavailable on this hardware configuration.";

	const char* MismatchedData = "There is not an implementation for this data type!";

	const char* NotAllThreadsCompleted = "TPA Non-Fatal Error: Not all threads completed execution.";

	const char* ArrayTooSmall = "The specified destination container is too small to hold the results.";

	const char* RequiresFloatingPointType = "This function requires a IEEE-754 Floating Point Type.";

	const char* FP_DivideByZero = "Floating-Point divide by zero!";

	const char* FP_Inexact = "Floating-Point Inexact!";

	const char* FP_Invaid = "Floating-Point Invalid!";

	const char* FP_Underflow = "Floating-Point Underflow!";

	const char* FP_Overflow = "Floating-Point Overflow";
}//End of namespace

/// <summary>
/// Namespace to store TPA Exceptions
/// </summary>
namespace tpa::exceptions {

#pragma region floating-point exceptions

	/// <summary>
	/// <para>Temporarily disables floating-point exceptions</para>
	/// <para>Scope based, previous values are restored after leaving scope.</para>
	/// </summary>
	class FPExceptionDisabler
	{
	private:
		unsigned int mOldValues;
	public:
		FPExceptionDisabler()
		{
			_controlfp_s(&mOldValues, 0, 0);
			_controlfp_s(0, _MCW_EM, _MCW_EM);
		}//End constructor

		FPExceptionDisabler(FPExceptionDisabler const&) = delete;
		FPExceptionDisabler& operator=(FPExceptionDisabler const&) = delete;
		FPExceptionDisabler(FPExceptionDisabler&&) = delete;
		FPExceptionDisabler& operator=(FPExceptionDisabler&&) = delete;

		~FPExceptionDisabler()
		{
			_clearfp();
			_controlfp_s(0, mOldValues, _MCW_EM);
		}//End destructor
	};//End class

	class FP_DivideByZero : public std::exception
	{
		std::string message = tpa::error_codes::FP_DivideByZero;
		std::string fct_name;
	public:
		FP_DivideByZero(const std::string& param) : fct_name(param)
		{
			message += " In function: ";
			message += fct_name;
			message += ".\n";
		}

		virtual const char* what() const throw()
		{
			return message.c_str();
		}
	};

	class FP_Inexact : public std::exception
	{
		std::string message = tpa::error_codes::FP_Inexact;
		std::string fct_name;
	public:
		FP_Inexact(const std::string& param) : fct_name(param)
		{
			message += " In function: ";
			message += fct_name;
			message += ".\n";
		}

		virtual const char* what() const throw()
		{
			return message.c_str();
		}
	};

	class FP_Invaid : public std::exception
	{
		std::string message = tpa::error_codes::FP_Invaid;
		std::string fct_name;
	public:
		FP_Invaid(const std::string& param) : fct_name(param)
		{
			message += " In function: ";
			message += fct_name;
			message += ".\n";
		}

		virtual const char* what() const throw()
		{
			return message.c_str();
		}
	};

	class FP_Underflow : public std::exception
	{
		std::string message = tpa::error_codes::FP_Underflow;
		std::string fct_name;
	public:
		FP_Underflow(const std::string& param) : fct_name(param)
		{
			message += " In function: ";
			message += fct_name;
			message += ".\n";
		}

		virtual const char* what() const throw()
		{
			return message.c_str();
		}
	};

	class FP_Overflow : public std::exception
	{
		std::string message = tpa::error_codes::FP_Overflow;
		std::string fct_name;
	public:
		FP_Overflow(const std::string& param) : fct_name(param)
		{
			message += " In function: ";
			message += fct_name;
			message += ".\n";
		}

		virtual const char* what() const throw()
		{
			return message.c_str();
		}
	};

	/// <summary>
	/// <para>Tests for floating-point exceptions and outputs to std::cout if any are raised.</para>
	/// <para>Takes the name of the function as a parameter</para>
	/// <para>Warning: All FP exceptions are fairly expensive to throw, only run this function if you need to!</para>
	/// </summary>
	/// <param name="fct_name"></param>
	/// <returns></returns>
	inline void catch_fp_exceptions(const std::string& fct_name)
	{
		int e = fetestexcept(FE_ALL_EXCEPT);

		if (e & FE_DIVBYZERO) {
			throw tpa::exceptions::FP_DivideByZero(fct_name);
		}
		if (e & FE_INEXACT) {
			throw tpa::exceptions::FP_Inexact(fct_name);
		}
		if (e & FE_INVALID) {
			throw tpa::exceptions::FP_Invaid(fct_name);
		}
		if (e & FE_UNDERFLOW) {
			throw tpa::exceptions::FP_Underflow(fct_name);
		}
		if (e & FE_OVERFLOW) {
			throw tpa::exceptions::FP_Overflow(fct_name);
		}		
	}//End of catch_fp_exceptions
#pragma endregion

	class ThreadPoolNotInitialized : public std::exception
	{
	public:
		virtual const char* what() const throw()
		{
			return tpa::error_codes::NotInit;
		}
	};

	class EmptyArray : public std::exception
	{
	public:
		virtual const char* what() const throw()
		{
			return tpa::error_codes::SizeOfZero;
		}
	};

	class ArrayTooSmall : public std::exception
	{
	public:
		virtual const char* what() const throw()
		{
			return tpa::error_codes::ArrayTooSmall;
		}
	};

	class NotArray : public std::exception
	{
	public:
		virtual const char* what() const throw()
		{
			return tpa::error_codes::NotArrayLike;
		}
	};

	class InvalidSIMDInstruction : public std::exception
	{
	public:
		virtual const char* what() const throw()
		{
			return tpa::error_codes::InvalidSimdInstruction;
		}
	};

	class SIMDUnavailable : public std::exception
	{
	public:
		virtual const char* what() const throw()
		{
			return tpa::error_codes::SIMD_Unavailable;
		}
	};

	class RequiresFloatingPoint : public std::exception
	{
	public:
		virtual const char* what() const throw()
		{
			return tpa::error_codes::RequiresFloatingPointType;
		}
	};

	class MismatchedData : public std::exception
	{
	public:
		virtual const char* what() const throw()
		{
			return tpa::error_codes::MismatchedData;
		}
	};

	class NotAllThreadsCompleted : public std::exception
	{
	private:
		uint32_t completed;
		std::string message = tpa::error_codes::NotAllThreadsCompleted;

	public:

		NotAllThreadsCompleted(const uint32_t& param): completed(param)
		{
			message += " Completed: ";
			message += std::to_string(completed);
			message += '\n';
		}

		virtual const char* what() const throw()
		{
			return message.c_str();
		}
	};
}//End of namespace
