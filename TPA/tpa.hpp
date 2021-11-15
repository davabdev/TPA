#pragma once
/*
* Truly Parallel Algorithms Library - Thread Pool Initialization and general library management 
* By: David Aaron Braun
* 2021-01-24
* Describes a Thread Pool Object
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include "ThreadPool.hpp"

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa {

	/// <summary>
	/// Pointer to a Thread Pool Singleton
	/// </summary>
	static tpa_thread_pool_private::ThreadPool* tp = &tp->instance();
}//End of namespace tpa
