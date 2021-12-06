#pragma once

#include <cstdint>
#include <cstdlib>

#ifndef __cpp_size_t_suffix
/// <summary>
/// <para>Literal Suffix for size_t</para>
/// <para>Manually implemented in tpa/size_t_lit.hpp before C++23</para>
/// </summary>
/// <param name="n"></param>
/// <returns></returns>
consteval std::size_t operator ""uz(std::size_t n)
{
	return n;
}

/// <summary>
/// <para>Literal Suffix for size_t</para>
/// <para>Manually implemented in tpa/size_t_lit.hpp before C++23</para>
/// </summary>
/// <param name="n"></param>
/// <returns></returns>
consteval std::size_t operator ""UZ(std::size_t n)
{
	return n;
}

#endif
