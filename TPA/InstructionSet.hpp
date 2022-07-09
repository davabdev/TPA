#pragma once
/*
* Truly Parallel Algorithms Library - Instruction Set Availability Information
* By: David Aaron Braun based on 
* https://docs.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?redirectedfrom=MSDN&view=msvc-160
* 2021-10-28
*/

/*
*           Copyright David Aaron Braun 2021 - .
*   Distributed under the Boost Software License, Version 1.0.
*       (See accompanying file LICENSE_1_0.txt or copy at
*           https://www.boost.org/LICENSE_1_0.txt)
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <bitset>
#include <array>
#include <string>
#include <thread>

#ifdef  _MSC_VER 
        #include <intrin.h>
#define CPUID(registers, function) __cpuid((int*)registers, (int)function);
#define CPUIDEX(registers, function, extFunction) __cpuidex((int*)registers, (int)function, (int)extFunction);
#else 
#define CPUID(registers, function) asm volatile ("cpuid" : "=a" (registers[0]), "=b" (registers[1]), "=c" (registers[2]), "=d" (registers[3]) : "a" (function), "c" (0));
#define CPUIDEX(registers, function, extFunction) asm volatile ("cpuid" : "=a" (registers[0]), "=b" (registers[1]), "=c" (registers[2]), "=d" (registers[3]) : "a" (function), "c" (extFunction));
#endif

#include "size_t_lit.hpp"
#include "tpa_macros.hpp"

/// <summary>
/// <para>CPUID Functions</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa_cpuid_private {

    /// <summary>
    /// <para>Provides a means of identifying the CPU and available features at runtime.</para>
    /// <para>Note: All functions to check for the presence of instruction set extentions are only defined for specific architectures and you must wrap any calls to these functions in preprocessor checks for the appropriate architecture (such as _M_AMD64, _M_ARM64).</para>
    /// <para>Only 64-Bit platforms are officially supported</para>
    /// </summary>
    class InstructionSet
    {
    public:
        std::string Vendor(void) const { return vendor_; }
        std::string Brand(void) const { return brand_; }

#if defined(TPA_X86_64)        
        
#pragma region SIMD
        /// <summary>
        /// <para>CPU has Multi-Media eXtentions instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=MMX</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool MMX(void) const noexcept { return f_1_EDX_[23]; }

        /// <summary>
        /// <para>CPU has Extended Multi-Media eXtentions instructions if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/Extended_MMX</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool MMXEXT(void) const noexcept { return isAMD_ && f_81_EDX_[22]; }

        /// <summary>
        /// <para>CPU has 3DNow! instructions if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/3DNow!</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool _3DNOW(void) const noexcept { return isAMD_ && f_81_EDX_[31]; }

        /// <summary>
        /// <para>CPU has Extended 3DNow! / 3DNow! Enchanced / 3DNow!+ instructions if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/3DNow!</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool _3DNOWEXT(void) const noexcept { return isAMD_ && f_81_EDX_[30]; }

        /// <summary>
        /// <para>CPU has Streaming SIMD (Singe Instruction Multiple Data) Extentions instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SSE(void) const noexcept { return f_1_EDX_[25]; }

        /// <summary>
        /// <para>CPU has Streaming SIMD (Singe Instruction Multiple Data) Extentions 2.0 instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE2</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SSE2(void) const noexcept { return f_1_EDX_[26]; }

        /// <summary>
        /// <para>CPU has Streaming SIMD (Singe Instruction Multiple Data) Extentions 3.0 instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE3</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SSE3(void) const noexcept { return f_1_ECX_[0]; }

        /// <summary>
        /// <para>CPU has Supplemental Streaming SIMD (Singe Instruction Multiple Data) Extentions 3.0 instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSSE3</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SSSE3(void) const noexcept { return f_1_ECX_[9]; }

        /// <summary>
        /// <para>CPU has Streaming SIMD (Singe Instruction Multiple Data) Extentions 4.0a instructions if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/SSE4#SSE4a</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SSE4a(void) const noexcept { return isAMD_ && f_81_ECX_[6]; }

        /// <summary>
        /// <para>CPU has Streaming SIMD (Singe Instruction Multiple Data) Extentions 4.1 instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE4_1</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SSE41(void) const noexcept { return f_1_ECX_[19]; }

        /// <summary>
        /// <para>CPU has Streaming SIMD (Singe Instruction Multiple Data) Extentions 4.2 instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE4_2</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SSE42(void) const noexcept { return f_1_ECX_[20]; }

        /// <summary>
        /// <para>CPU has eXtended Operations instructions if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/XOP_instruction_set</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool XOP(void) const noexcept { return isAMD_ && f_81_ECX_[11]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX(void) const noexcept { return f_1_ECX_[28]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 2.0 instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX2</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX2(void) const noexcept { return f_7_EBX_[5]; }

        /// <summary>
        /// <para>CPU has Fused Multiply Add 3.0 instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=FMA</para>
        /// <para>Note: Older AMD CPUs may also support the FMA 4.0 instruction set. This function does not detect FMA 4.0 as FMA 4.0 has been removed on AMD since Zen1 and was never implemented on Intel. </para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool FMA(void) const noexcept { return f_1_ECX_[12]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions Vector Neural Network Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=AVX_VNNI</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX_VNNI(void) const noexcept { return f_7_ECX_[4]; /*eax bit 4*/ }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Foundation instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512F</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX512F(void) const noexcept { return f_7_EBX_[16]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Prefetch instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512PF</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX512PF(void) const noexcept { return f_7_EBX_[26]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Exponential and Reciprocal Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512ER</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX512ER(void) const noexcept { return f_7_EBX_[27]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Conflict Detection Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512CD</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX512CD(void) const noexcept { return f_7_EBX_[28]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Byte and Word Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512BW</para>
        /// <para>512-bit vector operations overloaded for vectors of bytes (unsigned char / uint8_t / char / int8_t) and shorts (uint16_t / int16_t)</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX512BW(void) const noexcept { return f_7_EBX_[30]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Vector Length Extensions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512VL</para>
        /// <para>AVX512-VL provides 128-bit and 256-bit overloads for most AVX-512 instructions</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX512VL(void) const noexcept { return f_7_EBX_[31]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Doubleword and Quadword Instructions Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512DQ</para>
        /// <para>Essential instructions for bitwise operations on floats (from SSE and AVX) as well as 64-bit integer multiply</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX512DQ(void) const noexcept { return f_7_EBX_[17]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Integer Fused Multiply Add Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool IFMA(void) const noexcept { return f_7_EBX_[21]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Vector Byte Manipulation 1.0 Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512_VBMI</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool VBMI(void) const noexcept { return f_7_ECX_[1]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Vector Byte Manipulation 2.0 Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512_VBMI2</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool VBMI2(void) const noexcept { return f_7_ECX_[6]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Vector Neural Network Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512_VNNI</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool VNNI(void) const noexcept { return f_7_ECX_[11]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Vector Neural Network Instructions Word variable precision Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512_4VNNIW</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool FOUR_VNNIW(void) const noexcept { return f_81_EDX_[2]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Vector Fused Multiply Accumulation Packed Single precision Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512_4FMAPS</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool FOUR_MAPS(void) const noexcept { return f_81_EDX_[3]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Vector Population Count Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512VPOPCNTDQ</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool VPOPCNTDQ(void) const noexcept { return f_7_EBX_[32]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Bit Algorithms Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512_BITALG</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool BITALG(void) const noexcept { return f_7_ECX_[12]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Vector Pair Intersection Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512_VP2INTERSECT</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool VP2INTERSECT(void) const noexcept { return f_81_EDX_[8]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Galois Field Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=GFNI</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool GFNI(void) const noexcept { return f_7_ECX_[8]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtentions 512-Bit Carry-Less Multiply Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=VPCLMULQDQ </para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool VPCLMULQDQ(void) const noexcept { return f_7_ECX_[10]; }
        
        /// <summary>
        /// <para>CPU has Knights Landing Architecture if returns true</para>
        /// <para>(AVX-512F, CD, ER, PF)</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=KNC</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool KNCNI(void) const noexcept
        {
            return f_7_EBX_[16] && f_7_EBX_[28] &&
                f_7_EBX_[27] && f_7_EBX_[26];
        }

#pragma endregion 

        /// <summary>
        /// <para>CPU has Advanced Vector eXtension 512-Bit (Brain Float 16) Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512_BF16</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AVX512_FP16(void) const noexcept { return f_81_EDX_[23]; }

        /// <summary>
        /// <para>CPU has Advanced Matrix eXtension (Brain Float 16) Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#amxtechs=AMXBF16</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AMXBF16(void) const noexcept { return f_81_EDX_[22]; /*eax bit 5*/ }
           
        /// <summary>
        /// <para>CPU has Advanced Matrix eXtension Tile Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#amxtechs=AMXTILE</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AMXTILE(void) const noexcept { return f_81_EDX_[24]; }

        /// <summary>
        /// <para>CPU has Advanced Matrix eXtension (byte/ char) Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#amxtechs=AMXINT8</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AMXINT8(void) const noexcept { return f_81_EDX_[25]; }

#pragma region security
        /// <summary>
        /// <para>CPU has Advanced Encryption Standard Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=AES</para>
        /// <para>See also: https://en.wikipedia.org/wiki/Advanced_Encryption_Standard</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool AES(void) const noexcept { return f_1_ECX_[25]; }

        /// <summary>
        /// <para>CPU has Advanced Vector eXtensions (512-Bit) Advanced Encryption Standard Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=VAES</para>
        /// <para>See also: https://en.wikipedia.org/wiki/Advanced_Encryption_Standard</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool VAES(void) const noexcept { return f_7_ECX_[9]; }

        /// <summary>
        /// <para>CPU has Secure Hash Algorithim Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=SHA</para>
        /// <para>See also: https://en.wikipedia.org/wiki/Secure_Hash_Algorithms</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SHA(void) const noexcept { return f_7_EBX_[29]; }

        /// <summary>
        /// <para>CPU has Software Guard eXtention Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=3769,2723&text=sgx</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SGX(void) const noexcept { return f_7_EBX_[2]; }

        /// <summary>
        /// <para>CPU has Keylocker Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=KEYLOCKER</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool KEYLOCKER(void) const noexcept { return f_1_ECX_[23]; }

        /// <summary>
        /// <para>CPU has Keylocker Wide Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=KEYLOCKER_WIDE</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool KEYLOCKER_WIDE(void) const noexcept { return f_7_EBX_[0]; }

        // <summary>
        /// <para>CPU has Supervisor Mode Access Prevention if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/Supervisor_Mode_Access_Prevention</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SMAP(void) const noexcept { return f_7_EBX_[20]; }

        /// <summary>
        /// <para>CPU has Supervisor Mode Execution Prevention if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/Supervisor_Mode_Access_Prevention</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SMEP(void) const noexcept { return f_7_EBX_[7]; }

        /// <summary>
        /// <para>CPU has User Mode Instruction Prevention if returns true</para>
        /// <para>See: https://cateee.net/lkddb/web-lkddb/X86_INTEL_UMIP.html</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool UMIP(void) const noexcept { return f_1_ECX_[2]; }

        /// <summary>
        /// <para>CPU has Protected Keys for Supervisor Mode if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/Supervisor_Mode_Access_Prevention</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool PKS(void) const noexcept { return f_1_ECX_[31]; }

        /// <summary>
        /// <para>CPU has Protected Keys for User Mode if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/Supervisor_Mode_Access_Prevention</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool PKU(void) const noexcept { return f_7_ECX_[3]; }

        /// <summary>
        /// <para>CPU has Operating System Protected Keys for User Mode if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/Supervisor_Mode_Access_Prevention</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool OSPKE(void) const noexcept { return f_7_ECX_[4]; }

        /// <summary>
        /// <para>CPU has Trusted Domain eXtensions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-trust-domain-extensions.html</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool TDX(void) const noexcept { return false; }

        /// <summary>
        /// <para>CPU has Virtual Machine eXtensions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/develop/documentation/debug-extensions-windbg-hyper-v-user-guide/top/vmx-instructions-for-intel-64-and-ia-32-architectures.html</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool VMX(void) const noexcept { return f_1_ECX_[5]; }

        /// <summary>
        /// <para>CPU has Safe Mode eXtensions if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/Safe_mode</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool SMX(void) const noexcept { return f_1_ECX_[6]; }

#pragma endregion

#pragma region bit_mapiulation 
        /// <summary>
        /// <para>CPU has Bit Manipulation Instructions 1.0 if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=BMI1</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool BMI1(void) const noexcept { return f_7_EBX_[3]; }

        /// <summary>
        /// <para>CPU has Bit Manipulation Instructions 2.0 if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=BMI2</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool BMI2(void) const noexcept { return f_7_EBX_[8]; }

        /// <summary>
        /// <para>CPU has Advanced Bit Manipulation Instructions if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/X86_Bit_manipulation_instruction_set#ABM_(Advanced_Bit_Manipulation)</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool ABM(void) const noexcept { return isAMD_ && f_81_ECX_[5]; }

        /// <summary>
        /// <para>CPU has Trailing Bit Manipulation Instructions if returns true</para>
        /// <para>See: https://en.wikipedia.org/wiki/X86_Bit_manipulation_instruction_set#TBM_(Trailing_Bit_Manipulation)</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool TBM(void) const noexcept { return isAMD_ && f_81_ECX_[21]; }

        /// <summary>
        /// <para>CPU has Population Count Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=POPCNT</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool POPCNT(void) const noexcept { return f_1_ECX_[23]; }

        /// <summary>
        /// <para>CPU has Leading-Zero Count Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=LZCNT</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool LZCNT(void) const noexcept { return isIntel_ && f_81_ECX_[5]; }

#pragma endregion

        //Advanced Move Instructions
        bool CMOV(void) const noexcept { return f_1_EDX_[15]; }
        bool REP_MOV(void) const noexcept { return f_81_EDX_[4]; }
        bool MOVBE(void) const noexcept { return f_1_ECX_[22]; }
        bool MOVDIRI(void) const noexcept { return f_1_ECX_[27]; }
        bool MOVDIR64B(void) const noexcept { return f_1_ECX_[28]; }
        bool ENQCMD(void) const noexcept { return f_1_ECX_[29]; }
        bool CMPXCHG16B(void) const noexcept { return f_1_ECX_[13]; }
        bool CMPXCHG8B(void) const noexcept { return f_81_EDX_[8]; }
        bool MOVSB(void) const noexcept { return f_7_EBX_[9]; }
        bool STOSB(void) const noexcept { return f_7_EBX_[9]; }
        bool CMPSB(void) const noexcept { return false; /*eax bit 12*/ }

        //Multi-Precision Add-Carry Instruction Extensions
        bool ADX(void) const noexcept { return f_7_EBX_[19]; }

        //Flush Cache Line Optimized
        bool CLFLUSHOPT(void) const noexcept { return f_7_EBX_[23]; }

        //Cache Line Write Back
        bool CLWB(void) const noexcept { return f_7_EBX_[24]; }

        //Debug 
        bool DE(void) const noexcept { return f_1_EDX_[2]; }

        //Debug Store
        bool DS(void) const noexcept { return f_1_EDX_[21]; }

        /// <summary>
        /// <para>CPU has Random Seed Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#cats=Random&ig_expand=5625
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool RDSEED(void) const noexcept { return f_7_EBX_[18]; }

        /// <summary>
        /// <para>CPU has Random Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#cats=Random&ig_expand=5625
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool RDRAND(void) const noexcept { return f_1_ECX_[30]; }

        //Other
        bool FPU(void) const noexcept { return f_1_EDX_[0]; }
        bool F16C(void) const noexcept { return f_1_ECX_[29]; }
        bool FDP_EXCPTN_ONLY(void) const noexcept { return f_7_EBX_[6]; }
        bool FXSR(void) const noexcept { return f_1_EDX_[24]; }
        bool PCLMULQDQ(void) const noexcept { return f_1_ECX_[1]; }
        bool MONITOR(void) const noexcept { return f_1_ECX_[3]; }
        bool WAITPKG(void) const noexcept { return f_1_ECX_[5]; }
        bool CET_SS(void) const noexcept { return f_1_ECX_[7]; }
        bool TME_EN(void) const noexcept { return f_1_ECX_[13]; }
        bool RDPID(void) const noexcept { return f_7_ECX_[22]; }
        bool XSAVE(void) const noexcept { return f_1_ECX_[26]; }
        bool OSXSAVE(void) const noexcept { return f_1_ECX_[27]; }        
        bool LAM(void) const noexcept { return false; /*eax bit 26*/ }

        bool MSR(void) const noexcept { return f_1_EDX_[5]; }
        bool MD_CLEAR(void) const noexcept { return f_81_EDX_[10]; }
        bool SEP(void) const noexcept { return f_81_EDX_[11]; }
        bool SERIALIZE(void) const noexcept { return f_81_EDX_[14]; }
        bool HYBRID_PROCESSOR(void) const noexcept { return f_81_EDX_[15]; }
        bool PCONFIG(void) const noexcept { return f_81_EDX_[18]; }
        bool CLFSH(void) const noexcept { return f_1_EDX_[19]; }

        bool FSGSBASE(void) const noexcept { return f_7_EBX_[0]; }
        bool HLE(void) const noexcept { return isIntel_ && f_7_EBX_[4]; }
        bool ERMS(void) const noexcept { return f_7_EBX_[9]; }
        bool INVPCID(void) const noexcept { return f_7_EBX_[10]; }
        
        bool RTM(void) const noexcept { return isIntel_ && f_7_EBX_[11]; }
        bool RDT_M(void) const noexcept { return f_7_EBX_[12]; }
        bool RDT_A(void) const noexcept { return f_7_EBX_[15]; }
                
        bool LAHF(void) const noexcept { return f_81_ECX_[0]; }
        bool SYSCALL(void) const noexcept { return isIntel_ && f_81_EDX_[11]; }
        bool RDTSCP(void) const noexcept { return isIntel_ && f_81_EDX_[27]; }

        /// <summary>
        /// <para>CPU has the Prefetch Instructions if returns true</para>
        /// <para>See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#othertechs=PREFETCHWT1</para>
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool PREFETCHWT1(void) const noexcept { return f_7_ECX_[0]; }

        /// <summary>
        /// <para>Prints the 'common' or 'interesting' CPU features to the console.</para>
        /// <para>This is not an exhaustive list, the InstructionSet class has many more functions available and is capable of determining the presence of all CPU features known as of 2021-10-28.</para>
        /// </summary>
        const void output_CPU_info() const noexcept
        {
            std::cout << "CPU Info\n";
            std::cout << "-----------------------------\n";

            std::cout << std::left << std::setw(21) << "CPU Vendor: " <<
                std::setw(20) << std::setfill(' ') << Vendor() << "\n";

            std::cout << std::left << std::setw(21) << "CPU Brand: " <<
                std::setw(49) << std::setfill(' ') << Brand() << "\n";

            std::cout << std::left << std::setw(21) << "Logical Threads: " <<
                std::setw(49) << std::setfill(' ') << std::thread::hardware_concurrency() << "\n";

            std::cout << std::left << std::setw(21) << "Hybrid Architecture: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << HYBRID_PROCESSOR() << "\n";

            std::cout << std::left << std::setw(21) << "MMX: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << MMX() << "\n";

            std::cout << std::left << std::setw(21) << "MMXEXT: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << MMXEXT() << "\n";

            std::cout << std::left << std::setw(21) << "3D Now!: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << _3DNOW() << "\n";

            std::cout << std::left << std::setw(21) << "3D Now! Ext: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << _3DNOWEXT() << "\n";

            std::cout << std::left << std::setw(21) << "SSE: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << SSE() << "\n";

            std::cout << std::left << std::setw(21) << "SSE2: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << SSE2() << "\n";

            std::cout << std::left << std::setw(21) << "SSE3: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << SSE3() << "\n";

            std::cout << std::left << std::setw(21) << "SSSE3: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << SSSE3() << "\n";

            std::cout << std::left << std::setw(21) << "SSE4a: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << SSE4a() << "\n";

            std::cout << std::left << std::setw(21) << "SSE4.1: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << SSE41() << "\n";

            std::cout << std::left << std::setw(21) << "SSE4.2: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << SSE42() << "\n";

            std::cout << std::left << std::setw(21) << "XOP: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << XOP() << "\n";

            std::cout << std::left << std::setw(21) << "BMI1: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << BMI1() << "\n";

            std::cout << std::left << std::setw(21) << "BMI2: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << BMI2() << "\n";

            std::cout << std::left << std::setw(21) << "POPCNT: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << POPCNT() << "\n";

            std::cout << std::left << std::setw(21) << "LZCNT: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << LZCNT() << "\n";

            std::cout << std::left << std::setw(21) << "ABM: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << ABM() << "\n";

            std::cout << std::left << std::setw(21) << "AVX: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << AVX() << "\n";

            std::cout << std::left << std::setw(21) << "AVX2: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << AVX2() << "\n";

            std::cout << std::left << std::setw(21) << "FMA: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << FMA() << "\n";

            std::cout << std::left << std::setw(21) << "AVX-VNNI: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << AVX_VNNI() << "\n";

            std::cout << std::left << std::setw(21) << "AVX512F: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << AVX512F() << "\n";

            std::cout << std::left << std::setw(21) << "AVX512PF: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << AVX512PF() << "\n";

            std::cout << std::left << std::setw(21) << "AVX512CD: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << AVX512CD() << "\n";

            std::cout << std::left << std::setw(21) << "AVX512ER: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << AVX512ER() << "\n";

            std::cout << std::left << std::setw(21) << "AVX512BW: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << AVX512BW() << "\n";

            std::cout << std::left << std::setw(21) << "AVX512DQ: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << AVX512DQ() << "\n";

            std::cout << std::left << std::setw(21) << "AVX512VL: " <<
                std::setw(25) << std::setfill(' ') << std::boolalpha << AVX512VL() << "\n";

            std::cout << "-----------------------------\n\n";
        }//End of output_CPU_info

#elif defined(TPA_ARM)
        bool NEON(void) const noexcept { return true; }
        bool SVE(void) const noexcept { return true; }
        bool SVE2(void) const noexcept { return true; }
        bool HELIUM(void) const noexcept { return true; }

        void output_CPU_info() const noexcept
        {
            std::cout << "CPU Info\n";
            std::cout << "-----------------------------\n";

            std::cout << std::left << std::setw(21) << "CPU Vendor: " <<
                std::setw(20) << std::setfill(' ') << Vendor() << "\n";

            std::cout << std::left << std::setw(21) << "CPU Brand: " <<
                std::setw(49) << std::setfill(' ') << Brand() << "\n";

            std::cout << std::left << std::setw(21) << "Logical Threads: " <<
                std::setw(49) << std::setfill(' ') << std::thread::hardware_concurrency() << "\n";

            std::cout << std::left << std::setw(21) << "NEON: " <<
                std::setw(49) << std::setfill(' ') << NEON() << "\n";

            std::cout << std::left << std::setw(21) << "SVE: " <<
                std::setw(49) << std::setfill(' ') << SVE() << "\n";

            std::cout << std::left << std::setw(21) << "SVE2: " <<
                std::setw(49) << std::setfill(' ') << SVE2() << "\n";

            std::cout << std::left << std::setw(21) << "Helium: " <<
                std::setw(49) << std::setfill(' ') << HELIUM() << "\n";

            std::cout << "-----------------------------\n\n";
        }//End of output_CPU_info
#else
        void output_CPU_info() const noexcept
        {
            std::cout << "CPU Info\n";
            std::cout << "-----------------------------\n";

            std::cout << std::left << std::setw(21) << "CPU Vendor: " <<
            std::setw(20) << std::setfill(' ') << Vendor() << "\n";

            std::cout << std::left << std::setw(21) << "CPU Brand: " <<
            std::setw(49) << std::setfill(' ') << Brand() << "\n";

            std::cout << std::left << std::setw(21) << "Logical Threads: " <<
            std::setw(49) << std::setfill(' ') << std::thread::hardware_concurrency() << "\n";
        }//End of output_CPU_info()
#endif

    private:
        std::string vendor_ = {};
        std::string brand_ = {};

#if defined(TPA_X86_64)
        int32_t nIds_ = 0;
        int32_t nExIds_ = 0;
        bool isIntel_ = false;
        bool isAMD_ = false;
        std::bitset<32> f_1_ECX_;
        std::bitset<32> f_1_EDX_;
        std::bitset<32> f_7_EBX_;
        std::bitset<32> f_7_ECX_;
        std::bitset<32> f_81_ECX_;
        std::bitset<32> f_81_EDX_;
        std::vector<std::array<int32_t, 4>> data_;
        std::vector<std::array<int32_t, 4>> extdata_;
        std::array<int32_t, 4> cpui = {};

    public:
        /// <summary>
        /// Construct Instruction Set Availability Object
        /// </summary>
        InstructionSet() noexcept
                : nIds_{ 0 },
                nExIds_{ 0 },
                isIntel_{ false },
                isAMD_{ false },
                f_1_ECX_{ 0 },
                f_1_EDX_{ 0 },
                f_7_EBX_{ 0 },
                f_7_ECX_{ 0 },
                f_81_ECX_{ 0 },
                f_81_EDX_{ 0 },
                data_{},
                extdata_{}
        {  

            CPUID(cpui.data(), 0);
            nIds_ = cpui[0];

            for (size_t i = 0uz; i <= nIds_; ++i)
            {
                CPUIDEX(cpui.data(), static_cast<int32_t>(i), 0);
                data_.emplace_back(cpui);
            }//End for

            // Capture vendor string
            char vendor[0x20];
            memset(vendor, 0, sizeof(vendor));
            *reinterpret_cast<int32_t*>(vendor) = data_[0][1];
            *reinterpret_cast<int32_t*>(vendor + 4) = data_[0][3];
            *reinterpret_cast<int32_t*>(vendor + 8) = data_[0][2];
            vendor_ = vendor;
            if (vendor_ == "GenuineIntel")
            {
                isIntel_ = true;
            }//End if
            else if (vendor_ == "AuthenticAMD")
            {
                isAMD_ = true;
            }//End if

            // load bitset with flags for function 0x00000001
            if (nIds_ >= 1)
            {
                f_1_ECX_ = data_[1][2];
                f_1_EDX_ = data_[1][3];
            }//End if

            // load bitset with flags for function 0x00000007
            if (nIds_ >= 7)
            {
                f_7_EBX_ = data_[7][1];
                f_7_ECX_ = data_[7][2];
            }//End if

            // Calling __cpuid with 0x80000000 as the function_id argument
            // gets the number of the highest valid extended ID.
            CPUID(cpui.data(), 0x80000000);
            nExIds_ = cpui[0];

            char brand[0x40];
            memset(brand, 0, sizeof(brand));

            for (int32_t i = 0x80000000; i <= nExIds_; ++i)
            {
                CPUIDEX(cpui.data(), i, 0);
                extdata_.push_back(cpui);
            }//End for

            // load bitset with flags for function 0x80000001
            if (nExIds_ >= 0x80000001)
            {
                f_81_ECX_ = extdata_[1][2];
                f_81_EDX_ = extdata_[1][3];
            }//End if

            // Interpret CPU brand string if reported
            if (nExIds_ >= 0x80000004)
            {
                memcpy(brand, extdata_[2].data(), sizeof(cpui));
                memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
                memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
                brand_ = brand;
            }//End if
        };//End of constructor
#elif defined(TPA_ARM)
public:
        /// <summary>
        /// Construct Instruction Set Availability Object
        /// </summary>
        InstructionSet() noexcept
        {
            vendor_ = "ARM";
            brand_ = "Unknown";
        }//End of constructor
#else
public:
        /// <summary>
        /// Construct Instruction Set Availability Object
        /// </summary>
        InstructionSet() noexcept
        {
            vendor_ = "Unknown";
            brand_ = "Unknown";
        }//End of constructor
#endif
        InstructionSet(InstructionSet const&) = delete;
        InstructionSet& operator=(InstructionSet const&) = delete;
        InstructionSet(InstructionSet&&) = delete;
        InstructionSet& operator=(InstructionSet&&) = delete;

        ~InstructionSet() = default;      
    };
}//End of namespace

namespace tpa{
static const tpa_cpuid_private::InstructionSet runtime_instruction_set;

#if defined(TPA_X86_64)
static const bool hasMMX = runtime_instruction_set.MMX();//Automatically set to true if system has MMX at runtime - note that MMX intrinsics should be avoided as Intel has deprecated them and down-clocked them severely in order to cripple thier performance to encourage the use of SSE or better!

static const bool has_SSE = runtime_instruction_set.SSE();//Automatically set to true if system has SSE at runtime
static const bool has_SSE2 = runtime_instruction_set.SSE2();//Automatically set to true if system has SSE2 at runtime
static const bool has_SSE3 = runtime_instruction_set.SSE3();//Automatically set to true if system has SSE3 at runtime
static const bool has_SSSE3 = runtime_instruction_set.SSSE3();//Automatically set to true if system has SSSE3 at runtime
static const bool has_SSE41 = runtime_instruction_set.SSE41();//Automatically set to true if system has SSE4.1 at runtime
static const bool has_SSE42 = runtime_instruction_set.SSE42();//Automatically set to true if system has SSE4.2 at runtime

static const bool hasAVX = runtime_instruction_set.AVX();//Automatically set to true if system has AVX at runtime
static const bool hasAVX2 = runtime_instruction_set.AVX2();//Automatically set to true if system has AVX2 at runtime
static const bool hasFMA = runtime_instruction_set.FMA();//Automatically set to true if system has FMA at runtime
static const bool hasAVX512 = runtime_instruction_set.AVX512F();//Automatically set to true if system has AVX512 (foundation) at runtime
static const bool hasAVX512_ByteWord = runtime_instruction_set.AVX512BW();//Automatically set to true if system has AVX512 Byte & Word Instructions at runtime
static const bool hasAVX512_DWQW = runtime_instruction_set.AVX512DQ();//Automatically set to true if system has AVX512 Double-Word and Quad-Word Instructions at runtime

static const bool hasBMI1 = runtime_instruction_set.BMI1();//Automatically set to true if system has BMI1 Instructions at runtime
static const bool hasBMI2 = runtime_instruction_set.BMI2();//Automatically set to true if system has BMI2 Instructions at runtime
static const bool hasPOPCNT = runtime_instruction_set.POPCNT();//Automatically set to true if system has POP COUNT Instructions at runtime
static const bool hasLZCNT = runtime_instruction_set.LZCNT();//Automatically set to true if system has Leading Zero Count Instructions at runtime
static const bool hasABM = runtime_instruction_set.ABM();//Automatically set to true if system has ABM Instructions at runtime

static const bool hasRD_RAND = runtime_instruction_set.RDRAND() && runtime_instruction_set.RDSEED();//Automatically set to true if system has Random Number Instructions at runtime

#elif defined(TPA_ARM)
    
static const bool hasNeon = runtime_instruction_set.NEON(); // Automatically set to true if system has NEON Instructions at compile time (required for TPA on ARM)
static const bool has_SVE = runtime_instruction_set.SVE(); // Automatically set to true if system has SVE Instructions at compile time (required for TPA on ARM)
static const bool has_SVE2 = runtime_instruction_set.SVE2(); // Automatically set to true if system has SVE2 Instructions at compile time (required for TPA on ARM)
static const bool hasHelium = runtime_instruction_set.HELIUM(); // Automatically set to true if system has HELIUM Instructions at compile time (required for TPA on ARM)
#endif

}//End of namespace