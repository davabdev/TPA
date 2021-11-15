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

#ifdef _M_AMD64
    #include <immintrin.h>

#ifdef _WIN32
        #include <intrin.h>
#define CPUID(registers, function) __cpuid((int*)registers, (int)function);
#define CPUIDEX(registers, function, extFunction) __cpuidex((int*)registers, (int)function, (int)extFunction);
#else 
#define CPUID(registers, function) asm volatile ("cpuid" : "=a" (registers[0]), "=b" (registers[1]), "=c" (registers[2]), "=d" (registers[3]) : "a" (function), "c" (0));
#define CPUIDEX(registers, function, extFunction) asm volatile ("cpuid" : "=a" (registers[0]), "=b" (registers[1]), "=c" (registers[2]), "=d" (registers[3]) : "a" (function), "c" (extFunction));
#endif
#elif defined(_M_ARM64)
#ifdef _WIN32
    #include "arm64_neon.h"
#else
    #include "arm_neon.h"
#endif
#endif

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

#if defined(_M_AMD64)        
        
        //MMX
        bool MMX(void) const noexcept { return f_1_EDX_[23]; }
        bool MMXEXT(void) const noexcept { return isAMD_ && f_81_EDX_[22]; }

        //3D_Now!
        bool _3DNOW(void) const noexcept { return isAMD_ && f_81_EDX_[31]; }
        bool _3DNOWEXT(void) const noexcept { return isAMD_ && f_81_EDX_[30]; }

        //SSE
        bool SSE(void) const noexcept { return f_1_EDX_[25]; }
        bool SSE2(void) const noexcept { return f_1_EDX_[26]; }
        bool SSE3(void) const noexcept { return f_1_ECX_[0]; }
        bool SSSE3(void) const noexcept { return f_1_ECX_[9]; }
        bool SSE4a(void) const noexcept { return isAMD_ && f_81_ECX_[6]; }
        bool SSE41(void) const noexcept { return f_1_ECX_[19]; }
        bool SSE42(void) const noexcept { return f_1_ECX_[20]; }
        bool XOP(void) const noexcept { return isAMD_ && f_81_ECX_[11]; }

        //AVX
        bool AVX(void) const noexcept { return f_1_ECX_[28]; }
        bool AVX2(void) const noexcept { return f_7_EBX_[5]; }
        bool FMA(void) const noexcept { return f_1_ECX_[12]; }
        bool AVX_VNNI(void) const noexcept { return f_7_ECX_[4]; /*eax bit 4*/ }

        //AVX-512 and extentions
        bool AVX512F(void) const noexcept { return f_7_EBX_[16]; }
        bool AVX512PF(void) const noexcept { return f_7_EBX_[26]; }
        bool AVX512ER(void) const noexcept { return f_7_EBX_[27]; }
        bool AVX512CD(void) const noexcept { return f_7_EBX_[28]; }

        bool AVX512BW(void) const noexcept { return f_7_EBX_[30]; }
        bool AVX512VL(void) const noexcept { return f_7_EBX_[31]; }
        bool AVX512DQ(void) const noexcept { return f_7_EBX_[17]; }

        bool IFMA(void) const noexcept { return f_7_EBX_[21]; }

        bool VBMI(void) const noexcept { return f_7_ECX_[1]; }
        bool VBMI2(void) const noexcept { return f_7_ECX_[6]; }

        bool VNNI(void) const noexcept { return f_7_ECX_[11]; }
        bool FOUR_VNNIW(void) const noexcept { return f_81_EDX_[2]; }
        bool FOUR_MAPS(void) const noexcept { return f_81_EDX_[3]; }

        bool VPOPCNTDQ(void) const noexcept { return f_7_EBX_[32]; }

        bool BITALG(void) const noexcept { return f_7_ECX_[12]; }

        bool VP2INTERSECT(void) const noexcept { return f_81_EDX_[8]; }

        bool GFNI(void) const noexcept { return f_7_ECX_[8]; }
        bool VPCLMULQDQ(void) const noexcept { return f_7_ECX_[10]; }
        
        bool PREFETCHWT1(void) const noexcept { return f_7_ECX_[0]; }

        /// <summary>
        /// Knights Landing
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool KNCNI(void) const noexcept
        {
            return f_7_EBX_[16] && f_7_EBX_[28] &&
                f_7_EBX_[27] && f_7_EBX_[26];
        }

        //AMX
        bool AMXBF16(void) const noexcept { return f_81_EDX_[22]; /*eax bit 5*/ }
        bool AVX512_FP16(void) const noexcept { return f_81_EDX_[23]; }
        bool AMXTILE(void) const noexcept { return f_81_EDX_[24]; }
        bool AMXINT8(void) const noexcept { return f_81_EDX_[25]; }

        /*
        * Cryptography & Security
        */

        bool AES(void) const noexcept { return f_1_ECX_[25]; }

        //AVX-512 (Vector AES)
        bool VAES(void) const noexcept { return f_7_ECX_[9]; }

        bool SHA(void) const noexcept { return f_7_EBX_[29]; }

        bool SGX(void) const noexcept { return f_7_EBX_[2]; }

        bool KEYLOCKER(void) const noexcept { return f_1_ECX_[23]; }
        bool AESKLE(void) const noexcept { return f_7_EBX_[0]; }

        //Suppervisor Mode Access Prevention
        bool SMAP(void) const noexcept { return f_7_EBX_[20]; }

        //Suppervisor Mode Execution Prevention
        bool SMEP(void) const noexcept { return f_7_EBX_[7]; }

        //User Mode Instruction Prevention
        bool UMIP(void) const noexcept { return f_1_ECX_[2]; }

        //Protected Keys for Suppervisor Mode
        bool PKS(void) const noexcept { return f_1_ECX_[31]; }

        //Protected Keys for User Mode
        bool PKU(void) const noexcept { return f_7_ECX_[3]; }
        bool OSPKE(void) const noexcept {  return f_7_ECX_[4]; }

        bool TDX(void) const noexcept { return false; }

        bool VMX(void) const noexcept { return f_1_ECX_[5]; }
        bool SMX(void) const noexcept { return f_1_ECX_[6]; }

        //Bit Manipulation
        bool BMI1(void) const noexcept { return f_7_EBX_[3]; }
        bool BMI2(void) const noexcept { return f_7_EBX_[8]; }
        bool ABM(void) const noexcept { return isAMD_ && f_81_ECX_[5]; }
        bool TBM(void) const noexcept { return isAMD_ && f_81_ECX_[21]; }
        bool POPCNT(void) const noexcept { return f_1_ECX_[23]; }
        bool LZCNT(void) const noexcept { return isIntel_ && f_81_ECX_[5]; }

        //Advanced Move Instructions
        bool CMOV(void) const noexcept { return f_1_EDX_[15]; }
        bool REP_MOV(void) const noexcept { return f_81_EDX_[4]; }
        bool MOVBE(void) const noexcept { return f_1_ECX_[22]; }
        bool MOVDIRI(void) const noexcept { return f_1_ECX_[27]; }
        bool MOVDIR64B(void) const noexcept { return f_1_ECX_[28]; }
        bool ENQCMD(void) const noexcept { return f_1_ECX_[29]; }
        bool CMPXCHG16B(void) const noexcept { return f_1_ECX_[13]; }
        bool CX8(void) const noexcept { return f_81_EDX_[8]; }
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
        bool RDRAND(void) const noexcept { return f_1_ECX_[30]; }
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
        bool RDSEED(void) const noexcept { return f_7_EBX_[18]; }

        bool RTM(void) const noexcept { return isIntel_ && f_7_EBX_[11]; }
        bool RDT_M(void) const noexcept { return f_7_EBX_[12]; }
        bool RDT_A(void) const noexcept { return f_7_EBX_[15]; }
                
        bool LAHF(void) const noexcept { return f_81_ECX_[0]; }
        bool SYSCALL(void) const noexcept { return isIntel_ && f_81_EDX_[11]; }
        bool RDTSCP(void) const noexcept { return isIntel_ && f_81_EDX_[27]; }

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

#elif defined(_M_ARM64)
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
#endif

    private:
        std::string vendor_ = {};
        std::string brand_ = {};

#if defined(_M_AMD64)
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

            for (size_t i = 0; i <= nIds_; ++i)
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
#elif defined (_M_ARM64)
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

#if defined(_M_AMD64)
static const bool hasMMX = runtime_instruction_set.MMX();//Automatically set to true if system has MMX at runtime - note that MMX intrinsics should be avoided as Intel has deprecated them and down-clocked them severely in order to cripple thier performance to encourage the use of SSE or better!

static const bool has_SSE = runtime_instruction_set.SSE();//Automatically set to true if system has SSE at runtime
static const bool has_SSE2 = runtime_instruction_set.SSE2();//Automatically set to true if system has SSE2 at runtime
static const bool has_SSE3 = runtime_instruction_set.SSE3();//Automatically set to true if system has SSE3 at runtime
static const bool has_SSSE3 = runtime_instruction_set.SSSE3();//Automatically set to true if system has SSSE3 at runtime
static const bool has_SSE41 = runtime_instruction_set.SSE41();//Automatically set to true if system has SSE41 at runtime
static const bool has_SSE42 = runtime_instruction_set.SSE42();//Automatically set to true if system has SSE42 at runtime

static const bool hasAVX = runtime_instruction_set.AVX();//Automatically set to true if system has AVX at runtime
static const bool hasAVX2 = runtime_instruction_set.AVX2();//Automatically set to true if system has AVX2 at runtime
static const bool hasFMA = runtime_instruction_set.FMA();//Automatically set to true if system has FMA at runtime
static const bool hasAVX512 = runtime_instruction_set.AVX512F();//Automatically set to true if system has AVX512 (foundation) at runtime
static const bool hasAVX512_ByteWord = runtime_instruction_set.AVX512BW();//Automatically set to true if system has AVX512 Byte & Word Instructions at runtime
static const bool hasAVX512_DWQW = runtime_instruction_set.AVX512DQ();//Automatically set to true if system has AVX512 Double-Word and Quad-Word Instructions at runtime
#elif defined(_M_ARM64)
    
static const bool hasNeon = runtime_instruction_set.NEON(); // Automatically set to true if system has NEON Instructions at compile time (required for TPA on ARM)
static const bool has_SVE = runtime_instruction_set.SVE(); // Automatically set to true if system has SVE Instructions at compile time (required for TPA on ARM)
static const bool has_SVE2 = runtime_instruction_set.SVE2(); // Automatically set to true if system has SVE2 Instructions at compile time (required for TPA on ARM)
static const bool hasHelium = runtime_instruction_set.HELIUM(); // Automatically set to true if system has HELIUM Instructions at compile time (required for TPA on ARM)
#endif

}//End of namespace