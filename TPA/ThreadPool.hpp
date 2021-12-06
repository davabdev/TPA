#pragma once
/*
* Truly Parallel Algorithms Library - Thread Pool
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

#include <queue>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <barrier>
#include <functional>
#include <future>
#include <memory>

#include "InstructionSet.hpp"
#include "size_t_lit.hpp"

//#define USE_GENERIC_THREAD_POOL

/// <summary>
/// <para>Truly Parallel Algorithms</para>
/// <para>By David Aaron Braun</para>
/// <para>Version 0.1</para> 
/// </summary>
namespace tpa {
	static const uint32_t nThreads = std::thread::hardware_concurrency();
}//End of namespace

/// <summary>
/// This namespace is used to store a Thread Pool Class, it is not inteded to be directly accesed by users of this library. Access the Thread Pool through 'tpa::tp' instead.
/// </summary>
namespace tpa_thread_pool_private {

#if defined(_WIN32) && defined(_M_AMD64) && !defined(USE_GENERIC_THREAD_POOL)
#include <Windows.h>
	//Note: Many win32 api declspec types cannot be a member of a class
	//Note: queue must be outside class or causes access violation
	std::atomic<uint32_t> barrierCount = static_cast<uint32_t>(tpa::nThreads + 1);
	SYNCHRONIZATION_BARRIER barrierSB;

	//Wake Conditon
	CRITICAL_SECTION wakeCS;
	CRITICAL_SECTION consoleCS;
	CONDITION_VARIABLE wakeCV;
	std::atomic<bool> morePossibleWork = true;
	std::queue<std::function<void()>> tasks;//Queue of funtion pointers to store tasks
#endif

	class ThreadPool
	{
	public:
		/// <summary>
		/// Returns a static reference to the Thread Pool singleton object
		/// </summary>
		/// <returns></returns>
		static ThreadPool& instance() noexcept
		{
			static ThreadPool INSTANCE;
			return INSTANCE;
		}//Singleton Constructor

		ThreadPool(ThreadPool const&) = delete;
		ThreadPool& operator=(ThreadPool const&) = delete;
		ThreadPool(ThreadPool&&) = delete;
		ThreadPool& operator=(ThreadPool&&) = delete;

#if defined(_WIN32) && defined(_M_AMD64) && !defined(USE_GENERIC_THREAD_POOL)

	//Windows Implementation
	private:
		std::vector<HANDLE> threads;	//Vector to store handles to threads

		inline DWORD WINAPI performTask(LPVOID) noexcept
		{
			EnterSynchronizationBarrier(&barrierSB, 0);	//Check all threads exist

			for (;;)
			{
				EnterCriticalSection(&wakeCS);

				while (tasks.empty() && morePossibleWork)
					SleepConditionVariableCS(&wakeCV, &wakeCS, INFINITE);

				if (tasks.empty() && !morePossibleWork) {
					LeaveCriticalSection(&wakeCS);
					break;
				}

				std::function<void()> t;	//Store a new task
				bool haveTask = false;
				if (!tasks.empty()) {
					t = std::move(tasks.front());
					tasks.pop();
					haveTask = true;
				}
				LeaveCriticalSection(&wakeCS);

				if (haveTask)
				{
					//Run task
					t();
				}//End if
			}//End infinite for
			return 0;
		}//End of perfromTask()

		/// <summary>
		/// Nessessary when passing a member function to CreateThread
		/// </summary>
		/// <param name="Param"></param>
		/// <returns></returns>
		inline static DWORD WINAPI StaticThreadStart(LPVOID Param) noexcept
		{
			ThreadPool* This = (ThreadPool*)Param;
			return This->performTask(Param);
		}//End of StaticThreadStart

		/// <summary>
		/// Creates an instance of a thread pool object and prepares threads for work
		/// </summary>
		ThreadPool() noexcept
		{

#ifdef _DEBUG
			std::cout << "Using Windows Native Thread Pool.\n";
#endif

			//Prepare Kernal Barriers
			InitializeCriticalSection(&wakeCS);
			InitializeCriticalSection(&consoleCS);
			InitializeConditionVariable(&wakeCV);
			InitializeSynchronizationBarrier(&barrierSB, barrierCount, 10);

			//Prepare Threads
			threads.reserve(tpa::nThreads);
			for (size_t i = 0uz; i != tpa::nThreads; ++i)
			{
				threads.emplace_back(CreateThread(NULL, 0, StaticThreadStart, NULL, 0, NULL));
			}//End for

			//Set to higher priority and pin each thread to a logical core
			for (size_t i = 0uz; i < threads.size(); ++i)
			{
				SetThreadPriority(threads[i], THREAD_PRIORITY_HIGHEST);
				SetThreadAffinityMask(threads[i], static_cast<DWORD_PTR>(1ull << i));
			}//End for

			//Set Barrier
			EnterSynchronizationBarrier(&barrierSB, 0);

			//Clean up existing AVX Registar Data
#ifdef _M_AMD64
			if (tpa::hasAVX || tpa::hasAVX512)
			{
				//Clean Up Registers
				_mm256_zeroall();
			}//End else
#endif
		};//End of constructor

		~ThreadPool() noexcept
		{
			morePossibleWork = false;
			WakeAllConditionVariable(&wakeCV);

			// cleanup
			WaitForMultipleObjects((DWORD)threads.size(), threads.data(), TRUE, INFINITE);
			for (auto& t : threads)
			{
				CloseHandle(t);
			}//End for

			threads.clear();

			DeleteCriticalSection(&wakeCS);
			DeleteCriticalSection(&consoleCS);
			DeleteSynchronizationBarrier(&barrierSB);

			//Clean up AVX Registar Data
#ifdef _M_AMD64
			if (tpa::hasAVX || tpa::hasAVX512)
			{
				//Clean Up Registers
				_mm256_zeroall();
			}//End else
#endif
		}//End of destructor
	public:
		/// <summary>
		/// Add a task to the thread pool
		/// </summary>
		template <class T>
		inline auto addTask(T f)->std::future<decltype(f())>
		{
			auto taskWrap = std::make_shared<std::packaged_task<decltype(f()) ()>>(std::move(f));

			{
				EnterCriticalSection(&wakeCS);
				tasks.emplace([=] {
					(*taskWrap)();
					});
				LeaveCriticalSection(&wakeCS);
			}

			WakeConditionVariable(&wakeCV);

			return taskWrap->get_future();
		}//End of addTask()	
#else
	private:
		std::vector<std::thread> threads;			//Vector to store threads
		std::queue<std::function<void()>> tasks;	//Queue of funtion pointers to store tasks
		std::mutex taskMTX;							//Mutex to lock 'tasks' queue

		//Wake Conditon
		std::mutex wakeMutex;
		std::condition_variable wakeCond;
		std::atomic<bool> morePossibleWork = true;

		//C++ 11 Barrier
		/*
		std::atomic<uint64_t> barrierThreshold = static_cast<uint64_t>(tpa::nThreads + 1);
		std::atomic<uint64_t> barrierCount = barrierThreshold.load();
		std::atomic<uint64_t> barrierGeneration = 0;
		std::mutex barrierMTX;
		std::condition_variable barrierCond;

		//C++ 11 User-Level Software Based Memory Barrier
		void barrier()
		{
			std::unique_lock<std::mutex> lock(barrierMTX);
			uint64_t gen = barrierGeneration;

			if (--barrierCount == 0)
			{
				++barrierGeneration;
				barrierCond.notify_all();
			}//End if
			else
			{
				while (gen == barrierGeneration)
				{
					barrierCond.wait(lock);
				}//End while
			}//End else
		}//End of barrier()
		*/

		//C++ 20 Barrier
		std::shared_ptr<std::barrier<>> barrier = std::make_shared<std::barrier<>>
			(static_cast<uint32_t>(tpa::nThreads + 1));

		inline void performTask()
		{
			barrier->arrive_and_wait();	//Check all threads exist

			while (morePossibleWork)
			{
				{
					std::unique_lock<std::mutex> lk(wakeMutex);
					wakeCond.wait(lk);
				}//End lock

				while (!tasks.empty())
				{
					std::function<void()> t;
					bool haveTask = false;
					{
						std::scoped_lock<std::mutex> lk(taskMTX);
						if (!tasks.empty())
						{
							t = tasks.front();
							tasks.pop();
							haveTask = true;
						}//End if
					}//End lock

					if (haveTask)
					{
						//Run task
						t();
					}//End if
				}//End while
			}//End while
		}//End of perfrom_task()

			/// <summary>
			/// Creates an instance of a thread pool object and prepares threads for work
			/// </summary>
		ThreadPool() noexcept
		{
#ifdef _DEBUG
			std::cout << "Using Generic Thread Pool.\n";
#endif
			//Prepare Threads
			threads.reserve(tpa::nThreads);
			for (size_t i = 0uz; i != tpa::nThreads; ++i)
			{
				threads.emplace_back([this] { this->performTask(); });
			}//End for 

			//Make sure all threads are ready to do work
			barrier->arrive_and_wait();

			//Clean up existing AVX Registar Data
#ifdef _M_AMD64
			if (tpa::hasAVX || tpa::hasAVX512)
			{
				//Clean Up Registers
				_mm256_zeroall();
			}//End else
#endif
		};//End of constructor

	public:
		~ThreadPool()
		{
			morePossibleWork = false;
			wakeCond.notify_all();

			for (auto& t : threads)
			{
				t.join();
			}//End for

			threads.clear();

			//Clean up AVX Registar Data
#ifdef _M_AMD64
			if (tpa::hasAVX || tpa::hasAVX512)
			{
				//Clean Up Registers
				_mm256_zeroall();
			}//End else
#endif
		}//End of destructor

		/// <summary>
		/// Add a task to the thread pool
		/// </summary>
		template <class T>
		inline auto addTask(T f)->std::future<decltype(f())>
		{
			auto taskWrap = std::make_shared<std::packaged_task<decltype(f()) ()>>(std::move(f));
			{
				std::scoped_lock<std::mutex> lk(taskMTX);

				tasks.emplace([=] {
					(*taskWrap)();
					});
			}//End lock
			wakeCond.notify_one();

			return taskWrap->get_future();
		}//End of AddTask()
#endif
	};//End of class ThreadPool
}//End of namespace