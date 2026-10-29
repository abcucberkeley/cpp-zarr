// How the parallel reader and writer share out their chunks and set up their
// threads (internal to the library, not installed)
#ifndef CPPZARR_PARALLELWORK_H
#define CPPZARR_PARALLELWORK_H
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Contiguous blocks of work items with work stealing. Each thread starts with its
// own contiguous block of the items, so neighboring chunks stay with one thread and
// its writes to the array stay together (and on the thread's NUMA node); a thread
// that runs out takes the back half of the largest block left. Fixed blocks left up
// to a third of the threads' time idle when chunks took different times to decode or
// compress, and handing chunks out one at a time from a shared counter split
// neighboring chunks between threads, which made C-order reads up to 38% slower.
class WorkRanges {
    struct alignas(64) Range {
        std::mutex m;
        std::atomic<int64_t> begin{0}, end{0};
    };
    std::vector<Range> ranges;
public:
    WorkRanges(const int threads, const int64_t items) : ranges(threads > 0 ? threads : 1) {
        const int64_t t = (int64_t)ranges.size();
        for(int64_t i = 0; i < t; i++){
            ranges[i].begin.store(items*i/t, std::memory_order_relaxed);
            ranges[i].end.store(items*(i+1)/t, std::memory_order_relaxed);
        }
    }
    // The item thread t takes next from its own block, or -1 (another thread may still
    // take it first)
    int64_t peek(const int t) const {
        const int64_t b = ranges[t].begin.load(std::memory_order_relaxed);
        return b < ranges[t].end.load(std::memory_order_relaxed) ? b : -1;
    }
    // The end of thread t's own block (its items before it may still be taken by others)
    int64_t ownEnd(const int t) const { return ranges[t].end.load(std::memory_order_relaxed); }
    // The next item of thread t's own block, or -1 (none taken from other threads)
    int64_t nextOwn(const int t){
        std::lock_guard<std::mutex> g(ranges[t].m);
        const int64_t b = ranges[t].begin.load(std::memory_order_relaxed);
        if(b >= ranges[t].end.load(std::memory_order_relaxed)) return -1;
        ranges[t].begin.store(b+1, std::memory_order_relaxed);
        return b;
    }
    // The next item for thread t, or -1 once none is left anywhere
    int64_t next(const int t){
        {
            std::lock_guard<std::mutex> g(ranges[t].m);
            const int64_t b = ranges[t].begin.load(std::memory_order_relaxed);
            if(b < ranges[t].end.load(std::memory_order_relaxed)){
                ranges[t].begin.store(b+1, std::memory_order_relaxed);
                return b;
            }
        }
        while(true){
            int victim = -1;
            int64_t most = 0;
            for(int v = 0; v < (int)ranges.size(); v++){
                const int64_t left = ranges[v].end.load(std::memory_order_relaxed)-ranges[v].begin.load(std::memory_order_relaxed);
                if(left > most){
                    most = left;
                    victim = v;
                }
            }
            if(victim < 0) return -1;
            int64_t first, last;
            {
                std::lock_guard<std::mutex> g(ranges[victim].m);
                const int64_t b = ranges[victim].begin.load(std::memory_order_relaxed);
                last = ranges[victim].end.load(std::memory_order_relaxed);
                if(last <= b) continue;
                first = last-(last-b+1)/2;
                ranges[victim].end.store(first, std::memory_order_relaxed);
            }
            {
                std::lock_guard<std::mutex> g(ranges[t].m);
                ranges[t].begin.store(first+1, std::memory_order_relaxed);
                ranges[t].end.store(last, std::memory_order_relaxed);
            }
            return first;
        }
    }
};

// A pool of the buffers and coders a thread uses (T, with a shrink() that lets go of
// anything too big to keep), kept between calls. Allocating them in every call had
// glibc grow and trim each thread's heap in every call, and small calls spent most of
// their time faulting that memory back in and in mprotect and munmap calls, which
// stall every thread of the process. A thread gets back the ones it gave back last
// (place: its thread number) unless another call took them: handed out in any order,
// they were in other cores' caches and on the other socket's memory, which made small
// writes up to 8% slower.
template <typename T>
class ScratchPool {
    std::mutex m;
    std::vector<std::unique_ptr<T>> byPlace, idle;
public:
    // place < 0: any
    std::unique_ptr<T> take(const int place){
        {
            std::lock_guard<std::mutex> g(m);
            if(place >= 0 && place < (int)byPlace.size() && byPlace[place]) return std::move(byPlace[place]);
            if(!idle.empty()){
                std::unique_ptr<T> s = std::move(idle.back());
                idle.pop_back();
                return s;
            }
            for(auto &b : byPlace) if(b) return std::move(b);
        }
        return std::unique_ptr<T>(new T());
    }
    void give(const int place, std::unique_ptr<T> s){
        if(!s) return;
        s->shrink();
        std::lock_guard<std::mutex> g(m);
        if(place >= 0){
            if(place >= (int)byPlace.size()) byPlace.resize(place+1);
            if(!byPlace[place]){
                byPlace[place] = std::move(s);
                return;
            }
        }
        idle.push_back(std::move(s));
    }
};

// Wait until done() is true (and return true), or until stop() is (and return false).
// The thread yields its CPU at first, then sleeps a little longer each time up to 0.16
// ms: a long wait leaves the CPU to others (the thread being waited for among them, when
// the system has paused it), while a short one ends soon after.
template <typename Done, typename Stop>
bool waitFor(const Done &done, const Stop &stop){
    for(int i = 0; !done(); i++){
        if(stop()) return false;
        if(i < 32) std::this_thread::yield();
        else std::this_thread::sleep_for(std::chrono::microseconds(10 << std::min(i-32, 4)));
    }
    return true;
}

// For the length of a call, a thread with the default memory policy gets an explicit
// local one: its memory still comes from its own NUMA node, but Linux's automatic NUMA
// balancing skips threads with an explicit policy. Otherwise long-running threads (as
// in MATLAB) have the kernel mark the process's pages, and writes took up to millions
// of hinting faults and page migrations a call (8-30% of their time). A thread with a
// policy of its own (numactl, say) keeps it.
class LocalMemoryPolicy {
#if defined(__linux__) && defined(SYS_get_mempolicy) && defined(SYS_set_mempolicy)
    bool set = false;
public:
    LocalMemoryPolicy(){
        int mode = -1;
        if(syscall(SYS_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) == 0 && mode == 0 /* MPOL_DEFAULT */){
            set = syscall(SYS_set_mempolicy, 4 /* MPOL_LOCAL */, nullptr, 0UL) == 0;
        }
    }
    ~LocalMemoryPolicy(){
        if(set) syscall(SYS_set_mempolicy, 0 /* MPOL_DEFAULT */, nullptr, 0UL);
    }
#else
public:
    LocalMemoryPolicy(){}
#endif
    LocalMemoryPolicy(const LocalMemoryPolicy&) = delete;
    LocalMemoryPolicy& operator=(const LocalMemoryPolicy&) = delete;
};

// Threads that read files for the calls of the process, so that a call keeps more reads in
// flight than it has threads of its own (reading from network storage mostly waits on the
// server). Jobs run in the order they were given, as many at once as the pool has threads;
// threads are started as calls ask for more (up to the most any asked for) and then kept,
// blocked while there is nothing to read (not spinning, which would take the CPU from
// other threads and processes). They end, once the jobs given are done, when the pool does:
// when the library is unloaded (MATLAB's clear mex) or the process ends, as their code goes
// with it.
class IoPool {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::function<void()>> jobs;
    std::vector<std::thread> threads;
    int started = 0, idle = 0;
    bool stop = false;
    void work(){
        const LocalMemoryPolicy policy;
        std::unique_lock<std::mutex> lk(m);
        while(true){
            idle++;
            cv.wait(lk, [&]{ return stop || !jobs.empty(); });
            idle--;
            if(jobs.empty()) return;
            std::function<void()> j = std::move(jobs.front());
            jobs.pop_front();
            lk.unlock();
            j();
            lk.lock();
        }
    }
public:
    IoPool(){}
    IoPool(const IoPool&) = delete;
    IoPool& operator=(const IoPool&) = delete;
    ~IoPool(){
        {
            std::lock_guard<std::mutex> g(m);
            stop = true;
        }
        cv.notify_all();
        for(std::thread &t : threads) if(t.joinable()) t.join();
    }
    // Runs job on a thread of the pool, one more started (up to most) when none is idle: as
    // many as the reads a process keeps in flight need, so that a call of a few reads does not
    // start them all (64 thread starts took about 3 ms). A job must not throw. Throws (and does
    // not run job) only when job cannot be queued.
    void run(const int most, std::function<void()> job){
        size_t slot = 0;
        bool start = false;
        {
            std::lock_guard<std::mutex> g(m);
            jobs.push_back(std::move(job));
            if((int64_t)jobs.size() > idle && started < most){
                // (the thread's place, taken first: once started, it must be kept to be joined)
                try{
                    threads.emplace_back();
                    slot = threads.size()-1;
                    start = true;
                    started++;
                }
                catch(const std::exception&){}
            }
        }
        cv.notify_one();
        if(!start) return;
        // (started outside the lock, which the pool's threads need to take their jobs)
        try{
            std::thread t(&IoPool::work, this);
            std::lock_guard<std::mutex> g(m);
            threads[slot] = std::move(t);
        }
        catch(const std::exception&){
            // (no thread could be started: the caller runs the jobs waiting instead, so that
            // every job runs)
            {
                std::lock_guard<std::mutex> g(m);
                started--;
            }
            while(true){
                std::function<void()> j;
                {
                    std::lock_guard<std::mutex> g(m);
                    if(jobs.empty()) break;
                    j = std::move(jobs.front());
                    jobs.pop_front();
                }
                j();
            }
        }
    }
};
// The process's pool, made on first use and ended with the library. A child process made by
// fork() has none of the parent's threads: it gets a pool of its own (the parent's, whose lock
// a thread may have held at the fork, is left alone).
inline std::atomic<IoPool*>& ioPoolSlot(){
    static std::atomic<IoPool*> p{nullptr};
    return p;
}
struct IoPoolOwner {
    IoPoolOwner(){
        ioPoolSlot().store(new IoPool(), std::memory_order_release);
#ifdef __linux__
        pthread_atfork(nullptr, nullptr, []{ ioPoolSlot().store(new IoPool(), std::memory_order_release); });
#endif
    }
    IoPoolOwner(const IoPoolOwner&) = delete;
    IoPoolOwner& operator=(const IoPoolOwner&) = delete;
    ~IoPoolOwner(){ delete ioPoolSlot().exchange(nullptr); }
};
inline IoPool& ioPool(){
    static IoPoolOwner owner;
    return *ioPoolSlot().load(std::memory_order_acquire);
}

// Reads in flight that a thread waits for: each calls done() as it ends (false: it failed)
struct Pending {
    std::atomic<int> left{0};
    std::atomic<bool> ok{true};
    void add(const int n){ left.fetch_add(n, std::memory_order_relaxed); }
    void done(const bool good){
        if(!good) ok.store(false, std::memory_order_relaxed);
        left.fetch_sub(1, std::memory_order_release);
    }
    bool idle() const { return left.load(std::memory_order_acquire) == 0; }
};

#endif
