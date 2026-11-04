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
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <pthread.h>
#endif
#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
// (Linux 4.5; older headers lack it, and an older kernel refuses it, which leaves the memory as it was)
#ifndef MADV_FREE
#define MADV_FREE 8
#endif
#endif

// A buffer kept for later calls given back to the system lazily while it is idle (Linux's
// MADV_FREE, on the whole pages of its first n bytes): it stays the process's, and is reused
// without page faults, unless the system or the job's memory limit needs the memory, which
// then takes it instead of the job running out (its contents are lost then; a buffer kept is
// always written before it is read). Kept buffers up to the cap pushed jobs sized close to
// what they need over their limit: a 4-CPU PetaKit5D task that runs in a job of 2000 MB was
// killed in one of 2050 MB.
inline void freeLazily(void* p, const uint64_t n){
#ifdef __linux__
    static const uintptr_t page = (uintptr_t)std::max(4096L, sysconf(_SC_PAGESIZE));
    const uintptr_t a = ((uintptr_t)p+page-1) & ~(page-1), b = ((uintptr_t)p+n) & ~(page-1);
    if(p && b > a) madvise(reinterpret_cast<void*>(a), b-a, MADV_FREE);
#else
    (void)p;
    (void)n;
#endif
}

// fork() and the library. A child made by fork() has only the thread that forked, and the rest
// of the library's state as it was then:
// - GCC's OpenMP (libgomp) keeps the thread's team of threads, which are not in the child: its
//   next parallel region of more than one thread waits for them forever (a training loader's
//   workers, made by fork() after the main process had read with more than one thread, hung at
//   their first read). So in such a child, the functions that run parallel regions run them on
//   a thread of the child when called on the thread that forked (onRegionThread): OpenMP makes a
//   team anew for it. Threads the child makes have teams of their own.
// - A lock that another thread held at the fork would be held in the child for good: the locks of
//   the library's pools of buffers are taken around fork() (guardAcrossFork), so that the child
//   gets the pools whole and unlocked.
class ForkSafety {
    // (a thread of a child for the parallel regions of the thread that forked)
    class RegionThread {
        std::mutex m;
        std::condition_variable cv;
        const std::function<void()>* job = nullptr;
        std::exception_ptr error;
        bool busy = false, stop = false;
        std::thread t;
        void work(){
            std::unique_lock<std::mutex> lk(m);
            while(true){
                cv.wait(lk, [&]{ return busy || stop; });
                if(!busy) return;
                lk.unlock();
                std::exception_ptr e;
                try{
                    (*job)();
                }
                catch(...){
                    e = std::current_exception();
                }
                lk.lock();
                error = e;
                busy = false;
                cv.notify_all();
            }
        }
    public:
        RegionThread() : t(&RegionThread::work, this) {}
        RegionThread(const RegionThread&) = delete;
        RegionThread& operator=(const RegionThread&) = delete;
        ~RegionThread(){
            {
                std::lock_guard<std::mutex> g(m);
                stop = true;
            }
            cv.notify_all();
            if(t.joinable()) t.join();
        }
        void run(const std::function<void()> &f){
            std::unique_lock<std::mutex> lk(m);
            job = &f;
            busy = true;
            cv.notify_all();
            cv.wait(lk, [&]{ return !busy; });
            job = nullptr;
            if(error){
                std::exception_ptr e = error;
                error = nullptr;
                std::rethrow_exception(e);
            }
        }
    };
    std::mutex* guarded[8];
    std::atomic<int> nGuarded{0};
#ifndef _WIN32
    std::atomic<int> forked{0};
    pthread_t forker;
    std::mutex regionMutex;
    std::atomic<RegionThread*> regions{nullptr};
    static void prepare(){
        ForkSafety &f = get();
        for(int i = 0; i < f.nGuarded.load(std::memory_order_acquire); i++) f.guarded[i]->lock();
    }
    static void parent(){
        ForkSafety &f = get();
        for(int i = f.nGuarded.load(std::memory_order_acquire)-1; i >= 0; i--) f.guarded[i]->unlock();
    }
    static void child(){
        ForkSafety &f = get();
        for(int i = f.nGuarded.load(std::memory_order_acquire)-1; i >= 0; i--) f.guarded[i]->unlock();
        // (the parent's region thread, if any, is not in the child: left alone)
        new (&f.regionMutex) std::mutex();
        f.regions.store(nullptr, std::memory_order_relaxed);
        f.forker = pthread_self();
        f.forked.fetch_add(1, std::memory_order_release);
    }
#endif
    ForkSafety(){
#ifndef _WIN32
        pthread_atfork(&ForkSafety::prepare, &ForkSafety::parent, &ForkSafety::child);
#endif
    }
public:
    // The process's (made at the library's start, by the pools of buffers; never destroyed, as a
    // child of a fork may run its handlers until it ends)
    static ForkSafety& get(){
        static ForkSafety* f = new ForkSafety();
        return *f;
    }
    // Hold m around fork() (m must never be held while taking another lock)
    void guardAcrossFork(std::mutex &m){
        const int i = nGuarded.load(std::memory_order_relaxed);
        if(i < 8){
            guarded[i] = &m;
            nGuarded.store(i+1, std::memory_order_release);
        }
    }
    // Whether this thread's parallel regions must run on another (see above)
    bool needRegionThread(){
#ifndef _WIN32
        return forked.load(std::memory_order_acquire) != 0 && pthread_equal(pthread_self(), forker);
#else
        return false;
#endif
    }
    // Run f on the child's thread for the parallel regions of the thread that forked (made on first
    // use), and wait for it; an exception f throws is thrown here
    void onRegionThread(const std::function<void()> &f){
#ifndef _WIN32
        RegionThread* r = regions.load(std::memory_order_acquire);
        if(!r){
            std::lock_guard<std::mutex> g(regionMutex);
            r = regions.load(std::memory_order_acquire);
            if(!r){
                r = new RegionThread();
                regions.store(r, std::memory_order_release);
            }
        }
        r->run(f);
#else
        f();
#endif
    }
    // (at the library's end: the region thread of a child, if any, is ended)
    void end(){
#ifndef _WIN32
        delete regions.exchange(nullptr);
#endif
    }
};
// Ends the region thread with the library (its code goes with it)
struct ForkSafetyOwner {
    ForkSafetyOwner(){ ForkSafety::get(); }
    ~ForkSafetyOwner(){ ForkSafety::get().end(); }
};

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
    ScratchPool(){ ForkSafety::get().guardAcrossFork(m); }
    ScratchPool(const ScratchPool&) = delete;
    ScratchPool& operator=(const ScratchPool&) = delete;
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
    // f run on every scratch the pool holds (under its lock; those calls are using are not in it)
    template <typename F>
    void forEach(const F &f){
        std::lock_guard<std::mutex> g(m);
        for(auto &b : byPlace) if(b) f(*b);
        for(auto &s : idle) if(s) f(*s);
    }
    // Every scratch the pool holds freed (those calls are using are not in it, and come back)
    void clear(){
        std::vector<std::unique_ptr<T>> gone;
        {
            std::lock_guard<std::mutex> g(m);
            for(auto &b : byPlace) if(b) gone.push_back(std::move(b));
            for(auto &s : idle) gone.push_back(std::move(s));
            byPlace.clear();
            idle.clear();
        }
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
// when the process ends, or when the library is unloaded, as their code goes with it. (On
// Linux glibc does not unload the library MATLAB's mex files load: the unique symbols of its
// static C++ runtime keep it loaded, so clear mex unloads the mex files only.)
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
