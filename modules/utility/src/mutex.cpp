#include "utility.hpp"

#include <stdexcept>

#if defined(OPENCV_OS_FAMILY_WINDOWS)

//
// OPENCV_OS_FAMILY_WINDOWS
//

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

class cv::Mutex::Impl
{
public:
    Impl();
    ~Impl();

    void lockImpl();
    bool tryLockImpl();
    void unlockImpl();

private:
    CRITICAL_SECTION cs_;
};

inline cv::Mutex::Impl::Impl()
{
    InitializeCriticalSectionAndSpinCount(&cs_, 4000);
}

inline cv::Mutex::Impl::~Impl()
{
    DeleteCriticalSection(&cs_);
}

inline void cv::Mutex::Impl::lockImpl()
{
    EnterCriticalSection(&cs_);
}

inline bool cv::Mutex::Impl::tryLockImpl()
{
    return TryEnterCriticalSection(&cs_) != 0;
}

inline void cv::Mutex::Impl::unlockImpl()
{
    LeaveCriticalSection(&cs_);
}

#else

//
// POSIX
//

#include <pthread.h>
#include <errno.h>

class cv::Mutex::Impl
{
public:
    Impl();
    ~Impl();

    void lockImpl();
    bool tryLockImpl();
    void unlockImpl();

private:
    pthread_mutex_t mutex_;
};

inline cv::Mutex::Impl::Impl()
{
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);

#if defined(PTHREAD_MUTEX_RECURSIVE_NP)
    pthread_mutexattr_settype_np(&attr, PTHREAD_MUTEX_NORMAL_NP);
#elif !defined(OPENCV_VXWORKS)
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_NORMAL);
#endif

    if (pthread_mutex_init(&mutex_, &attr))
    {
        pthread_mutexattr_destroy(&attr);
        throw std::runtime_error("Can't create mutex");
    }

    pthread_mutexattr_destroy(&attr);
}

inline cv::Mutex::Impl::~Impl()
{
    pthread_mutex_destroy(&mutex_);
}

inline void cv::Mutex::Impl::lockImpl()
{
    if (pthread_mutex_lock(&mutex_))
        throw std::runtime_error("Can't lock mutex");
}

inline bool cv::Mutex::Impl::tryLockImpl()
{
    int rc = pthread_mutex_trylock(&mutex_);
    if (rc == 0)
        return true;
    else if (rc == EBUSY)
        return false;
    else
        throw std::runtime_error("Can't lock mutex");
}

inline void cv::Mutex::Impl::unlockImpl()
{
    if (pthread_mutex_unlock(&mutex_))
        throw std::runtime_error("Can't unlock mutex");
}

#endif

cv::Mutex::Mutex() : impl_(0)
{
    impl_ = new Impl;
}

cv::Mutex::~Mutex()
{
    delete impl_;
}

void cv::Mutex::lock()
{
    impl_->lockImpl();
}

bool cv::Mutex::tryLock()
{
    return impl_->tryLockImpl();
}

void cv::Mutex::unlock()
{
    impl_->unlockImpl();
}
