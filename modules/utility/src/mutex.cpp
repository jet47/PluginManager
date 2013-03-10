#include "utility.hpp"

#include <stdexcept>

#if defined(OPENCV_OS_FAMILY_WINDOWS)

    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>

    #if defined(_WIN32_WCE)
        class cv::Mutex::Impl
        {
        public:
            Impl();
            ~Impl();

            void lockImpl();
            bool tryLockImpl();
            void unlockImpl();

        private:
            HANDLE mutex_;
        };

        cv::Mutex::Impl::Impl()
        {
            mutex_ = CreateMutexW(NULL, FALSE, NULL);
            if (!mutex_) throw std::runtime_error("cannot create mutex");
        }

        cv::Mutex::Impl::::~Impl()
        {
            CloseHandle(mutex_);
        }

        void cv::Mutex::Impl::lockImpl()
        {
            switch (WaitForSingleObject(mutex_, INFINITE))
            {
            case WAIT_OBJECT_0:
                return;
            default:
                throw std::runtime_error("cannot lock mutex");
            }
        }

        bool cv::Mutex::Impl::tryLockImpl()
        {
            switch (WaitForSingleObject(mutex_, 0))
            {
            case WAIT_TIMEOUT:
                return false;
            case WAIT_OBJECT_0:
                return true;
            default:
                throw std::runtime_error("cannot lock mutex");
            }
        }

        void cv::Mutex::Impl::unlockImpl()
        {
            ReleaseMutex(mutex_);
        }
    #else
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

        cv::Mutex::Impl::Impl()
        {
            InitializeCriticalSectionAndSpinCount(&cs_, 4000);
        }

        cv::Mutex::Impl::~Impl()
        {
            DeleteCriticalSection(&cs_);
        }

        void cv::Mutex::Impl::lockImpl()
        {
            EnterCriticalSection(&cs_);
        }

        bool cv::Mutex::Impl::tryLockImpl()
        {
            return TryEnterCriticalSection(&cs_) != 0;
        }

        void cv::Mutex::Impl::unlockImpl()
        {
            LeaveCriticalSection(&cs_);
        }
    #endif
#elif defined(OPENCV_VXWORKS)
    #include <semLib.h>
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
        SEM_ID sem_;
    };

    cv::Mutex::Impl::Impl()
    {
        sem_ = semMCreate(SEM_Q_PRIORITY, SEM_FULL);
        if (sem_ == 0)
            throw std::runtime_error("cannot create mutex");
    }

    cv::Mutex::Impl::~mpl()
    {
        semDelete(sem_);
    }

    void cv::Mutex::Impl::lockImpl()
    {
        if (semTake(sem_, WAIT_FOREVER) != OK)
            throw std::runtime_error("cannot lock mutex");
    }

    bool cv::Mutex::Impl::tryLockImpl()
    {
        return semTake(sem_, NO_WAIT) == OK;
    }

    void cv::Mutex::Impl::unlockImpl()
    {
        if (semGive(sem_) != OK)
            throw std::runtime_error("cannot unlock mutex");
    }
#else
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

    cv::Mutex::Impl::Impl()
    {
    #if defined(OPENCV_VXWORKS)
        // This workaround is for VxWorks 5.x where
        // pthread_mutex_init() won't properly initialize the mutex
        // resulting in a subsequent freeze in pthread_mutex_destroy()
        // if the mutex has never been used.
        std::memset(&mutex_, 0, sizeof(mutex_));
    #endif
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
            throw std::runtime_error("cannot create mutex");
        }
        pthread_mutexattr_destroy(&attr);
    }

    cv::Mutex::Impl::~Impl()
    {
        pthread_mutex_destroy(&mutex_);
    }

    void cv::Mutex::Impl::lockImpl()
    {
        if (pthread_mutex_lock(&mutex_))
            throw std::runtime_error("cannot lock mutex");
    }

    bool cv::Mutex::Impl::tryLockImpl()
    {
        int rc = pthread_mutex_trylock(&mutex_);
        if (rc == 0)
            return true;
        else if (rc == EBUSY)
            return false;
        else
            throw std::runtime_error("cannot lock mutex");
    }

    void cv::Mutex::Impl::unlockImpl()
    {
        if (pthread_mutex_unlock(&mutex_))
            throw std::runtime_error("cannot unlock mutex");
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
