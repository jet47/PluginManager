#pragma once

#ifndef __OPENCV_UTILITY_HPP__
#define __OPENCV_UTILITY_HPP__

#include <stdexcept>
#include <string>
#include <vector>

#include "platform.h"

#if defined(PLUGIN_MANAGER_LIB) && (OPENCV_OS == OPENCV_OS_WINDOWS_NT)
    #define OPENCV_EXPORT __declspec(dllexport)
#else
    #define OPENCV_EXPORT
#endif

namespace cv
{
    template <class M>
    class ScopedLock
    {
    public:
        explicit ScopedLock(M& mutex) : mutex_(mutex)
        {
            mutex_.lock();
        }

        ~ScopedLock()
        {
            mutex_.unlock();
        }

    private:
        M& mutex_;

        ScopedLock();
        ScopedLock(const ScopedLock&);
        ScopedLock& operator = (const ScopedLock&);
    };

    class OPENCV_EXPORT Mutex
    {
    public:
        typedef cv::ScopedLock<Mutex> ScopedLock;

        Mutex();
        ~Mutex();

        void lock();
        bool tryLock();
        void unlock();

    private:
        Mutex(const Mutex&);
        Mutex& operator = (const Mutex&);

        class Impl;
        Impl* impl_;
    };

    class OPENCV_EXPORT AtomicCounter
    {
    public:
        typedef int ValueType; /// The underlying integer type.

        AtomicCounter();
        explicit AtomicCounter(ValueType initialValue);
        AtomicCounter(const AtomicCounter& counter);
        ~AtomicCounter();

        AtomicCounter& operator = (const AtomicCounter& counter);
        AtomicCounter& operator = (ValueType value);

        operator ValueType () const;
        ValueType value() const;

        ValueType operator ++ ();    // prefix
        ValueType operator ++ (int); // postfix

        ValueType operator -- ();    // prefix
        ValueType operator -- (int); // postfix
        bool operator ! () const;

    private:
    #if OPENCV_OS == OPENCV_OS_WINDOWS_NT
        typedef volatile long ImplType;
    #elif OPENCV_OS == OPENCV_OS_MAC_OS_X
        typedef int32_t ImplType;
    #elif defined(OPENCV_HAVE_GCC_ATOMICS)
        typedef int ImplType;
    #else // generic implementation based on FastMutex
        struct ImplType
        {
            mutable Mutex mutex;
            volatile int  value;
        };
    #endif // OPENCV_OS

        ImplType counter_;
    };

    class OPENCV_EXPORT RefCountedObject
    {
    public:
        RefCountedObject();

        virtual void release() const;

        void duplicate() const;

        int referenceCount() const;

    protected:
        virtual ~RefCountedObject();

    private:
        RefCountedObject(const RefCountedObject&);
        RefCountedObject& operator = (const RefCountedObject&);

        mutable AtomicCounter counter_;
    };

    template <class C>
    class AutoPtr
    {
    public:
        AutoPtr() : ptr_(0)
        {
        }

        AutoPtr(C* ptr) : ptr_(ptr)
        {
        }

        AutoPtr(C* ptr, bool shared) : ptr_(ptr)
        {
            if (shared && ptr_) ptr_->duplicate();
        }

        AutoPtr(const AutoPtr& ptr) : ptr_(ptr.ptr_)
        {
            if (ptr_) ptr_->duplicate();
        }

        template <class Other>
        AutoPtr(const AutoPtr<Other>& ptr) : ptr_(const_cast<Other*>(ptr.get()))
        {
            if (ptr_) ptr_->duplicate();
        }

        ~AutoPtr()
        {
            if (ptr_) ptr_->release();
        }

        AutoPtr& assign(C* ptr)
        {
            if (ptr_ != ptr)
            {
                if (ptr_) ptr_->release();
                ptr_ = ptr;
            }
            return *this;
        }

        AutoPtr& assign(C* ptr, bool shared)
        {
            if (ptr_ != ptr)
            {
                if (ptr_) ptr_->release();
                ptr_ = ptr;
                if (shared && ptr_) ptr_->duplicate();
            }
            return *this;
        }

        AutoPtr& assign(const AutoPtr& ptr)
        {
            if (&ptr != this)
            {
                if (ptr_) ptr_->release();
                ptr_ = ptr.ptr_;
                if (ptr_) ptr_->duplicate();
            }
            return *this;
        }

        template <class Other>
        AutoPtr& assign(const AutoPtr<Other>& ptr)
        {
            if (ptr.get() != ptr_)
            {
                if (ptr_) ptr_->release();
                ptr_ = const_cast<Other*>(ptr.get());
                if (ptr_) ptr_->duplicate();
            }
            return *this;
        }

        AutoPtr& operator = (C* ptr)
        {
            return assign(ptr);
        }

        AutoPtr& operator = (const AutoPtr& ptr)
        {
            return assign(ptr);
        }

        template <class Other>
        AutoPtr& operator = (const AutoPtr<Other>& ptr)
        {
            return assign<Other>(ptr);
        }

        void swap(AutoPtr& ptr)
        {
            std::swap(ptr_, ptr.ptr_);
        }

        template <class Other>
        AutoPtr<Other> cast() const
        {
            Other* pOther = dynamic_cast<Other*>(ptr_);
            return AutoPtr<Other>(pOther, true);
        }

        template <class Other>
        AutoPtr<Other> unsafeCast() const
        {
            Other* pOther = static_cast<Other*>(ptr_);
            return AutoPtr<Other>(pOther, true);
        }

        C* operator -> ()
        {
            if (ptr_)
                return ptr_;
            else
                throw std::runtime_error("Null Pointer");
        }

        const C* operator -> () const
        {
            if (ptr_)
                return ptr_;
            else
                throw std::runtime_error("Null Pointer");
        }

        C& operator * ()
        {
            if (ptr_)
                return *ptr_;
            else
                throw std::runtime_error("Null Pointer");
        }

        const C& operator * () const
        {
            if (ptr_)
                return *ptr_;
            else
                throw std::runtime_error("Null Pointer");
        }

        C* get()
        {
            return ptr_;
        }

        const C* get() const
        {
            return ptr_;
        }

        operator C* ()
        {
            return ptr_;
        }

        operator const C* () const
        {
            return ptr_;
        }

        bool operator ! () const
        {
            return ptr_ == 0;
        }

        bool isNull() const
        {
            return ptr_ == 0;
        }

        C* duplicate()
        {
            if (ptr_) ptr_->duplicate();
            return ptr_;
        }

        bool operator == (const AutoPtr& ptr) const
        {
            return ptr_ == ptr.ptr_;
        }

        bool operator == (const C* ptr) const
        {
            return ptr_ == ptr;
        }

        bool operator == (C* ptr) const
        {
            return ptr_ == ptr;
        }

        bool operator != (const AutoPtr& ptr) const
        {
            return ptr_ != ptr.ptr_;
        }

        bool operator != (const C* ptr) const
        {
            return ptr_ != ptr;
        }

        bool operator != (C* ptr) const
        {
            return ptr_ != ptr;
        }

        bool operator < (const AutoPtr& ptr) const
        {
            return ptr_ < ptr.ptr_;
        }

        bool operator < (const C* ptr) const
        {
            return ptr_ < ptr;
        }

        bool operator < (C* ptr) const
        {
            return ptr_ < ptr;
        }

        bool operator <= (const AutoPtr& ptr) const
        {
            return ptr_ <= ptr.ptr_;
        }

        bool operator <= (const C* ptr) const
        {
            return ptr_ <= ptr;
        }

        bool operator <= (C* ptr) const
        {
            return ptr_ <= ptr;
        }

        bool operator > (const AutoPtr& ptr) const
        {
            return ptr_ > ptr.ptr_;
        }

        bool operator > (const C* ptr) const
        {
            return ptr_ > ptr;
        }

        bool operator > (C* ptr) const
        {
            return ptr_ > ptr;
        }

        bool operator >= (const AutoPtr& ptr) const
        {
            return ptr_ >= ptr.ptr_;
        }

        bool operator >= (const C* ptr) const
        {
            return ptr_ >= ptr;
        }

        bool operator >= (C* ptr) const
        {
            return ptr_ >= ptr;
        }

    private:
        C* ptr_;
    };

    template <class C>
    inline void swap(AutoPtr<C>& p1, AutoPtr<C>& p2)
    {
        p1.swap(p2);
    }

    template <class S>
    class SingletonHolder
    {
    public:
        SingletonHolder() : pS_(0)
        {
        }

        ~SingletonHolder()
        {
            delete pS_;
        }

        S* get()
        {
            Mutex::ScopedLock lock(m_);
            if (!pS_) pS_ = new S;
            return pS_;
        }

    private:
        S* pS_;
        Mutex m_;
    };

    class OPENCV_EXPORT SharedLibrary
    {
    public:
        SharedLibrary();
        SharedLibrary(const std::string& path);
        ~SharedLibrary();

        void load(const std::string& path);
        void unload();
        bool isLoaded() const;

        bool hasSymbol(const std::string& name);
        void* getSymbol(const std::string& name);

        static std::string suffix();

    private:
        SharedLibrary(const SharedLibrary&);
        SharedLibrary& operator = (const SharedLibrary&);

        class Impl;
        Impl* impl_;
    };

    class OPENCV_EXPORT Path
    {
    public:
         static bool isDirectory(const std::string& path);
         static void glob(const std::string& pattern, std::vector<std::string>& result, bool recursive = false);
    };

    class OPENCV_EXPORT Environment
    {
    public:
        static bool has(const std::string& name);
        static std::string get(const std::string& name);
        static std::string get(const std::string& name, const std::string& defaultValue);
    };
}

#endif // __OPENCV_UTILITY_HPP__
