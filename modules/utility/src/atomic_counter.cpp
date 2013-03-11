#include "utility.hpp"

#include <stdexcept>

#if OPENCV_OS == OPENCV_OS_WINDOWS_NT
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
#elif OPENCV_OS == OPENCV_OS_MAC_OS_X
    #include <libkern/OSAtomic.h>
#endif

#if OPENCV_OS == OPENCV_OS_WINDOWS_NT

cv::AtomicCounter::AtomicCounter() : counter_(0)
{
}

cv::AtomicCounter::AtomicCounter(cv::AtomicCounter::ValueType initialValue) : counter_(initialValue)
{
}

cv::AtomicCounter::AtomicCounter(const cv::AtomicCounter& other) : counter_(other.value())
{
}

cv::AtomicCounter::~AtomicCounter()
{
}

cv::AtomicCounter& cv::AtomicCounter::operator = (const cv::AtomicCounter& other)
{
    InterlockedExchange(&counter_, other.value());
    return *this;
}

cv::AtomicCounter& cv::AtomicCounter::operator = (cv::AtomicCounter::ValueType value)
{
    InterlockedExchange(&counter_, value);
    return *this;
}

cv::AtomicCounter::operator cv::AtomicCounter::ValueType () const
{
    return counter_;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::value() const
{
    return counter_;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator ++ () // prefix
{
    return InterlockedIncrement(&counter_);
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator ++ (int) // postfix
{
    ValueType result = InterlockedIncrement(&counter_);
    return --result;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator -- () // prefix
{
    return InterlockedDecrement(&counter_);
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator -- (int) // postfix
{
    ValueType result = InterlockedDecrement(&counter_);
    return ++result;
}

bool cv::AtomicCounter::operator ! () const
{
    return counter_ == 0;
}

#elif OPENCV_OS == OPENCV_OS_MAC_OS_X

cv::AtomicCounter::AtomicCounter() : counter_(0)
{
}

cv::AtomicCounter::AtomicCounter(cv::AtomicCounter::ValueType initialValue) : counter_(initialValue)
{
}

cv::AtomicCounter::AtomicCounter(const cv::AtomicCounter& other) : counter_(other.value())
{
}

cv::AtomicCounter::~AtomicCounter()
{
}

cv::AtomicCounter& cv::AtomicCounter::operator = (const cv::AtomicCounter& other)
{
    counter_ = other.value();
    return *this;
}

cv::AtomicCounter& cv::AtomicCounter::operator = (cv::AtomicCounter::ValueType value)
{
    counter_ = value;
    return *this;
}

cv::AtomicCounter::operator cv::AtomicCounter::ValueType () const
{
    return counter_;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::value() const
{
    return counter_;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator ++ () // prefix
{
    return OSAtomicIncrement32(&counter_);
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator ++ (int) // postfix
{
    ValueType result = OSAtomicIncrement32(&counter_);
    return --result;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator -- () // prefix
{
    return OSAtomicDecrement32(&counter_);
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator -- (int) // postfix
{
    ValueType result = OSAtomicDecrement32(&counter_);
    return ++result;
}

bool cv::AtomicCounter::operator ! () const
{
    return counter_ == 0;
}

#elif defined(OPENCV_HAVE_GCC_ATOMICS)

cv::AtomicCounter::AtomicCounter() : counter_(0)
{
}

cv::AtomicCounter::AtomicCounter(cv::AtomicCounter::ValueType initialValue) : counter_(initialValue)
{
}

cv::AtomicCounter::AtomicCounter(const cv::AtomicCounter& other) : counter_(other.value())
{
}

cv::AtomicCounter::~AtomicCounter()
{
}

cv::AtomicCounter& cv::AtomicCounter::operator = (const cv::AtomicCounter& other)
{
    __sync_lock_test_and_set(&counter_, other.value());
    return *this;
}

cv::AtomicCounter& cv::AtomicCounter::operator = (cv::AtomicCounter::ValueType value)
{
    __sync_lock_test_and_set(&counter_, value);
    return *this;
}

cv::AtomicCounter::operator cv::AtomicCounter::ValueType () const
{
    return counter_;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::value() const
{
    return counter_;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator ++ () // prefix
{
    return __sync_add_and_fetch(&counter_, 1);
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator ++ (int) // postfix
{
    return __sync_fetch_and_add(&counter_, 1);
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator -- () // prefix
{
    return __sync_sub_and_fetch(&counter_, 1);
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator -- (int) // postfix
{
    return __sync_fetch_and_sub(&counter_, 1);
}

bool cv::AtomicCounter::operator ! () const
{
    return counter_ == 0;
}

#else

cv::AtomicCounter::AtomicCounter()
{
    counter_.value = 0;
}

cv::AtomicCounter::AtomicCounter(cv::AtomicCounter::ValueType initialValue)
{
    counter_.value = initialValue;
}

cv::AtomicCounter::AtomicCounter(const cv::AtomicCounter& other)
{
    counter_.value = other.value();
}

cv::AtomicCounter::~AtomicCounter()
{
}

cv::AtomicCounter& cv::AtomicCounter::operator = (const cv::AtomicCounter& counter)
{
    Mutex::ScopedLock lock(counter_.mutex);
    counter_.value = counter.value();
    return *this;
}

cv::AtomicCounter& cv::AtomicCounter::operator = (cv::AtomicCounter::ValueType value)
{
    Mutex::ScopedLock lock(counter_.mutex);
    counter_.value = value;
    return *this;
}

cv::AtomicCounter::operator cv::AtomicCounter::ValueType () const
{
    ValueType result;
    {
        Mutex::ScopedLock lock(counter_.mutex);
        result = counter_.value;
    }
    return result;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::value() const
{
    ValueType result;
    {
        Mutex::ScopedLock lock(counter_.mutex);
        result = counter_.value;
    }
    return result;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator ++ () // prefix
{
    ValueType result;
    {
        Mutex::ScopedLock lock(counter_.mutex);
        result = ++counter_.value;
    }
    return result;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator ++ (int) // postfix
{
    ValueType result;
    {
        Mutex::ScopedLock lock(counter_.mutex);
        result = counter_.value++;
    }
    return result;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator -- () // prefix
{
    ValueType result;
    {
        Mutex::ScopedLock lock(counter_.mutex);
        result = --counter_.value;
    }
    return result;
}

cv::AtomicCounter::ValueType cv::AtomicCounter::operator -- (int) // postfix
{
    ValueType result;
    {
        Mutex::ScopedLock lock(counter_.mutex);
        result = counter_.value--;
    }
    return result;
}

bool cv::AtomicCounter::operator ! () const
{
    bool result;
    {
        Mutex::ScopedLock lock(counter_.mutex);
        result = counter_.value == 0;
    }
    return result;
}

#endif // OPENCV_OS
