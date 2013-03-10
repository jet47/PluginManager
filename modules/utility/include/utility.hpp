#pragma once

#ifndef __OPENCV_UTILITY_HPP__
#define __OPENCV_UTILITY_HPP__

#include <stdexcept>

#include <Poco/SharedPtr.h>
#include <Poco/AtomicCounter.h>

#include "opencv_export.h"

namespace cv
{
    class OPENCV_EXPORT Object
    {
    public:
        virtual ~Object();
    };

    template <typename T>
    class Ptr
    {
    public:
        Ptr() : impl_()
        {
        }

        Ptr(T* ptr) : impl_(ptr)
        {
        }

        template <typename U>
        Ptr(const Ptr<U>& ptr) : impl_(ptr.impl_)
        {
        }

        Ptr(const Ptr& ptr) : impl_(ptr.impl_)
        {
        }

        Ptr& operator = (T* ptr)
        {
            impl_.assign(ptr);
            return *this;
        }

        template <typename U>
        Ptr<U> cast() const
        {
            Ptr<U> other;
            other.impl_ = impl_.cast<U>();
            return other;
        }

        T* operator ->()
        {
            return impl_.operator ->();
        }

        const T* operator ->() const
        {
            return impl_.operator ->();
        }

        bool isNull() const
        {
            return impl_.isNull();
        }

    private:
        Poco::SharedPtr<T> impl_;

        template <typename U> friend class Ptr;
    };
}

#endif // __OPENCV_UTILITY_HPP__
