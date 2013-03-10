#include "utility.hpp"

//////////////////////////////////////////////////////////
// RefCountedObject

cv::RefCountedObject::RefCountedObject() : counter_(1)
{
}

cv::RefCountedObject::~RefCountedObject()
{
}

void cv::RefCountedObject::release() const
{
    if (--counter_ == 0) delete this;
}

void cv::RefCountedObject::duplicate() const
{
    ++counter_;
}

int cv::RefCountedObject::referenceCount() const
{
    return counter_.value();
}
