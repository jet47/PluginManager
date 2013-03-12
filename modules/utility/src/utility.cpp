#include "utility.hpp"

//////////////////////////////////////////////////////////
// RefCountedObject

cv::RefCountedObject::RefCountedObject() : counter_(1)
{
}

void cv::RefCountedObject::duplicate()
{
    ++counter_;
}

void cv::RefCountedObject::release()
{
    if (--counter_ == 0)
        deleteImpl();
}

int cv::RefCountedObject::referenceCount() const
{
    return counter_.value();
}

cv::RefCountedObject::~RefCountedObject()
{
}

void cv::RefCountedObject::deleteImpl()
{
    delete this;
}
