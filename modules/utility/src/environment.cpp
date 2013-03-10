#include "utility.hpp"

#include <cstdlib>
#include <cstdio>

#if defined(OPENCV_OS_FAMILY_VMS)

#include <stdlib.h>
#include <starlet.h>
#include <descrip.h>
#include <ssdef.h>
#include <syidef.h>
#include <iledef.h>
#include <lnmdef.h>
#include <ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <net/if.h>
#include <inet.h>
#include <netdb.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <unistd.h>

namespace
{
    class EnvironmentImpl
    {
    public:
        static std::string getImpl(const std::string& name);
        static bool hasImpl(const std::string& name);

    private:
        static Mutex mutex_;
    };

    FastMutex EnvironmentImpl::mutex_;

    std::string EnvironmentImpl::getImpl(const std::string& name)
    {
        Mutex::ScopedLock lock(mutex_);

        const char* val = getenv(name.c_str());

        if (val)
            return std::string(val);
        else
            throw std::runtime_error("NotFoundException");
    }

    bool EnvironmentImpl::hasImpl(const std::string& name)
    {
        Mutex::ScopedLock lock(mutex_);

        return getenv(name.c_str()) != 0;
    }
}

#elif defined(OPENCV_VXWORKS)

#include <VxWorks.h>
#include <envLib.h>
#include <hostLib.h>
#include <ifLib.h>
#include <sockLib.h>
#include <ioLib.h>
#include <version.h>
#include <cstring>
#include <unistd.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <netinet/if_ether.h>
#include <ifLib.h>
#include <unistd.h>

namespace
{
    class EnvironmentImpl
    {
    public:
        static std::string getImpl(const std::string& name);
        static bool hasImpl(const std::string& name);

    private:
        static Mutex mutex_;
    };

    FastMutex EnvironmentImpl::mutex_;

    std::string EnvironmentImpl::getImpl(const std::string& name)
    {
        Mutex::ScopedLock lock(mutex_);

        const char* val = getenv(name.c_str());

        if (val)
            return std::string(val);
        else
            throw std::runtime_error("NotFoundException");
    }

    bool EnvironmentImpl::hasImpl(const std::string& name)
    {
        Mutex::ScopedLock lock(mutex_);

        return getenv(name.c_str()) != 0;
    }
}

#elif defined(OPENCV_OS_FAMILY_UNIX)

#include <cstring>
#include <unistd.h>
#include <stdlib.h>
#include <sys/utsname.h>
#include <sys/param.h>
#include <cstring>
#if defined(OPENCV_OS_FAMILY_BSD)
    #include <sys/sysctl.h>
#elif OPENCV_OS == OPENCV_OS_HPUX
    #include <pthread.h>
#endif

namespace
{
    class EnvironmentImpl
    {
    public:
        static std::string getImpl(const std::string& name);
        static bool hasImpl(const std::string& name);

    private:
        static Mutex mutex_;
    };

    FastMutex EnvironmentImpl::mutex_;

    std::string EnvironmentImpl::getImpl(const std::string& name)
    {
        Mutex::ScopedLock lock(mutex_);

        const char* val = getenv(name.c_str());

        if (val)
            return std::string(val);
        else
            throw std::runtime_error("NotFoundException");
    }

    bool EnvironmentImpl::hasImpl(const std::string& name)
    {
        Mutex::ScopedLock lock(mutex_);

        return getenv(name.c_str()) != 0;
    }
}

#elif defined(OPENCV_OS_FAMILY_WINDOWS)

#include <sstream>
#include <cstring>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <iphlpapi.h>

namespace
{
    class EnvironmentImpl
    {
    public:
        static std::string getImpl(const std::string& name);
        static bool hasImpl(const std::string& name);
    };

    std::string EnvironmentImpl::getImpl(const std::string& name)
    {
        DWORD len = GetEnvironmentVariableA(name.c_str(), 0, 0);

        if (len == 0)
            throw std::runtime_error("NotFoundException");

        char* buffer = new char[len];
        GetEnvironmentVariableA(name.c_str(), buffer, len);

        std::string result(buffer);
        delete [] buffer;

        return result;
    }

    bool EnvironmentImpl::hasImpl(const std::string& name)
    {
        DWORD len = GetEnvironmentVariableA(name.c_str(), 0, 0);
        return len > 0;
    }
}

#endif

bool cv::Environment::has(const std::string& name)
{
    return EnvironmentImpl::hasImpl(name);
}

std::string cv::Environment::get(const std::string& name)
{
    return EnvironmentImpl::getImpl(name);
}

std::string cv::Environment::get(const std::string& name, const std::string& defaultValue)
{
    if (has(name))
        return get(name);
    else
        return defaultValue;
}
