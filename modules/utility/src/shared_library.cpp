#include "utility.hpp"

#if defined(hpux) || defined(_hpux)

#include <dl.h>

class cv::SharedLibrary::Impl
{
public:
    Impl();
    ~Impl();

    void loadImpl(const std::string& path);
    void unloadImpl();
    bool isLoadedImpl() const;

    void* findSymbolImpl(const std::string& name);
    const std::string& getPathImpl() const;

    static std::string suffixImpl();

private:
    shl_t handle_;
    std::string path_;
    static cv::Mutex mutex_;
};

cv::Mutex cv::SharedLibrary::Impl::mutex_;

cv::SharedLibrary::Impl::Impl()
{
    handle_ = 0;
}

cv::SharedLibrary::Impl::~Impl()
{
}

void cv::SharedLibrary::Impl::loadImpl(const std::string& path)
{
    Mutex::ScopedLock lock(mutex_);

    if (handle_)
        throw std::runtime_error("Library Already Loaded");

    handle_ = shl_load(path.c_str(), BIND_DEFERRED, 0);

    if (!handle_)
        throw std::runtime_error("Library Load");

    path_ = path;
}

void cv::SharedLibrary::Impl::unloadImpl()
{
    Mutex::ScopedLock lock(mutex_);

    if (handle_)
    {
        shl_unload(handle_);
        handle_ = 0;
        path_.clear();
    }
}

bool cv::SharedLibrary::Impl::isLoadedImpl() const
{
    return handle_ != 0;
}

void* cv::SharedLibrary::Impl::findSymbolImpl(const std::string& name)
{
    Mutex::ScopedLock lock(mutex_);

    void* result = 0;
    if (handle_ && shl_findsym(&handle_, name.c_str(), TYPE_UNDEFINED, &result) !=  -1)
        return result;
    else
        return 0;
}

const std::string& cv::SharedLibrary::Impl::getPathImpl() const
{
    return path_;
}

std::string cv::SharedLibrary::Impl::suffixImpl()
{
    return ".sl";
}

#elif defined(OPENCV_VXWORKS)

#include <cstring>
#include <moduleLib.h>
#include <loadLib.h>
#include <unldLib.h>
#include <ioLib.h>
#include <symLib.h>
#include <sysSymTbl.h>

class cv::SharedLibrary::Impl
{
public:
    Impl();
    ~Impl();

    void loadImpl(const std::string& path);
    void unloadImpl();
    bool isLoadedImpl() const;

    void* findSymbolImpl(const std::string& name);

    static std::string suffixImpl();

private:
    MODULE_ID moduleId_;
    static cv::Mutex mutex_;
};

struct SymLookup
{
    const char* name;
    int         group;
    void*       addr;
};

extern "C" bool lookupFunc(char* name, int val, SYM_TYPE type, int arg, UINT16 group)
{
    SymLookup* symLookup = reinterpret_cast<SymLookup*>(arg);

    if (group == symLookup->group && std::strcmp(name, symLookup->name) == 0)
    {
        symLookup->addr = reinterpret_cast<void*>(val);
        return TRUE;
    }

    return FALSE;
}

cv::Mutex cv::SharedLibrary::Impl::mutex_;

cv::SharedLibrary::Impl::Impl()
{
    moduleId_ = 0;
}

cv::SharedLibrary::Impl::~Impl()
{
}

void cv::SharedLibrary::Impl::loadImpl(const std::string& path)
{
    Mutex::ScopedLock lock(mutex_);

    if (moduleId_)
        throw std::runtime_error("Library Already Loaded");

    const int fd = open(const_cast<char*>(path.c_str()), O_RDONLY, 0);

    if (!fd)
        throw std::runtime_error("Library Load");

    moduleId_ = loadModule(fd, LOAD_GLOBAL_SYMBOLS);

    if (!moduleId_)
    {
        int err = errno;
        close(fd);
        throw std::runtime_error("Library Load");
    }
}

void cv::SharedLibrary::Impl::unloadImpl()
{
    Mutex::ScopedLock lock(mutex_);

    if (moduleId_)
    {
        unldByModuleId(moduleId_, 0);
        moduleId_ = 0;
    }
}

bool cv::SharedLibrary::Impl::isLoadedImpl() const
{
    return moduleId_ != 0;
}

void* cv::SharedLibrary::Impl::findSymbolImpl(const std::string& name)
{
    Mutex::ScopedLock lock(mutex_);

    MODULE_INFO mi;
    if (!moduleInfoGet(moduleId_, &mi))
        return 0;

    SymLookup symLookup;
    symLookup.name  = name.c_str();
    symLookup.group = mi.group;
    symLookup.addr  = 0;
    symEach(sysSymTbl, reinterpret_cast<FUNCPTR>(lookupFunc), reinterpret_cast<int>(&symLookup));

    return symLookup.addr;
}

std::string cv::SharedLibrary::Impl::suffixImpl()
{
    return ".out";
}

#elif defined(OPENCV_OS_FAMILY_UNIX)

#include <dlfcn.h>

class cv::SharedLibrary::Impl
{
public:
    Impl();
    ~Impl();

    void loadImpl(const std::string& path);
    void unloadImpl();
    bool isLoadedImpl() const;

    void* findSymbolImpl(const std::string& name);
    const std::string& getPathImpl() const;

    static std::string suffixImpl();

private:
    void* handle_;
    static cv::Mutex mutex_;
};

cv::Mutex cv::SharedLibrary::Impl::mutex_;

cv::SharedLibrary::Impl::Impl()
{
    handle_ = 0;
}

cv::SharedLibrary::Impl::~Impl()
{
}

void cv::SharedLibrary::Impl::loadImpl(const std::string& path)
{
    Mutex::ScopedLock lock(mutex_);

    if (handle_)
        throw std::runtime_error("Library Already Loaded");

    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);

    if (!handle_)
        throw std::runtime_error("Library Load");
}

void cv::SharedLibrary::Impl::unloadImpl()
{
    Mutex::ScopedLock lock(mutex_);

    if (handle_)
    {
        dlclose(handle_);
        handle_ = 0;
    }
}

bool cv::SharedLibrary::Impl::isLoadedImpl() const
{
    return handle_ != 0;
}

void* cv::SharedLibrary::Impl::findSymbolImpl(const std::string& name)
{
    Mutex::ScopedLock lock(mutex_);

    void* result = 0;

    if (handle_)
        result = dlsym(handle_, name.c_str());

    return result;
}

std::string cv::SharedLibrary::Impl::suffixImpl()
{
#if defined(__APPLE__)
    return ".dylib";
#elif defined(hpux) || defined(_hpux)
    return ".sl";
#elif defined(__CYGWIN__)
    return ".dll";
#else
    return ".so";
#endif
}

#elif defined(OPENCV_OS_FAMILY_WINDOWS)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

class cv::SharedLibrary::Impl
{
public:
    Impl();
    ~Impl();

    void loadImpl(const std::string& path);
    void unloadImpl();
    bool isLoadedImpl() const;

    void* findSymbolImpl(const std::string& name);
    const std::string& getPathImpl() const;

    static std::string suffixImpl();

private:
    HMODULE handle_;
    static cv::Mutex mutex_;
};

cv::Mutex cv::SharedLibrary::Impl::mutex_;

cv::SharedLibrary::Impl::Impl()
{
    handle_ = 0;
}

cv::SharedLibrary::Impl::~Impl()
{
}

void cv::SharedLibrary::Impl::loadImpl(const std::string& path)
{
    Mutex::ScopedLock lock(mutex_);

    if (handle_)
        throw std::runtime_error("Library Already Loaded");

    handle_ = LoadLibraryExA(path.c_str(), 0, 0);

    if (!handle_)
        throw std::runtime_error("Library Load");
}

void cv::SharedLibrary::Impl::unloadImpl()
{
    Mutex::ScopedLock lock(mutex_);

    if (handle_)
    {
        FreeLibrary(handle_);
        handle_ = 0;
    }
}

bool cv::SharedLibrary::Impl::isLoadedImpl() const
{
    return handle_ != 0;
}

void* cv::SharedLibrary::Impl::findSymbolImpl(const std::string& name)
{
    Mutex::ScopedLock lock(mutex_);

    void* result = 0;

    if (handle_)
        result = (void*) GetProcAddress(handle_, name.c_str());

    return result;
}

std::string cv::SharedLibrary::Impl::suffixImpl()
{
    return ".dll";
}

#endif

cv::SharedLibrary::SharedLibrary()
{
    impl_ = new Impl;
}

cv::SharedLibrary::SharedLibrary(const std::string& path)
{
    impl_ = new Impl;
    impl_->loadImpl(path);
}

cv::SharedLibrary::~SharedLibrary()
{
    delete impl_;
}

void cv::SharedLibrary::load(const std::string& path)
{
    impl_->loadImpl(path);
}

void cv::SharedLibrary::unload()
{
    impl_->unloadImpl();
}

bool cv::SharedLibrary::isLoaded() const
{
    return impl_->isLoadedImpl();
}

bool cv::SharedLibrary::hasSymbol(const std::string& name)
{
    return impl_->findSymbolImpl(name) != 0;
}

void* cv::SharedLibrary::getSymbol(const std::string& name)
{
    void* result = impl_->findSymbolImpl(name);

    if (result)
        return result;
    else
        throw std::runtime_error("NotFoundException");
}

std::string cv::SharedLibrary::suffix()
{
    return Impl::suffixImpl();
}
