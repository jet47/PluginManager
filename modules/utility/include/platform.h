#pragma once

#ifndef __OPENCV_PLATFORM_H__
#define __OPENCV_PLATFORM_H__

//
// Platform Identification
//

#define OPENCV_OS_FREE_BSD      0x0001
#define OPENCV_OS_AIX           0x0002
#define OPENCV_OS_HPUX          0x0003
#define OPENCV_OS_TRU64         0x0004
#define OPENCV_OS_LINUX         0x0005
#define OPENCV_OS_MAC_OS_X      0x0006
#define OPENCV_OS_NET_BSD       0x0007
#define OPENCV_OS_OPEN_BSD      0x0008
#define OPENCV_OS_IRIX          0x0009
#define OPENCV_OS_SOLARIS       0x000a
#define OPENCV_OS_QNX           0x000b
#define OPENCV_OS_VXWORKS       0x000c
#define OPENCV_OS_CYGWIN        0x000d
#define OPENCV_OS_UNKNOWN_UNIX  0x00ff
#define OPENCV_OS_WINDOWS_NT    0x1001
#define OPENCV_OS_WINDOWS_CE    0x1011
#define OPENCV_OS_VMS           0x2001

#if defined(__FreeBSD__)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS_FAMILY_BSD 1
    #define OPENCV_OS OPENCV_OS_FREE_BSD
#elif defined(_AIX) || defined(__TOS_AIX__)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS OPENCV_OS_AIX
#elif defined(hpux) || defined(_hpux)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS OPENCV_OS_HPUX
#elif defined(__digital__) || defined(__osf__)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS OPENCV_OS_TRU64
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__TOS_LINUX__)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS OPENCV_OS_LINUX
#elif defined(__APPLE__) || defined(__TOS_MACOS__)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS_FAMILY_BSD 1
    #define OPENCV_OS OPENCV_OS_MAC_OS_X
#elif defined(__NetBSD__)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS_FAMILY_BSD 1
    #define OPENCV_OS OPENCV_OS_NET_BSD
#elif defined(__OpenBSD__)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS_FAMILY_BSD 1
    #define OPENCV_OS OPENCV_OS_OPEN_BSD
#elif defined(sgi) || defined(__sgi)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS OPENCV_OS_IRIX
#elif defined(sun) || defined(__sun)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS OPENCV_OS_SOLARIS
#elif defined(__QNX__)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS OPENCV_OS_QNX
#elif defined(unix) || defined(__unix) || defined(__unix__)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS OPENCV_OS_UNKNOWN_UNIX
#elif defined(_WIN32_WCE)
    #define OPENCV_OS_FAMILY_WINDOWS 1
    #define OPENCV_OS OPENCV_OS_WINDOWS_CE
#elif defined(_WIN32) || defined(_WIN64)
    #define OPENCV_OS_FAMILY_WINDOWS 1
    #define OPENCV_OS OPENCV_OS_WINDOWS_NT
#elif defined(__CYGWIN__)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS OPENCV_OS_CYGWIN
#elif defined(__VMS)
    #define OPENCV_OS_FAMILY_VMS 1
    #define OPENCV_OS OPENCV_OS_VMS
#elif defined(OPENCV_VXWORKS)
    #define OPENCV_OS_FAMILY_UNIX 1
    #define OPENCV_OS OPENCV_OS_VXWORKS
#endif

#if !defined(OPENCV_OS)
    #error "Unknown Platform."
#endif

//
// Hardware Architecture and Byte Order
//

#define OPENCV_ARCH_ALPHA   0x01
#define OPENCV_ARCH_IA32    0x02
#define OPENCV_ARCH_IA64    0x03
#define OPENCV_ARCH_MIPS    0x04
#define OPENCV_ARCH_HPPA    0x05
#define OPENCV_ARCH_PPC     0x06
#define OPENCV_ARCH_POWER   0x07
#define OPENCV_ARCH_SPARC   0x08
#define OPENCV_ARCH_AMD64   0x09
#define OPENCV_ARCH_ARM     0x0a
#define OPENCV_ARCH_M68K    0x0b
#define OPENCV_ARCH_S390    0x0c
#define OPENCV_ARCH_SH      0x0d
#define OPENCV_ARCH_NIOS2   0x0e

#if defined(__ALPHA) || defined(__alpha) || defined(__alpha__) || defined(_M_ALPHA)
    #define OPENCV_ARCH OPENCV_ARCH_ALPHA
    #define OPENCV_ARCH_LITTLE_ENDIAN 1
#elif defined(i386) || defined(__i386) || defined(__i386__) || defined(_M_IX86)
    #define OPENCV_ARCH OPENCV_ARCH_IA32
    #define OPENCV_ARCH_LITTLE_ENDIAN 1
#elif defined(_IA64) || defined(__IA64__) || defined(__ia64__) || defined(__ia64) || defined(_M_IA64)
    #define OPENCV_ARCH OPENCV_ARCH_IA64
    #if defined(hpux) || defined(_hpux)
        #define OPENCV_ARCH_BIG_ENDIAN 1
    #else
        #define OPENCV_ARCH_LITTLE_ENDIAN 1
    #endif
#elif defined(__x86_64__) || defined(_M_X64)
    #define OPENCV_ARCH OPENCV_ARCH_AMD64
    #define OPENCV_ARCH_LITTLE_ENDIAN 1
#elif defined(__mips__) || defined(__mips) || defined(__MIPS__) || defined(_M_MRX000)
    #define OPENCV_ARCH OPENCV_ARCH_MIPS
    #define OPENCV_ARCH_BIG_ENDIAN 1
#elif defined(__hppa) || defined(__hppa__)
    #define OPENCV_ARCH OPENCV_ARCH_HPPA
    #define OPENCV_ARCH_BIG_ENDIAN 1
#elif defined(__PPC) || defined(__POWERPC__) || defined(__powerpc) || defined(__PPC__) || \
      defined(__powerpc__) || defined(__ppc__) || defined(__ppc) || defined(_ARCH_PPC) || defined(_M_PPC)
    #define OPENCV_ARCH OPENCV_ARCH_PPC
    #define OPENCV_ARCH_BIG_ENDIAN 1
#elif defined(_POWER) || defined(_ARCH_PWR) || defined(_ARCH_PWR2) || defined(_ARCH_PWR3) || \
      defined(_ARCH_PWR4) || defined(__THW_RS6000)
    #define OPENCV_ARCH OPENCV_ARCH_POWER
    #define OPENCV_ARCH_BIG_ENDIAN 1
#elif defined(__sparc__) || defined(__sparc) || defined(sparc)
    #define OPENCV_ARCH OPENCV_ARCH_SPARC
    #define OPENCV_ARCH_BIG_ENDIAN 1
#elif defined(__arm__) || defined(__arm) || defined(ARM) || defined(_ARM_) || defined(__ARM__) || defined(_M_ARM)
    #define OPENCV_ARCH OPENCV_ARCH_ARM
    #if defined(__ARMEB__)
        #define OPENCV_ARCH_BIG_ENDIAN 1
    #else
        #define OPENCV_ARCH_LITTLE_ENDIAN 1
    #endif
#elif defined(__m68k__)
    #define OPENCV_ARCH OPENCV_ARCH_M68K
    #define OPENCV_ARCH_BIG_ENDIAN 1
#elif defined(__s390__)
    #define OPENCV_ARCH OPENCV_ARCH_S390
    #define OPENCV_ARCH_BIG_ENDIAN 1
#elif defined(__sh__) || defined(__sh) || defined(SHx) || defined(_SHX_)
    #define OPENCV_ARCH OPENCV_ARCH_SH
    #if defined(__LITTLE_ENDIAN__) || (OPENCV_OS == OPENCV_OS_WINDOWS_CE)
        #define OPENCV_ARCH_LITTLE_ENDIAN 1
    #else
        #define OPENCV_ARCH_BIG_ENDIAN 1
    #endif
#elif defined (nios2) || defined(__nios2) || defined(__nios2__)
    #define OPENCV_ARCH OPENCV_ARCH_NIOS2
    #if defined(__nios2_little_endian) || defined(nios2_little_endian) || defined(__nios2_little_endian__)
        #define OPENCV_ARCH_LITTLE_ENDIAN 1
    #else
        #define OPENCV_ARCH_BIG_ENDIAN 1
    #endif
#endif

#if !defined(OPENCV_ARCH)
    #error "Unknown Hardware Architecture."
#endif

#if ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 2) || __GNUC__ > 4) && (defined(__x86_64__) || defined(__i386__))
    #if !defined(OPENCV_HAVE_GCC_ATOMICS) && !defined(OPENCV_NO_GCC_ATOMICS)
        #define OPENCV_HAVE_GCC_ATOMICS
    #endif
#elif ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 3) || __GNUC__ > 4)
    #if !defined(OPENCV_HAVE_GCC_ATOMICS) && !defined(OPENCV_NO_GCC_ATOMICS)
        #define OPENCV_HAVE_GCC_ATOMICS
    #endif
#endif // OPENCV_OS

//
// Thread-safety of local static initialization
//

#if __cplusplus >= 201103L || __GNUC__ >= 4 || defined(__clang__)
    #ifndef OPENCV_LOCAL_STATIC_INIT_IS_THREADSAFE
        #define OPENCV_LOCAL_STATIC_INIT_IS_THREADSAFE 1
    #endif
#endif

#endif // __OPENCV_PLATFORM_H__
