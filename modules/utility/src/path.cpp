#include "utility.hpp"

#include <algorithm>

#if defined WIN32 || defined _WIN32 || defined WINCE

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace
{
    const char dir_separators[] = "/\\";
    const char native_separator = '\\';

    struct dirent
    {
        const char* d_name;
    };

    struct DIR
    {
        WIN32_FIND_DATA data;
        HANDLE handle;
        dirent ent;
    };

    DIR* opendir(const char* path)
    {
        DIR* dir = new DIR;
        dir->ent.d_name = 0;
        dir->handle = ::FindFirstFileA((std::string(path) + "\\*").c_str(), &dir->data);
        if(dir->handle == INVALID_HANDLE_VALUE)
        {
            /*closedir will do all cleanup*/
            return 0;
        }
        return dir;
    }

    dirent* readdir(DIR* dir)
    {
        if (dir->ent.d_name != 0)
        {
            if (::FindNextFile(dir->handle, &dir->data) != TRUE)
                return 0;
        }
        dir->ent.d_name = dir->data.cFileName;
        return &dir->ent;
    }

    void closedir(DIR* dir)
    {
        ::FindClose(dir->handle);
        delete dir;
    }
}

#else

#include <dirent.h>
#include <sys/stat.h>

namespace
{
    const char dir_separators[] = "/";
    const char native_separator = '/';
}

#endif

namespace
{
    bool isDir(const std::string& path, DIR* dir)
    {
    #if defined WIN32 || defined _WIN32 || defined WINCE
        DWORD attributes;
        if (dir)
            attributes = dir->data.dwFileAttributes;
        else
            attributes = ::GetFileAttributes(path.c_str());

        return (attributes != INVALID_FILE_ATTRIBUTES) && ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0);
    #else
        struct stat stat_buf;
        stat( path.c_str(), &stat_buf);
        int is_dir = S_ISDIR( stat_buf.st_mode);
        (void)dir;

        return is_dir != 0;
    #endif
    }

    bool wildcmp(const char *string, const char *wild)
    {
        // Based on wildcmp written by Jack Handy - <A href="mailto:jakkhandy@hotmail.com">jakkhandy@hotmail.com</A>
        const char *cp = 0, *mp = 0;

        while ((*string) && (*wild != '*'))
        {
            if ((*wild != *string) && (*wild != '?'))
            {
                return false;
            }

            wild++;
            string++;
        }

        while (*string)
        {
            if (*wild == '*')
            {
                if (!*++wild)
                {
                    return true;
                }

                mp = wild;
                cp = string + 1;
            }
            else if ((*wild == *string) || (*wild == '?'))
            {
                wild++;
                string++;
            }
            else
            {
                wild = mp;
                string = cp++;
            }
        }

        while (*wild == '*')
        {
            wild++;
        }

        return *wild == 0;
    }

    void glob_rec(const std::string& directory, const std::string& wildchart, std::vector<std::string>& result, bool recursive)
    {
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir (directory.c_str())) != 0)
        {
            /* find all the files and directories within directory */
            try
            {
                while ((ent = readdir (dir)) != 0)
                {
                    const char* name = ent->d_name;
                    if((name[0] == 0) || (name[0] == '.' && name[1] == 0) || (name[0] == '.' && name[1] == '.' && name[2] == 0))
                        continue;

                    std::string path = directory + native_separator + name;

                    if (isDir(path, dir))
                    {
                        if (recursive)
                            glob_rec(path, wildchart, result, recursive);
                    }
                    else
                    {
                        if (wildchart.empty() || wildcmp(name, wildchart.c_str()))
                            result.push_back(path);
                    }
                }
            }
            catch (...)
            {
                closedir(dir);
                throw;
            }
            closedir(dir);
        }
        else
            throw std::runtime_error("could not open directory");
    }
}

bool cv::Path::isDirectory(const std::string& path)
{
    return isDir(path, 0);
}

void cv::Path::glob(const std::string& pattern, std::vector<std::string>& result, bool recursive)
{
    result.clear();
    std::string path, wildchart;

    if (isDir(pattern, 0))
    {
        if(strchr(dir_separators, pattern[pattern.size() - 1]) != 0)
        {
            path = pattern.substr(0, pattern.size() - 1);
        }
        else
        {
            path = pattern;
        }
    }
    else
    {
        size_t pos = pattern.find_last_of(dir_separators);
        if (pos == std::string::npos)
        {
            wildchart = pattern;
            path = ".";
        }
        else
        {
            path = pattern.substr(0, pos);
            wildchart = pattern.substr(pos + 1);
        }
    }

    glob_rec(path, wildchart, result, recursive);
    std::sort(result.begin(), result.end());
}
