#include "utility.hpp"

#include <cstdlib>
#include <stdexcept>

bool cv::Environment::has(const std::string& name)
{
    return std::getenv(name.c_str()) != 0;
}

std::string cv::Environment::get(const std::string& name)
{
    const char* val = std::getenv(name.c_str());

    if (val)
        return std::string(val);

    throw std::runtime_error("Not found");
}

std::string cv::Environment::get(const std::string& name, const std::string& defaultValue)
{
    const char* val = std::getenv(name.c_str());

    if (val)
        return std::string(val);

    return defaultValue;
}
