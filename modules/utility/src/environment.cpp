#include "utility.hpp"

#include <cstdlib>
#include <sstream>
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

    std::ostringstream msg;
    msg << "Can't find enviroment variable - " << name;
    throw std::runtime_error(msg.str());
}

std::string cv::Environment::get(const std::string& name, const std::string& defaultValue)
{
    const char* val = std::getenv(name.c_str());

    if (val)
        return std::string(val);

    return defaultValue;
}
