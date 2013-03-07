#include "plugin_cache.hpp"

int main(int argc, const char* argv[])
{
    cv::PluginCache cache;
    cache.load();
    return 0;
}
