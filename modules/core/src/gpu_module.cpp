#include "gpu_module.hpp"

#include <Poco/SingletonHolder.h>

namespace
{
    class GpuModule : public cv::PluginManagerBase
    {
    public:
        GpuModule();

    protected:
        cv::Ptr<cv::Object> createImpl(const std::string& interface, const cv::ParameterMap& params);

    private:
        cv::Ptr<cv::PluginManagerBase> impl_;
    };

    GpuModule::GpuModule()
    {
        impl_ = cv::thePluginManager().create<cv::PluginManagerBase>("gpu");
    }

    cv::Ptr<cv::Object> GpuModule::createImpl(const std::string& interface, const cv::ParameterMap& params)
    {
        return impl_->create<cv::Object>(interface, params);
    }
}

cv::PluginManagerBase& cv::theGpuModule()
{
    static Poco::SingletonHolder<GpuModule> holder;
    return *holder.get();
}
