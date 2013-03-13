#include "gpu_module.hpp"

namespace
{
    class GpuModule : public cv::PluginManagerBase
    {
    public:
        GpuModule();

        void init();

    protected:
        cv::AutoPtr<cv::RefCountedObject> createImpl(const std::string& interface, const cv::ParameterMap& params);

    private:
        cv::AutoPtr<cv::PluginManagerBase> impl_;
    };

    GpuModule::GpuModule()
    {
        impl_ = cv::thePluginManager()->create<cv::PluginManagerBase>("gpu.module");
    }

    void GpuModule::init()
    {
        impl_->init();
    }

    cv::AutoPtr<cv::RefCountedObject> GpuModule::createImpl(const std::string& interface, const cv::ParameterMap& params)
    {
        return impl_->create<cv::RefCountedObject>(interface, params);
    }
}

cv::PluginManagerBase* cv::theGpuModule()
{
    static cv::SingletonHolder<GpuModule> holder;
    return holder.get();
}
