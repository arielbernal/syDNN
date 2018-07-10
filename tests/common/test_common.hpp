#pragma once

#include <gtest/gtest.h>
#include <CL/cl.hpp>

extern cl::Context clContext;
extern cl::CommandQueue clQueue;

class TestEnvironment : public ::testing::Environment {
protected:
  virtual void SetUp() {
    std::string platform_name = "Intel";
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    auto platform_it = std::find_if(all_platforms.begin(), all_platforms.end(),
                        [platform_name](const cl::Platform& p) {return p.getInfo<CL_PLATFORM_NAME>().find(platform_name) != std::string::npos; });

    //if (platform_it == all_platforms.end())
      //throw std::runtime_error("TestEnvironment() : No platform found");

    //cl::Platform default_platform = *platform_it;
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    std::vector<cl::Device> all_gpu_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_gpu_devices);
    if(all_gpu_devices.size() == 0)
      throw std::runtime_error("TestEnvironment() : No gpu devices found");

    cl::Device default_device = all_gpu_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    clContext = cl::Context({default_device});
    clQueue = cl::CommandQueue(clContext, default_device);
  }

};
