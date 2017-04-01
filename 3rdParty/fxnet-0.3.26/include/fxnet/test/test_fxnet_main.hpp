// The main fxnet test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef FXNET_TEST_TEST_FXNET_MAIN_HPP_
#define FXNET_TEST_TEST_FXNET_MAIN_HPP_

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "fxnet/common.hpp"

using std::cout;
using std::endl;

#ifdef CMAKE_BUILD
  #include "fxnet_config.h"
#else
  #define CUDA_TEST_DEVICE -1
  #define CMAKE_SOURCE_DIR "src/"
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define CMAKE_EXT ""
#endif

int main(int argc, char** argv);

// @TODO MultiDevice test? 8bit, 16bit?
namespace hbot {

template <typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MultiDeviceTest() {
  	HbotEngine::set_mode(TypeParam::device);
  }
  virtual ~MultiDeviceTest() {}
};

typedef ::testing::Types<float, double> TestDtypes;

template <typename TypeParam>
struct CPUDevice {
  typedef TypeParam Dtype;
  static const HbotEngine::Brew device = HbotEngine::CPU;
};

template <typename Dtype>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype> > {
};

#ifdef CPU_ONLY
typedef ::testing::Types<CPUDevice<float>,
                         CPUDevice<double> > TestDtypesAndDevices;
#else

template <typename TypeParam>
struct GPUDevice {
  typedef TypeParam Dtype;
  static const HbotEngine::Brew device = HbotEngine::GPU;
};

template <typename Dtype>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype> > {
};

typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
                         GPUDevice<float>, GPUDevice<double> >
                         TestDtypesAndDevices;


#endif
}  // namespace hbot

#endif  // FXNET_TEST_TEST_FXNET_MAIN_HPP_
