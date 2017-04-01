#ifndef FXNET_COMMON_HPP_
#define FXNET_COMMON_HPP_


#include <assert.h>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "fxnet/util/device_alternate.hpp"
#include "hobot_core/base/base_common.hpp"


// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

/**
 * @TODO support for 8bit, 16bit, 24bit
 */
// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

#define INSTANTIATE_STRUCT(classname) \
  char gInstantiationGuard##classname; \
  template struct classname<float>; \
  template struct classname<double>


#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu( \
      const std::vector<Blob<float>*>& bottom, \
      const std::vector<Blob<float>*>& top); \
  template void classname<double>::Forward_gpu( \
      const std::vector<Blob<double>*>& bottom, \
      const std::vector<Blob<double>*>& top);

namespace hbot {
namespace fxnet {


/**
 * @TODO Further extension for multi-device
 */

// Common functions and classes from std that fxnet often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;



}  // namespace fxnet
}  // namespace hbot
#endif  // FXNET_COMMON_HPP_
