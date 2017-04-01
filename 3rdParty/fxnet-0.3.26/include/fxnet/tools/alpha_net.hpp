/*
 * alpha_net.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef FXNET_ALPHA_NET_HPP_
#define FXNET_ALPHA_NET_HPP_

#include "fxnet/tools/fxnet_core.hpp"

namespace hbot {
namespace fxnet {


class AlphaNet {
 public:
  explicit AlphaNet(const std::string proto_file, const std::string model_file);
  ~AlphaNet();
  void ReshapeBottom(const std::vector<size_t>& shape, const int bottom_id = 0);
  void ReshapeBottom(const int num, const int height, const int width,
      const int channels,const int bottom_id=0);
  void ForwardPrefilled();
  const std::vector<int>& TotalOutputShiftNum() const;
  std::vector<Tensor<int8_t> >& input_tensors(
      std::vector<Tensor<int8_t> >* tensors ) const;
  std::vector<Tensor<int32_t> >& output_tensors(
      std::vector<Tensor<int32_t> >* tensors) const;

 protected:
  void* net_;
  std::vector<void*> input_tensor_handles_;
  DISABLE_COPY_AND_ASSIGN(AlphaNet);
};


}  // namespace fxnet
}  // namespace hbot



#endif /* FXNET_ALPHA_NET_HPP_ */
