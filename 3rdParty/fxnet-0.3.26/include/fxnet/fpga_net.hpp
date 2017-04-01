/*
 * fpga_net.hpp
 *
 *  Created on: 2016年4月7日
 *      Author: Alan_Huang
 */

#ifndef FXNET_FPGA_NET_HPP_
#define FXNET_FPGA_NET_HPP_

#include <map>
#include <string>
#include <vector>
#include "fxnet/fpga_layers/fpga_layer.hpp"
#include "fxnet/net.hpp"
#include "fxnet/tools/fxnet_core.hpp"

namespace hbot {
namespace fxnet {

template <typename Dtype>
class FPGANet{
 public:
  explicit FPGANet(const Net<Dtype>& net) {
    Init(net);
  }
  explicit FPGANet(const FNet<Dtype>& net) {
    const void* handle = net.GetNetHandle();
    Init(*(static_cast<const Net<Dtype>*>(handle)));
  }
  ~FPGANet();

  void SaveToNet(const Net<Dtype>& net);

  void Init(const Net<Dtype>& net);
  const vector<Blob<int32_t>*>& ForwardPrefilled();
  void ForwardLayer(int layer_id);
  void ForwardFromTo(int start, int end);
  void Reshape();
  void InferShiftInfo();
  inline const std::vector<int>& TotalOutputShiftNum() {
    return output_shift_num_;
  }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_; }
  inline const vector<Blob<int32_t>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<vector<Blob<int32_t>*> >& bottom_vecs() const {
    return bottom_vecs_; }
  inline const vector<vector<Blob<int32_t>*> >& top_vecs() const {
    return top_vecs_; }
  inline FPGALayer<Dtype> * layer_ptr(int id) { return fpga_layers_[id]; }
  inline int layer_size(){ return fpga_layers_.size();}
  inline vector<Blob<int32_t>*>  layer_bottom_vecs(int id){
    return bottom_vecs_[id];
  };
  inline vector<Blob<int32_t>*>  layer_top_vecs(int id){
    return top_vecs_[id];
  };
  inline string layer_name(int id){ return layer_names_[id]; }


  inline const map<string, int>& layer_names_index() {
    return layer_names_index_;
  }
  inline vector< Blob<int32_t> * >& blobs() {
    return blobs_;
  }
  inline const vector<string>& blob_names() {
    return blob_names_;
  }

  inline const map<string, int> & blob_names_index() {
    return  blob_names_index_;
  }

 protected:

  string name_;
  // All layer that actually executed in FPGA(chip)
  vector< FPGALayer<Dtype> * > fpga_layers_;

  vector<string> layer_names_;

  map<string, int> layer_names_index_;

  // The param of each layer
  vector<vector<Blob<int32_t>*> > params_;

  // The actually left shift number of each output data blob.
  std::vector<int>  output_shift_num_;

  /*
   * @brief The input data blobs of network. It should be float data blob, and
   *        the network will
   */
  vector<Blob<Dtype>* > net_input_blobs_;

  vector<int> net_output_blob_indices_;
  vector<Blob<int32_t>*> net_output_blobs_;

  //  @brief the blobs(except inputs) storing intermediate results between
  //   the layers.
  vector< Blob<int32_t> * > blobs_;
  vector<string> blob_names_;
  map<string, int> blob_names_index_;

  /// bottom_vecs stores the vectors containing the input for each
  /// layer(except inputs). They don't actually host the blobs (blobs_ does),
  /// so we simply store  pointers.
  vector<vector<Blob<int32_t>*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;

  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<int32_t>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;

  DISABLE_COPY_AND_ASSIGN(FPGANet);
};

}  // namespace fxnet
}  // namespace hbot

#endif /* FXNET_FPGA_NET_HPP_ */
