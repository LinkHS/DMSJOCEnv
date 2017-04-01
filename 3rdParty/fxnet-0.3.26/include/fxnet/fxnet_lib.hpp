#ifndef FXNET_FXNET_LIB_HPP_
#define FXNET_FXNET_LIB_HPP_

#include <set>
#include <string>
#include <vector>
#include "fxnet/blob.hpp"
#include "fxnet/common.hpp"
#include "fxnet/filler.hpp"
#include "fxnet/fpga_net.hpp"
#include "fxnet/layer.hpp"
#include "fxnet/layer_factory.hpp"
#include "fxnet/layers/vision_layers.hpp"
#include "fxnet/net.hpp"
#include "fxnet/proto/fxnet.pb.h"
#include "fxnet/util/io.hpp"
#include "fxnet/util/upgrade_proto.hpp"
#include "hobot_core/base/thread.hpp"

namespace hbot {
namespace fxnet {

class FNetEngine{
 public:
  ~FNetEngine() {
    if (inst_fnet_set_.size() != 0) {
      LOG(INFO)<<"In ~FNetEngine, inst_fnet_set_.size() != 0. "
          << " Memory leak detected.";
    }
    google::protobuf::ShutdownProtobufLibrary();
    DBG(LOG(INFO) << "Call ~FNetEngine";)
  }
  static FNetEngine* Get() {
    static FNetEngine instance;
    return &instance;
  }
  void AddFNetInstance(void* ptr) {
    lock.WriteLock();
    this->inst_fnet_set_.insert(ptr);
    lock.WriteUnlock();
  }
  void DeleteFNetInstance(void* ptr) {
    lock.WriteLock();
    bool success = (this->inst_fnet_set_.erase(ptr) == 1);
    if (!success) {
      LOG(INFO) << "delete fxnet instance "<< ptr << " failed!";
    }
    lock.WriteUnlock();
  }
 protected:
  FNetEngine() {}
  std::set<void*> inst_fnet_set_;
  RWLock lock;
};

// A wrap of Net for user level.
template<typename Dtype>
class FXNet {
 public:
  explicit FXNet(const string& fxnet_model, bool is_caffemodel = true) {
    NetParameter* proto = new NetParameter();
    NetParameter* param = new NetParameter();
    net_ = NULL;
    bool success = DecodeProtoFromFile(fxnet_model, proto, param);

    bool upgrade_success = UpgradeNetAsNeeded(proto);
    if (!upgrade_success) {
      std::cout << "Could not upgrade " << fxnet_model << std::endl;
      std::abort();
    }
    if (success) {
      net_ = new Net<Dtype>(*proto);
      FNetEngine* eingine = FNetEngine::Get();
      eingine->AddFNetInstance(net_);
      if (is_caffemodel)
        net_->CopyTrainedLayersFrom(*param, STORATE_ORDER_NCHW);
      else
        net_->CopyTrainedLayersFrom(*param, STORATE_ORDER_NHWC);
    } else {
      std::cout << "Cannot open model " << fxnet_model << std::endl;
    }
    delete proto;
    delete param;
  }

  ~FXNet() {
    if (net_) {
      FNetEngine* eingine = FNetEngine::Get();
      eingine->DeleteFNetInstance(net_);
      delete net_;
    }
  }

  inline void set_int8_t_forward(bool flag) {
    net_->set_int8_t_forward_flag(flag);
  }


  inline const vector<Blob<Dtype>*>& ForwardPrefilled(bool use_cuda = false) {
    return net_->ForwardPrefilled(use_cuda);
  }

/**
 *  The following methods seem to be useless.
 *    Dtype ForwardFromTo(int start, int end){
        return net_->Forward(start,end);
      }
      Dtype ForwardFrom(int start){
        return net_->Forward(start);
      }
      Dtype ForwardTo(int end){
        return net_->Forward(end);
      }
 */

  /// @brief Run forward using a set of bottom blobs, and return the result.
  inline const vector<Blob<Dtype>*>& Forward(
      const vector<Blob<Dtype>* > & bottom) {
    return net_->Forward(bottom);
  }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_->input_blobs();
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_->output_blobs();
  }
  inline void Reshape() {
    net_->Reshape();
  }

 private:
  Net<Dtype> * net_;

  DISABLE_COPY_AND_ASSIGN(FXNet);
};




}  // namespace fxnet
}  // namespace hbot
#endif  // FXNET_FXNET_LIB_HPP_
