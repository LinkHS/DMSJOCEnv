#ifndef FXNET_NET_HPP_
#define FXNET_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "fxnet/blob.hpp"
#include "fxnet/common.hpp"
#include "fxnet/layer.hpp"
#include "fxnet/proto/fxnet.pb.h"
namespace hbot {
namespace fxnet {
template <typename Dtype>
class FPGANet;

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
  friend class FPGANet<Dtype>;
#ifdef PROTOBUF_FULL
  explicit Net(const string& param_file);
#endif
  explicit Net(const NetParameter& param);
  virtual ~Net();

  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward with the input Blob%s already fed separately.
   *
   * You can get the input blobs using input_blobs().
   */
  const vector<Blob<Dtype>*>& ForwardPrefilled();
  const vector<Blob<Dtype>*>& ForwardPrefilled(bool use_cuda);
  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  void ForwardFromTo(int start, int end, bool use_cuda);
  void ForwardFromTo(int start, int end);
  void ForwardFrom(int start);
  void ForwardTo(int end);
  /// @brief Run forward using a set of bottom blobs, and return the result.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom);
  /**
   * @brief Run forward using a serialized BlobProtoVector and return the
   *        result as a serialized BlobProtoVector
   */
  string Forward(const string& input_blob_protos);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net. Be careful, the default data arrange type is channel-major,
   *        while the original Caffe use row-major blobs.
   */
  void CopyTrainedLayersFrom(const NetParameter& param,
      DataStorageOrder src_data_arrange_type = STORATE_ORDER_NHWC);

#ifdef PROTOBUF_FULL
  void CopyTrainedLayersFrom(const string trained_filename,
      DataStorageOrder src_data_arrange_type = STORATE_ORDER_NHWC);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename,
      DataStorageOrder src_data_arrange_type = STORATE_ORDER_NHWC);

  bool LoadParamFrom(const string model_filename);

#endif

  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param) const;

  void SaveParam(const string out_name) const;

  /// @brief returns the network name.
  inline const string& name() const { return name_; }
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
  inline const vector< Blob<Dtype> * >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers
  inline const vector< Layer<Dtype> * >& layers() const {
    return layers_;
  }

  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }


  /// @brief returns the parameters
  inline const vector< Blob<Dtype> * >& params() const {
    return params_;
  }

  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  inline const vector<int>& param_owners() const { return param_owners_; }
  /// @brief Input and output blob numbers
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const Blob<Dtype>* blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const  Layer<Dtype> * layer_by_name(const string& layer_name) const;


//  // Helpers for Init.
//  /**
//   * @brief Remove layers that the user specified should be excluded
//   *        given the current phase, level, and stage.
//   */
//  static void FilterNet(const NetParameter& param,
//      NetParameter* param_filtered);


  void set_int8_t_forward_flag(bool flag, const int layer_num = -1);

 protected:
  // Helpers for Init.
  /// @brief Append a new input or top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward about input Blobs.
  void InputDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);


  /// @brief The network name
  string name_;

  /// @brief Individual layers in the net
  vector< Layer<Dtype> * > layers_;
  vector<string> layer_names_;
  map<string, int> layer_names_index_;

  /// @brief the blobs storing intermediate results between the layer.
  vector< Blob<Dtype> * > blobs_;
  vector<string> blob_names_;
  map<string, int> blob_names_index_;

  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob<Dtype>*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;

  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;

  vector<vector<int> > param_id_vecs_;
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int> > param_layer_indices_;
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  vector< Blob<Dtype> * > params_;



  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;



  DISABLE_COPY_AND_ASSIGN(Net);
};


}  // namespace fxnet
}  // namespace hbot
#endif  // FXNET_NET_HPP_
