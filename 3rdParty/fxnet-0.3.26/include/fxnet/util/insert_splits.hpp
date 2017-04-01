#ifndef _FXNET_UTIL_INSERT_SPLITS_HPP_
#define _FXNET_UTIL_INSERT_SPLITS_HPP_

#include <string>

#include "fxnet/proto/fxnet.pb.h"

namespace hbot {
namespace fxnet {

// Copy NetParameters with SplitLayers added to replace any shared bottom
// blobs with unique bottom blobs provided by the SplitLayer.
void InsertSplits(const NetParameter& param, NetParameter* param_split);

void ConfigureSplitLayer(const std::string& layer_name, const std::string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param);

std::string SplitLayerName(const std::string& layer_name, const std::string& blob_name,
    const int blob_idx);

std::string SplitBlobName(const std::string& layer_name, const std::string& blob_name,
    const int blob_idx, const int split_idx);

}  // namespace fxnet
}  // namespace hbot
#endif  // FXNET_UTIL_INSERT_SPLITS_HPP_
