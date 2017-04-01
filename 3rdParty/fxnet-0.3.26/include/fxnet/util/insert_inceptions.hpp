/*
 * insert_inceptions.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef _FXNET_UTIL_INSERT_INCEPTIONS_HPP_
#define _FXNET_UTIL_INSERT_INCEPTIONS_HPP_

#include <string>
#include "fxnet/proto/fxnet.pb.h"
using namespace std;


namespace hbot {
namespace fxnet {


void InsertInceptions(const NetParameter& param, NetParameter* param_split);

void InsertInception(const InceptionParameter& inception_param,
		const LayerParameter layer_param,NetParameter* param_split);

string InsertInceptionColumn(const LayerParameter& layer_param,
		const InceptionParameter& inception_param,
		const string& bottom_name, const string& layer_name,
		const InceptionColumnParameter& inception_column,
		NetParameter* param_split);

string InceptionSubLayerName(const string& layer_name, const string& column_name,
    const int blob_idx, string postfix = string(""));

string InceptionSubBlobName(const string& layer_name, const string& column_name,
	    const int blob_idx,string postfix = string(""));

string ConfigureInceptionConvLayer(const string& layer_name, const string& column_name,
    const int blob_idx,  const string& bottom_name,const vector<ParamSpec>& blob_params,
    LayerParameter* conv_layer_param, const ConvolutionParameter& conv_param);




}  // namespace fxnet
}  // namespace hbot
#endif  // _FXNET_UTIL_INSERT_INCEPTIONS_HPP_
