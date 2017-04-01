// fxnet.hpp is the header file that you need to include in your code. It wraps
// all the internal fxnet header files into one for simpler inclusion.

#ifndef FXNET_FXNET_CORE_HPP_
#define FXNET_FXNET_CORE_HPP_

#include <string>
#include <vector>
#include <sstream>
#include <stdint.h>
#include <set>

namespace hbot {
namespace fxnet {

#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)


template<typename Dtype>
class Tensor{
public:
	Tensor():shape_(NULL),tensor_handle_(NULL){}
	~Tensor(){};
	explicit Tensor(const void* tensor_handle_);

	const Dtype* cpu_data() const;
	Dtype* mutable_cpu_data();
	int offset(const int n, const int h = 0, const int w = 0, const int c = 0) const;

	inline Dtype data_at(const int n, const int h, const int w, const int c) const {
		return cpu_data()[offset(n, h, w, c)];
	}
  std::string shape_string() const;
  inline const std::vector<size_t>& shape() const { return *shape_; }
  int count() const;
  inline int num() const { return shape_->at(0); }
  inline int height() const { return shape_->at(1); }
  inline int width() const { return shape_->at(2); }
  inline int channels() const { return shape_->at(3); }

private:
	const std::vector<size_t>* shape_;
	void* tensor_handle_;
};



template<typename Dtype>
class FNet{
public:
	explicit FNet(const std::string& fxnet_model_file, bool WEIGHT_ORDER_NCHW = true);

	~FNet();

	std::string ModelName();
	int ModelVersion();
	std::vector<Tensor<Dtype> > input_tensors() const;
	std::vector<Tensor<Dtype> > output_tensors() const;
	void SetInt8ForwardFlag(bool flag, const int layer_num = -1);
	void SetNumThreads(int num);
	void ReshapeBottom(const std::vector<size_t>& shape, const int bottom_id = 0);
	void ReshapeBottom(const int num, const int height, const int width,
			const int channels,const int bottom_id=0);


	bool ForwardPrefilled(bool use_cuda = false);

	void* GetNetProto(){return net_proto_;}

	/**
	 * For debug: get the bottom and top tensors of a specific layer, and
	 * forward one layer.
	 */
	std::vector<Tensor<Dtype> > bottom_tensors_of_layer(int layer_id) const;
	std::vector<Tensor<Dtype> > top_tensors_of_layer(int layer_id) const;
	int layer_num() const;
	void LayerForward(int layer_id,bool use_cuda = false);
	std::string LayerName(int layer_id) const;
	const char* LayerType(int layer_id) const;
	std::vector<Tensor<Dtype> > param_tensors_of_layer(int layer_id) const;

	inline const void* GetNetHandle() const {return net_;}
	inline void* GetNetHandle() {return net_;}

private:
	void* net_;
	void* net_proto_;

	DISABLE_COPY_AND_ASSIGN(FNet);
};


#ifdef ARM_TEST
std::string net_valid_test(  std::string model,
		std::string input_file, std::string output_ref);

#endif


}  // namesapce fxnet
}  // namespace hbot
#endif  // FXNET_FXNET_CORE_HPP_
