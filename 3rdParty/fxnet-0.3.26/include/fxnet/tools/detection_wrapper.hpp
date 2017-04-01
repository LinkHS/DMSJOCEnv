/*
 *      Author: Alan_Huang
 */

#ifndef FXNET_DETECTION_WRAPPER_
#define FXNET_DETECTION_WRAPPER_

#include "fxnet/tools/io_process.hpp"
#include "fxnet/tools/fxnet_core.hpp"
#include <vector>
#include <iostream>

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#define UNUSED_VAR  __attribute__ ((unused))
#endif

namespace hbot {
namespace fxnet {

template <typename SRCOBJ, typename DSTOBJ >
class FXNetDetector{
public:
	explicit FXNetDetector(const std::string& fxnet_model_file){
		net_ = new FNet<float>(fxnet_model_file);
		input_tensors_ = net_->input_tensors();
		output_tensors_ = net_->output_tensors();
		input_max_batch_size_ = 1;
		score_threshold_ = 0.5;
		net_name_ = "FXNetDetector";
		use_cuda_ = false;
	}
	virtual ~FXNetDetector(){
		delete net_;
	}

	virtual void PredictResults(void* img_ptr, const int img_h, const int img_w,
			const int img_channel, std::vector<SRCOBJ>* vector_obj, std::vector<DSTOBJ>* vector_result) = 0;
	inline void SetNumThread(int num_thread){ if(num_thread > 0) net_->SetNumThreads(num_thread); }
	inline void SetInt8ForwardFlag(bool flag, const int layer_num = -1){ net_->SetInt8ForwardFlag(flag,layer_num);}
	virtual inline void SetInputMaxBatchSize(int batch_size){ input_max_batch_size_ = batch_size;}
	inline void SetScoreThreshold(float threshold){score_threshold_ = threshold;}
	inline int InputChannel() const{ return this->net_->input_tensors()[0].channels();}
	inline std::string ModelName() const {return this->net_->ModelName();}
	inline int ModelVersion() const {return this->net_->ModelVersion();}
	inline void SetUseCudaFlag(bool flag){use_cuda_ = flag;}

protected:
	std::vector<Tensor<float> > input_tensors_;
	std::vector<Tensor<float> > output_tensors_;
	int input_max_batch_size_;
	FNet<float>* net_;
	std::string net_name_;
	float score_threshold_;
	bool use_cuda_;
	DISABLE_COPY_AND_ASSIGN(FXNetDetector);
};




class FXNetImageParser : public FXNetDetector<Tensor<float>, Tensor<float> >{
public:
	explicit FXNetImageParser(const std::string& fxnet_model_file);
	virtual ~FXNetImageParser(){};

	virtual void PredictResults(void* img_ptr, const int img_h, const int img_w,
	    const int img_channel, std::vector<Tensor<float> >* vector_obj,
	    std::vector<Tensor<float> >* vector_result) override ;

	inline void SetScoreThreshold(){
	  std::cout<<"ScoreThreashold is unavailable in FXNetImageParser"<<std::endl;
	}
	virtual inline void SetInputMaxBatchSize(int batch_size) override{
		UNUSED(batch_size);
		input_max_batch_size_ = 1;
		std::cout<<"batch_size is always 1 in FXNetImageParser"<<std::endl;
	}
	inline std::vector<size_t> InputShape(){return input_tensors_[0].shape();}

protected:
	float input_mean_,input_scale_factor_;
	DISABLE_COPY_AND_ASSIGN(FXNetImageParser);
};


template <typename SRCOBJ, typename DSTOBJ >
class PyramidObjDetector : public FXNetDetector<SRCOBJ, DSTOBJ>{
 public:
  explicit PyramidObjDetector(const std::string& fxnet_model_file);
  virtual ~PyramidObjDetector();

  virtual void PredictResults(void* img_ptr,const int img_h, const int img_w,
      const int img_channel, std::vector<SRCOBJ>* vector_obj, std::vector<DSTOBJ>* vector_result) override;

 protected:
  float mean_bgr_[3];
  float input_mean_, input_scale_factor_;
  int heatmap_a_, heatmap_b_, max_downsample_ratio_;
  ImgPyramid<uint8_t>* img_pyramid;
  DISABLE_COPY_AND_ASSIGN(PyramidObjDetector);
};


/**
 * @param img_ptr  (void* )uchar* ,  original image, BGR channel order.
 * @param vector_obj   (void*) vector<BBox  >*   the roi bbox
 * @param vector_result   (void*) vector<BBox >*  the detection results
 */
template <typename SRCOBJ, typename DSTOBJ >
class ROIObjDetector : public FXNetDetector<SRCOBJ, DSTOBJ>{
public:

	explicit ROIObjDetector(const std::string& fxnet_model_file);
	virtual ~ROIObjDetector();

	virtual void PredictResults(void* img_ptr,const int img_h, const int img_w,
			const int img_channel, std::vector<SRCOBJ>* vector_obj, std::vector<DSTOBJ>* vector_result);

	inline void SetBBoxScaleFactor(int bbox_scale_height_factor, int bbox_scale_width_factor){
		bbox_scale_height_factor_ = bbox_scale_height_factor;
		bbox_scale_width_factor_ = bbox_scale_width_factor;
	}

protected:

	inline virtual int GetObjNum(std::vector<SRCOBJ> * vector_obj){
		return vector_obj->size();
	}

	virtual void PreProcessAndLoadSample(void * img_ptr, const int img_h, const int img_w,
			const int img_channel,std::vector<void*> param, int obj_id,
			int num_id_in_batch);

	virtual void PostProcess(std::vector<void*> param);

	float input_mean_,input_scale_factor_,bbox_scale_height_factor_,bbox_scale_width_factor_;
	int heatmap_a_, heatmap_b_;

	void * temp_uchar8_tensor_blob1_;
	void * temp_uchar8_tensor_blob2_;
	void * post_process_temp_vector_;

	DISABLE_COPY_AND_ASSIGN(ROIObjDetector);

};


/**
 * @brief Landmark detection. The param of PredictResults:
 * 				img_ptr  (void* )uchar* ,  original image, BGR channel order.
 * 				vector_obj   (void*) vector<BBox >*   the roi bbox
 * 				vector_result   (void*) vector<Landmarks>*  the detection results
 */
template <typename SRCOBJ, typename DSTOBJ >
class ROILandmarkDetector : public ROIObjDetector<SRCOBJ, DSTOBJ>{
public:
	explicit ROILandmarkDetector(const std::string& fxnet_model_file)
		: ROIObjDetector<SRCOBJ, DSTOBJ>(fxnet_model_file){
		this->net_name_ = "ROILandmarkDetector";
		this->score_threshold_ = 0.5;
	}

protected:

	virtual void PostProcess(std::vector<void*> param);
	DISABLE_COPY_AND_ASSIGN(ROILandmarkDetector);
};


/**
 * @brief Attribute detection. The param of PredictResults:
 * 				img_ptr  (void* )uchar* ,  original image, BGR channel order.
 * 				vector_obj   (void*) vector<Landmarks>*   the roi bbox
 * 				vector_result   (void*) vector<FaceAttributes>*  the detection results
 */
template <typename SRCOBJ, typename DSTOBJ >
class ROIAttributeDetector : public ROIObjDetector<SRCOBJ,DSTOBJ>{
public:
	explicit ROIAttributeDetector(const std::string& fxnet_model_file)
		: ROIObjDetector<SRCOBJ,DSTOBJ>(fxnet_model_file){
		this->net_name_ = "ROIAttributeDetector";
	}
	inline void SetBBoxScaleFactor(int bbox_scale_height_factor, int bbox_scale_width_factor){
		UNUSED(bbox_scale_height_factor);
		UNUSED(bbox_scale_width_factor);
		std::cout<<"SetBBoxScaleFactor is not supported in ROIAttributeDetector"<<std::endl;
	}
protected:
	virtual void PreProcessAndLoadSample(void * img_ptr, const int img_h, const int img_w,
			const int img_channel,std::vector<void*> param, int obj_id,
			int num_id_in_batch);

	virtual void PostProcess(std::vector<void*> param);

	float GetBBoxFromLandmarks(Landmarks& landmarks, BBox& bbox);

	DISABLE_COPY_AND_ASSIGN(ROIAttributeDetector);
};

}  // namespace fxnet
}  // namespace hbot


#endif /* FXNET_DETECTION_WRAPPER_ */
