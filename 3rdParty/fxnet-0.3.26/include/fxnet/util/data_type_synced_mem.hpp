/*
 * data_type_synced_mem.hpp
 *
 *  Created on: 2016年5月6日
 *      Author: Alan_Huang
 */

#ifndef DATA_TYPE_SYNCED_MEM_HPP_
#define DATA_TYPE_SYNCED_MEM_HPP_

#include "hobot_core/memory/syncedmem.hpp"
#include "hobot_core/math/math_functions.hpp"
#include "fxnet/blob_op.hpp"
#include "fxnet/blob_transform.hpp"

namespace hbot {
namespace fxnet{



//
#define REGISTER_TYPE_OF_DATATYPESYNCEDMEMORY_CPU_AND_SET_CPU_DATA(type)   \
	inline const type * cpu_##type##_data(){ 												\
		this->to_##type##_data();             				 	\
		return (const type *) (this->type##_data_->cpu_data());											\
	}																															\
	inline type * mutable_cpu_##type##_data(){	 	\
		this->to_##type##_data();     															\
		Dtype_synced_flag_ = (Dtype_synced_flag_ == UNINITIALIZED_TYPE)? \
													UNINITIALIZED_TYPE:UNSYNCED_TYPE ;				\
		int32_t_synced_flag_ = (int32_t_synced_flag_ == UNINITIALIZED_TYPE)? \
													UNINITIALIZED_TYPE:UNSYNCED_TYPE ;				\
		int8_t_synced_flag_ = (int8_t_synced_flag_ == UNINITIALIZED_TYPE)? \
														UNINITIALIZED_TYPE:UNSYNCED_TYPE ;				\
		type##_synced_flag_ = SYNCED_TYPE;														\
		return static_cast< type *> (this->type##_data_->mutable_cpu_data());						  \
	}																															\
	DataTypeSyncedFlag type##_synced_flag(){ return type##_synced_flag_; }

template <typename Dtype>

// @Todo Synchronize DataType for gpu_data.
class DataTypeSyncedMemory{
public:
	DataTypeSyncedMemory()
		:Dtype_synced_flag_(UNINITIALIZED_TYPE),  int32_t_synced_flag_(UNINITIALIZED_TYPE),
		 int8_t_synced_flag_(UNINITIALIZED_TYPE),size_(0){
		Dtype_data_ = int32_t_data_ = int8_t_data_ = NULL;
		int32_t_lshift_num_ = int32_t_valid_bit_num_ = int8_t_valid_bit_num_ = int8_t_lshift_num_ = 0;
	}
	explicit DataTypeSyncedMemory(size_t size)
		:Dtype_synced_flag_(UNINITIALIZED_TYPE),  int32_t_synced_flag_(UNINITIALIZED_TYPE),
		 int8_t_synced_flag_(UNINITIALIZED_TYPE),size_(size){
		Dtype_data_ = int32_t_data_ = int8_t_data_ = NULL;
		int32_t_lshift_num_ = int32_t_valid_bit_num_ = int8_t_valid_bit_num_ = int8_t_lshift_num_ = 0;
	}

	~DataTypeSyncedMemory(){
		if( this->Dtype_data_ != NULL){
			delete this->Dtype_data_;
		}
		if(this->int32_t_data_ != NULL){
			delete this->int32_t_data_;
		}
		if(this->int8_t_data_ != NULL){
			delete this->int8_t_data_;
		}
	}

	inline SyncedMemory::SyncedHead head(){
		return Dtype_data_->head();
	}

	inline const Dtype* cpu_data(){
		return this->cpu_Dtype_data();
	}
	inline void set_cpu_data(void* data){
		Dtype_data_->set_cpu_data(data);
		int32_t_synced_flag_ = (int32_t_synced_flag_ == UNINITIALIZED_TYPE)?
													UNINITIALIZED_TYPE:UNSYNCED_TYPE ;
		int8_t_synced_flag_ = (int8_t_synced_flag_ == UNINITIALIZED_TYPE)?
														UNINITIALIZED_TYPE:UNSYNCED_TYPE ;
		Dtype_synced_flag_ = SYNCED_TYPE;
	}

	inline Dtype* mutable_cpu_data(){
		return this->mutable_cpu_Dtype_data();
	}

  const Dtype* gpu_data(){
  	return  (const Dtype *) Dtype_data_->gpu_data();
  }
  void set_gpu_data(void* data){
  	Dtype_data_->set_gpu_data(data);
  }
  Dtype* mutable_gpu_data(){
  	return (Dtype*) Dtype_data_->mutable_gpu_data();
  }


  enum DataTypeSyncedFlag{UNINITIALIZED_TYPE, UNSYNCED_TYPE,SYNCED_TYPE};

	REGISTER_TYPE_OF_DATATYPESYNCEDMEMORY_CPU_AND_SET_CPU_DATA(Dtype)
	REGISTER_TYPE_OF_DATATYPESYNCEDMEMORY_CPU_AND_SET_CPU_DATA(int32_t)
	REGISTER_TYPE_OF_DATATYPESYNCEDMEMORY_CPU_AND_SET_CPU_DATA(int8_t)

  inline size_t size() { return size_; }


protected:
	void to_Dtype_data();
	void to_int32_t_data( );
	void to_int8_t_data( );


	DataTypeSyncedFlag Dtype_synced_flag_;
	DataTypeSyncedFlag int32_t_synced_flag_;
	DataTypeSyncedFlag int8_t_synced_flag_;

	SyncedMemory * Dtype_data_;
	SyncedMemory * int32_t_data_;
	SyncedMemory * int8_t_data_;

	size_t size_;


	int int32_t_lshift_num_;
	int int32_t_valid_bit_num_;
	int int8_t_lshift_num_;
	int int8_t_valid_bit_num_;


	DISABLE_COPY_AND_ASSIGN(DataTypeSyncedMemory);

};




#define DataTypeSyncedMemoryShiftBackToDtype(type)	fxnet::op::LeftShiftParam<Dtype> shift_back_param( \
							this->type##_valid_bit_num_ * -1, -FLT_MAX, FLT_MAX, this->size_);   \
							op::PtrConst<type> src( (const type *)(this->type##_data_->cpu_data()));  \
							op::PtrMutable<Dtype> dst( (Dtype *)(this->Dtype_data_->mutable_cpu_data()) );  \
							fxnet::fxnet_transform<op::LeftShift>(shift_back_param, src,  dst)

template<typename Dtype>
void DataTypeSyncedMemory<Dtype>::to_Dtype_data( ){
	switch(this->Dtype_synced_flag_){
	case UNINITIALIZED_TYPE:
	{
		this->Dtype_data_ = new SyncedMemory(sizeof(Dtype) * size_);
		if(this->int32_t_synced_flag_ == SYNCED_TYPE){
			DataTypeSyncedMemoryShiftBackToDtype(int32_t);
		}else if(this->int8_t_synced_flag_ == SYNCED_TYPE){
			DataTypeSyncedMemoryShiftBackToDtype(int8_t);
		}
		else{
			fxnet::fxnet_memset(sizeof(Dtype)*size_, 0, this->Dtype_data_->mutable_cpu_data());
		}
		this->Dtype_synced_flag_ = SYNCED_TYPE;
		break;
	}
	case UNSYNCED_TYPE:
	{
		if(this->int32_t_synced_flag_ == SYNCED_TYPE){
			DataTypeSyncedMemoryShiftBackToDtype(int32_t);
		}else if(this->int8_t_synced_flag_ == SYNCED_TYPE){
			DataTypeSyncedMemoryShiftBackToDtype(int8_t);
		}
		else{
			DBG(std::cout<<"Error,All are not synced in DataTypeSyncedMemory"<<std::endl;)
			assert_force(0);
		}
		this->Dtype_synced_flag_ = SYNCED_TYPE;
		break;
	}
	case SYNCED_TYPE:
		break;
	}
}

#define DataTypeSyncedMemoryShiftBack(srcType,dstType,shift_n)	fxnet::op::LeftShiftParam<Dtype> shift_back_param( \
					shift_n, fxnet::n_bit_int_lower_bound(this->dstType##_valid_bit_num_),   \
					fxnet::n_bit_int_upper_bound(this->dstType##_valid_bit_num_), this->size_);   \
					op::PtrConst<srcType> src( (const srcType *)(this->srcType##_data_->cpu_data()));  \
					op::PtrMutable<dstType> dst( (dstType *)(this->dstType##_data_->mutable_cpu_data()) );  \
					fxnet::fxnet_transform<op::LeftShift>(shift_back_param, src, dst )

template<typename Dtype>
void DataTypeSyncedMemory<Dtype>::to_int32_t_data( ){
	switch(this->int32_t_synced_flag_){
	case UNINITIALIZED_TYPE:
	{
		this->int32_t_data_ = new SyncedMemory(sizeof(int32_t) * size_);
		if(this->Dtype_synced_flag_ == SYNCED_TYPE){
			DataTypeSyncedMemoryShiftBack(Dtype, int32_t,int32_t_lshift_num_);
		}else if(this->int8_t_synced_flag_ == SYNCED_TYPE){
			DataTypeSyncedMemoryShiftBack(int8_t, int32_t,  int32_t_lshift_num_ - int8_t_lshift_num_);
		}
		else{
			fxnet::fxnet_memset(sizeof(int32_t)*size_, 0, this->int32_t_data_->mutable_cpu_data());
		}
		this->int32_t_synced_flag_ = SYNCED_TYPE;
		break;
	}
	case UNSYNCED_TYPE:
	{
		if(this->Dtype_synced_flag_ == SYNCED_TYPE){
			DataTypeSyncedMemoryShiftBack(Dtype, int32_t,int32_t_lshift_num_);
		}else if(this->int8_t_synced_flag_ == SYNCED_TYPE){
			DataTypeSyncedMemoryShiftBack(int8_t, int32_t, int32_t_lshift_num_ - int8_t_lshift_num_);
		}
		else{
			DBG(std::cout<<"Error,All are not synced in DataTypeSyncedMemory"<<std::endl;)
			assert_force(0);
		}
		this->int32_t_synced_flag_ = SYNCED_TYPE;
		break;
	}
	case SYNCED_TYPE:
		break;
	}

}
//

template<typename Dtype>
void DataTypeSyncedMemory<Dtype>::to_int8_t_data(){
	switch(this->int8_t_synced_flag_){
		case UNINITIALIZED_TYPE:
		{
			this->int8_t_data_ = new SyncedMemory(sizeof(int8_t) * size_);
			if(this->Dtype_synced_flag_ == SYNCED_TYPE){
				DataTypeSyncedMemoryShiftBack(Dtype, int8_t,int8_t_lshift_num_);
			}else if(this->int32_t_synced_flag_ == SYNCED_TYPE){
				DataTypeSyncedMemoryShiftBack(int32_t, int8_t,  int8_t_lshift_num_ - int32_t_lshift_num_);
			}
			else{
				fxnet::fxnet_memset(sizeof(int8_t)*size_, 0, this->int8_t_data_->mutable_cpu_data());
			}
			this->int8_t_synced_flag_ = SYNCED_TYPE;
			break;
		}
		case UNSYNCED_TYPE:
		{
			if(this->Dtype_synced_flag_ == SYNCED_TYPE){
				DataTypeSyncedMemoryShiftBack(Dtype, int8_t,int8_t_lshift_num_);
			}else if(this->int32_t_synced_flag_ == SYNCED_TYPE){
				DataTypeSyncedMemoryShiftBack(int32_t, int8_t,  int8_t_lshift_num_ - int32_t_lshift_num_);
			}
			else{
				DBG(std::cout<<"Error,All are not synced in DataTypeSyncedMemory"<<std::endl;)
				assert_force(0);
			}
			this->int8_t_synced_flag_ = SYNCED_TYPE;
			break;
		}
		case SYNCED_TYPE:
			break;
	}
}



}  // namespace fxnet
}  // namespace hbot


#endif /* DATA_TYPE_SYNCED_MEM_HPP_ */
