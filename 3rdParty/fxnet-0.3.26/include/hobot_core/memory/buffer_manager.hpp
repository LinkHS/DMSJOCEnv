/*
 * buffer_manager.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef HBOT_MEMORY__MEMORY_BUFFER_MANAGER_HPP_
#define HBOT_MEMORY__MEMORY_BUFFER_MANAGER_HPP_

#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace hbot {

template<typename Dtype>
class BufferManager {
 public:
  BufferManager() {};
  virtual ~BufferManager() {};
  virtual Dtype* Alloc(const size_t size) = 0;
  virtual void Free(const Dtype* buf_const) = 0;
};


template<typename Dtype>
class SimpleBuffer : public BufferManager<Dtype> {
 public:
  SimpleBuffer() : count(0) {}
  virtual ~SimpleBuffer() {
    CHECK_EQ(in_use_buf_.size(), 0);
    CHECK_EQ(count, available_buf_.size());
    for (auto& ptr : available_buf_) {
      LOG(INFO) <<"free entry of size " << ptr->size() << " in SimpleBuffer";
      delete ptr;
    }
  }
  virtual Dtype* Alloc(const size_t size) override {
    std::unique_lock<std::mutex> lck (mtx);
    if (available_buf_.empty()) {
      count++;
      available_buf_.push_back(new std::vector<Dtype>());
    }
    std::vector<Dtype>* front = available_buf_.front();
    available_buf_.pop_front();
    front->resize(size);
    Dtype* ret = &(*front)[0];
    CHECK_EQ(in_use_buf_.count(ret), 0);
    in_use_buf_[ret] = front;
    return ret;
  }
  virtual void Free(const Dtype* buf_const) override {
    std::unique_lock<std::mutex> lck (mtx);
    Dtype* buf = const_cast<Dtype*>(buf_const);
    CHECK_EQ(in_use_buf_.count(buf), 1);
    available_buf_.push_back(in_use_buf_[buf]);
    CHECK_EQ(1,in_use_buf_.erase(buf));
  }
 protected:
  std::mutex mtx;
  int count;
  std::list<std::vector<Dtype>*> available_buf_;
  std::unordered_map<Dtype*, std::vector<Dtype>*> in_use_buf_;
};



}  // namespace hbot



#endif /* HBOT_MEMORY__MEMORY_BUFFER_MANAGER_HPP_ */
