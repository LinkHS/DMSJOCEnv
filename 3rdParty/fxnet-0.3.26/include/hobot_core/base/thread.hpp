/*
 * thread.hpp
 *
 *  Created on: 2016年4月27日
 *      Author: Alan_Huang
 */

#ifndef HBOT_THREAD_HPP_
#define HBOT_THREAD_HPP_

#include "hobot_core/base/base_common.hpp"

namespace hbot {
// @brief A naive implementation of read-write lock.
class RWLock {
 public:
  RWLock():read_cnt(0), write_flag(false) {}
  ~RWLock() {
    read_mtx.unlock(); write_mtx.unlock();
  }
  inline void ReadLock() {
    read_mtx.lock(); if (++read_cnt == 1) { write_mtx.lock(); }
    read_mtx.unlock();
  }
  inline void ReadUnlock() {
    read_mtx.lock(); if (--read_cnt == 0) { write_mtx.unlock(); }
    read_mtx.unlock();
  }
  inline void WriteLock() { write_mtx.lock(); write_flag = true;}
  inline void WriteUnlock() { write_mtx.unlock(); write_flag = false;}
  inline int GetReadCount() const {return read_cnt;}
  inline bool CanRead() {return write_flag == false;}
  inline bool CanWrite() {return write_flag == false && read_cnt == 0;}

 protected:
  std::mutex read_mtx;
  std::mutex write_mtx;
  int read_cnt;
  bool write_flag;
  DISABLE_COPY_AND_ASSIGN(RWLock);
};

}  // namespace hbot



#endif /* HBOT_THREAD_HPP_ */
