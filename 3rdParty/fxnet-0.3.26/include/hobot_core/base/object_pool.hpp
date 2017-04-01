/*
 * chunked_list.hpp
 *
 *	Modified from MXNET.
 *      Author: Alan_Huang
 */

#ifndef HBOT_CHUNK_LIST_HPP_
#define HBOT_CHUNK_LIST_HPP_

#include <stdlib.h>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include "hobot_core/base/base_common.hpp"
#include "hobot_core/base/logging.hpp"
namespace hbot {
/*!
 * \brief Object pool for fast allocation and deallocation.
 */
template <typename T>
class ObjectPool {
 public:
  /*!
   * \brief Destructor.
   */
  ~ObjectPool();
  /*!
   * \brief Create new object.
   * \return Pointer to the new object.
   */
//  template <typename... Args>
//  T* New(Args&&... args);

  T* New();
  /*!
   * \brief Delete an existing object.
   * \param ptr The pointer to delete.
   *
   * Make sure the pointer to delete is allocated from this pool.
   */
  void Delete(T* ptr);

  /*!
   * \brief Get singleton instance of pool.
   * \return Object Pool.
   */
  static ObjectPool* Get();

  /*!
   * \brief Get a shared ptr of the singleton instance of pool.
   * \return Shared pointer to the Object Pool.
   */
  static std::shared_ptr<ObjectPool> _GetSharedRef();

 private:
  /*!
   * \brief Internal structure to hold pointers.
   */
  struct LinkedList {
    LinkedList():next(nullptr) {}
    T t;
    LinkedList* next;
  };
  /*!
   * \brief Page size of allocation.
   *
   * Currently defined to be 4KB.
   */
  constexpr static std::size_t kPageSize = PAGE_SIZE;
  /*! \brief internal mutex */
  std::mutex m_;
  /*!
   * \brief Head of free list.
   */
  LinkedList* head_;
  /*!
   * \brief Pages allocated.
   */
  std::vector<void*> allocated_;
  /*!
   * \brief Private constructor.
   */
  ObjectPool();
  /*!
   * \brief Allocate a page of raw objects.
   *
   * This function is not protected and must be called with caution.
   */
  void AllocateChunk();
  DISABLE_COPY_AND_ASSIGN(ObjectPool);
};  // class ObjectPool

/*!
 * \brief Helper trait class for easy allocation and deallocation.
 */
template <typename T>
struct ObjectPoolAllocatable {
  /*!
   * \brief Create new object.
   * \return Pointer to the new object.
   */
//  template <typename... Args>
//  static T* New(Args&&... args);
  static T* New();


  /*!
   * \brief Delete an existing object.
   * \param ptr The pointer to delete.
   *
   * Make sure the pointer to delete is allocated from this pool.
   */
  static void Delete(T* ptr);
};  // struct ObjectPoolAllocatable

template <typename T>
ObjectPool<T>::~ObjectPool() {
// TODO(hotpxl): mind destruction order
  for (auto i : allocated_) {
    memfree(i);
  }
}

template <typename T>
//  template <typename... Args>
//  T* ObjectPool<T>::New(Args&&... args) {
T* ObjectPool<T>::New() {
  LinkedList* ret;
  {
    std::lock_guard<std::mutex> lock{m_};
    if (head_->next == nullptr) {
      AllocateChunk();
    }
    ret = head_;
    head_ = head_->next;
  }
//  return new (static_cast<void*>(ret)) T(std::forward<Args>(args)...);
  return new (static_cast<void*>(ret)) T();
}

template <typename T>
void ObjectPool<T>::Delete(T* ptr) {
  ptr->~T();
  auto linked_list_ptr = reinterpret_cast<LinkedList*>(ptr);
  {
    std::lock_guard<std::mutex> lock(m_);
    linked_list_ptr->next = head_;
    head_ = linked_list_ptr;
  }
}

template <typename T>
ObjectPool<T>* ObjectPool<T>::Get() {
  return _GetSharedRef().get();
}

template <typename T>
std::shared_ptr<ObjectPool<T> > ObjectPool<T>::_GetSharedRef() {
  static std::shared_ptr<ObjectPool<T> > inst_ptr(new ObjectPool<T>());
//  static auto inst_ptr = std::make_shared(new ObjectPool<T>());
  return inst_ptr;
}

template <typename T>
ObjectPool<T>::ObjectPool():head_(nullptr) {
  AllocateChunk();
}

template <typename T>
void ObjectPool<T>::AllocateChunk() {
  static_assert(sizeof(LinkedList) <= kPageSize, "Object too big.");
  static_assert(sizeof(LinkedList) % alignof(LinkedList) == 0,
      "ObjectPooll Invariant");
  static_assert(alignof(LinkedList) % alignof(T) == 0,
      "ObjectPooll Invariant");
  static_assert(kPageSize % alignof(LinkedList) == 0, "ObjectPooll Invariant");
  void* new_chunk_ptr;

  new_chunk_ptr = memalign(kPageSize, kPageSize);
  CHECK(new_chunk_ptr != NULL) << "Allocation failed";

  allocated_.emplace_back(new_chunk_ptr);
  auto new_chunk = static_cast<LinkedList*>(new_chunk_ptr);
  auto size = kPageSize / sizeof(LinkedList);
  for (std::size_t i = 0; i < size - 1; ++i) {
    new_chunk[i].next = &new_chunk[i + 1];
  }
  new_chunk[size - 1].next = head_;
  head_ = new_chunk;
}

template <typename T>
//  template <typename... Args>
//  T* ObjectPoolAllocatable<T>::New(Args&&... args) {
T* ObjectPoolAllocatable<T>::New() {
  return ObjectPool<T>::Get()->New();
}

template <typename T>
void ObjectPoolAllocatable<T>::Delete(T* ptr) {
  ObjectPool<T>::Get()->Delete(ptr);
}

}  // namespace hbot

#endif /* HBOT_CHUNK_LIST_HPP_ */
