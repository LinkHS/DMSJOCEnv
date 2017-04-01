/*
 * mem_manager.hpp
 *
 *      Author: Alan_Huang
 */
#include <stdint.h>
#include <stdlib.h>
#include <list>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include "hobot_core/base/base_common.hpp"
#include "hobot_core/base/logging.hpp"
#include "hobot_core/base/object_pool.hpp"
#include "hobot_core/base/thread.hpp"
#include "hobot_core/memory/syncedmem.hpp"


#ifndef HBOT_MEMORY_MEM_MANAGEMENT_HPP_
#define HBOT_MEMORY_MEM_MANAGEMENT_HPP_

namespace hbot {

struct SyncedMemPtrEntry{
  SyncedMemPtrEntry(SyncedMemory* ptr, void* start)
    :synced_mem(ptr), mem_start(start) {}
  SyncedMemory* synced_mem;
  void* mem_start;
  bool operator<(const SyncedMemPtrEntry &b)const{
    return (synced_mem->size() < b.synced_mem->size() ) ||
        (synced_mem->size() == b.synced_mem->size()
        && mem_start < b.mem_start);
  }
};

typedef std::pair< std::set<SyncedMemPtrEntry>::iterator, bool>
        SyncedMemPtrEntrySetRet;
typedef std::set<SyncedMemPtrEntry>::iterator SyncedMemPtrEntrySetIter;

class SyncedMemPtrSet{
 public:
  inline SyncedMemPtrEntrySetRet Insert(SyncedMemPtrEntry to_insert) {
    return set_.insert(to_insert);
  }
  inline SyncedMemPtrEntrySetIter Lower_bound(
      const SyncedMemPtrEntry &to_query) {return set_.lower_bound(to_query);}
  inline int remove(SyncedMemPtrEntry to_remove) {
    return set_.erase(to_remove);
  }
  inline SyncedMemPtrEntrySetIter begin() { return set_.begin();}
  inline SyncedMemPtrEntrySetIter end() {return set_.end();}
  inline int size() {return set_.size();}
  inline void clear() {set_.clear();}
  std::set<SyncedMemPtrEntry> set_;
};

typedef std::list<SyncedMemory*>::iterator SyncedMemoryPtrListIter;
class OrderedSyncedMemPtrList{
 public:
  template <typename Condition>
  inline SyncedMemory* GetIf(Condition cond) {
    std::list<SyncedMemory*>::iterator it = list_.begin();
    for ( ; it != list_.end(); ++it) {
      if (cond(it)) {
        return (*it);
      }
    }
    return nullptr;
  }
  template <typename Condition>
  inline void InsertIf(SyncedMemory* inst, Condition cond) {
    std::list<SyncedMemory*>::iterator it = list_.begin();
    for ( ; it != list_.end(); ++it) {
        if (cond(it)) {
          break;
        }
    }
    list_.insert(it, inst);
  }

  inline void remove(SyncedMemory* inst) { list_.remove(inst); }
  inline void clear() {list_.clear();}
  inline SyncedMemoryPtrListIter begin() { return list_.begin();}
  inline SyncedMemoryPtrListIter end() {return list_.end();}
  inline void erase(const SyncedMemoryPtrListIter& it) { list_.erase(it);}
  inline int size() {
    int count = 0;
    for ( std::list<SyncedMemory*>::iterator it = list_.begin();
        it != list_.end(); ++it) {
      ++count;
    }
    return count;
  }
  std::list<SyncedMemory*> list_;
};

/*
 * @brief SyncedMemoryPool for memory block lazy allocation. Call LazyAlloc to allocate memory block,
 * 				and call Recycle to free memory block . The following class, SharedSyncedMemory, shows
 * 				how to use SyncedMemoryPool.
 */
template<typename EntryType, typename EntryHandleType>
class BaseSyncedMemoryPool {
 public:
  virtual ~BaseSyncedMemoryPool() {}
  inline int64_t size_in_use() {return size_in_use_;}
  inline int64_t size_available() {return size_available_;}

  // @brief Allocate lazily a block of SyncedMemory
  SyncedMemory* LazyAlloc(size_t size);

  // @brief Recycle the SyncedMemory. They are collected for reuse.
  bool Recycle(SyncedMemory* inst);

  // @brief Free all available memory blocks whose size is less than a threshold
  virtual void ShrinkAvailable(size_t  size) = 0;

  // @brief Print the memory usage.
  virtual std::string StateString();

 protected:
  BaseSyncedMemoryPool();

  /* @brief find a block of available allocated memory larger than "size", and store the iterators
   * 				in mem_list_iter and mem_flag_iter.  If cannot find one, both iterators point to the end.
   */
  virtual SyncedMemory* GetLowBoundAvailable(size_t size) = 0;

  // @brief Insert a new memory block into pool.
  virtual SyncedMemory* New(size_t size) = 0;

  /*
   * virtual void RecycleSyncedMemory(SyncedMemory* inst,  size_t size) = 0;
   */
  virtual void RecycleSyncedMemory(SyncedMemory* inst) = 0;

  virtual void RemoveFromAvailableList(SyncedMemory* inst) = 0;

  // @brief Pool that store all allocated memory
  std::shared_ptr<ObjectPool<EntryType> >   objpool_mem_entry_ref_;


  /*@brief mem_map_table_ store all SyncedMemory* . It is also a hash_map to
   *        find the EntryType in pool according to SyncedMemory*.
   */
  std::unordered_map<SyncedMemory*, EntryHandleType > mem_map_table_;

  RWLock rwlock;

  int64_t size_in_use_;
  int64_t size_available_;

  DISABLE_COPY_AND_ASSIGN(BaseSyncedMemoryPool);
};



/*
 * @brief The SyncedMemoryPoolEntry holds pointer to allocated SyncedMemory and ref_count.
 * 			  Note that one SyncedMemoryPoolEntry might be shared by more than one block of SyncedMemory
 */
struct SyncedMemoryPoolEntry {
  SyncedMemoryPoolEntry():synced_mem(nullptr), ref_count(0) {}
  SyncedMemory* synced_mem;
  size_t  ref_count;
};

/*
 * @brief The SyncedMemoryPoolEntryHandle holds pointer to allocated SyncedMemoryPoolEntry.
 * 				No more than two SyncedMemoryPoolEntryHandles can point to the same SyncedMemoryPoolEntry*.
 */
struct SyncedMemoryPoolEntryHandle {
  inline SyncedMemoryPoolEntryHandle():entry_handle(nullptr) {}
  inline explicit SyncedMemoryPoolEntryHandle(SyncedMemoryPoolEntry* ptr)
      :entry_handle(ptr) {}
  SyncedMemoryPoolEntry* entry_handle;
};

typedef std::list<SyncedMemoryPoolEntry>::iterator SyncedMemoryPoolEntryIter;


struct PredicateInsertMarginalGreater {
  inline explicit PredicateInsertMarginalGreater(size_t _size):size(_size) {}
  bool operator()(const SyncedMemoryPtrListIter& it) const {
    return ((*it)->size() >= size) ? true:false;
  }
  const size_t size;
};
/*
 * @brief SyncedMemoryPool for memory block lazy allocation. Call LazyAlloc to allocate memory block,
 * 				and call Recycle to free memory block . The following class, SharedSyncedMemory, shows
 * 				how to use SyncedMemoryPool.
 */
class SyncedMemoryPool: public BaseSyncedMemoryPool<
      SyncedMemoryPoolEntry, SyncedMemoryPoolEntryHandle> {
 public:
  virtual ~SyncedMemoryPool();
  static SyncedMemoryPool* Get() {
    static SyncedMemoryPool instance;
    return &instance;
  }
  virtual void ShrinkAvailable(size_t size) override;

 protected:
  SyncedMemoryPool();

  /* @brief find a block of available allocated memory larger than "size", and store the iterators
   * 				in mem_list_iter and mem_flag_iter.  If cannot find one, both iterators point to the end.
   */
  virtual SyncedMemory* GetLowBoundAvailable(size_t size) override;


  // @brief Insert a new memory block into pool.
  virtual SyncedMemory* New(size_t size) override;

  constexpr static size_t kAlign = 32;

  OrderedSyncedMemPtrList available_list_;

  virtual void RecycleSyncedMemory(SyncedMemory* inst) override {
    available_list_.InsertIf(inst,
        PredicateInsertMarginalGreater(inst->size()));
  }

  virtual void RemoveFromAvailableList(SyncedMemory* inst) override {
    available_list_.remove(inst);
  }

  /*  @brief inherited members
  //@brief Pool that store all allocated memory
  std::shared_ptr<ObjectPool<SyncedMemoryPoolEntry> >   objpool_mem_entry_ref_;

  //@brief Available memory list. Any malloc request will check this list first to see if memory can be reused.
  OrderedSyncedMemPtrList available_list_;

  //@brief mem_map_table_ store all SyncedMemory* . It is also a hash_map to find the SyncedMemoryPoolEntryHandle in pool according to SyncedMemory*.
  std::unordered_map<SyncedMemory*, SyncedMemoryPoolEntryHandle > mem_map_table_;
   */

  DISABLE_COPY_AND_ASSIGN(SyncedMemoryPool);
};


/*
 * @brief Similar to SyncedMemoryPoolEntryHandle. But any AllocatedSyncedMemoryEntryHandle can point to
 * 				the same SyncedMemoryPoolEntry*. The member offset record the memory address offset of SyncedMemory
 * 				in SyncedMemoryPoolEntry.
 *
 */
struct AllocatedSyncedMemoryEntryHandle{
  inline AllocatedSyncedMemoryEntryHandle():entry_handle(nullptr), offset(0) {}
  inline AllocatedSyncedMemoryEntryHandle(SyncedMemoryPoolEntry* _entry,
      size_t _offset):entry_handle(_entry), offset(_offset) {}
  SyncedMemoryPoolEntry* entry_handle;
  size_t   offset;
};

/*
 * @brief ChunkAllocatedSyncedMemoryPool for memory block lazy allocation. Unlike SyncedMemoryPool,
 * 				the memory are allocated by memory pages. Different blobks of SyncedMemory could share
 * 				one memory page.
 */

class ChunkAllocatedSyncedMemoryPool: public BaseSyncedMemoryPool<
    SyncedMemoryPoolEntry, AllocatedSyncedMemoryEntryHandle> {
 public:
  virtual ~ChunkAllocatedSyncedMemoryPool();
  static ChunkAllocatedSyncedMemoryPool* Get() {
    static ChunkAllocatedSyncedMemoryPool instance;
    return &instance;
  }
  virtual void ShrinkAvailable(size_t  size) override;
  virtual std::string StateString() override;

 protected:
  ChunkAllocatedSyncedMemoryPool();

  /* @brief Get or split one instance of SyncedMemory* with size from "SyncedMemory* inst". The flag
   * 				is_in_list should be true if  "SyncedMemory* inst" is in the available list, so that the
   * 				"SyncedMemory* inst" can be removed properly from available list.
   */
  SyncedMemory* GetOrSplitFromAllocatedSyncedMemoryEntryHandle(
      AllocatedSyncedMemoryEntryHandle handle,
      SyncedMemory* inst, size_t size, bool is_in_list);

  /* @brief find a block of available allocated memory larger than "size", and store the iterators
   * 				in mem_list_iter and mem_flag_iter.  If cannot find one, both iterators point to the end.
   */
  virtual SyncedMemory* GetLowBoundAvailable(size_t size) override;
  // @brief Insert a new memory block into pool.
  virtual SyncedMemory* New(size_t size) override;

  constexpr static size_t kAlign = 32;


  inline void Insert2AvailableList(SyncedMemory* inst) {
    if (!inst->cpu_ptr_)
      inst->cpu_data();
    SyncedMemPtrEntry entry(inst, inst->cpu_ptr_);
    available_list_.Insert(entry);
    void* start_p = static_cast<char*>(inst->cpu_ptr_);
    void* end_p = static_cast<char*>(start_p) + inst->size();
    DBG(CHECK_EQ(mem_end_table_.count(end_p), 0);)
    mem_end_table_[end_p] = inst;
    DBG(CHECK_EQ(mem_start_table_.count(start_p), 0);)
    mem_start_table_[start_p] = inst;
  }


  /*
   * @brief virtual void RecycleSyncedMemory(SyncedMemory* inst, size_t size) override;
   */
  virtual void RecycleSyncedMemory(SyncedMemory* inst) override;

  virtual void RemoveFromAvailableList(SyncedMemory* inst) override {
    if (!inst->cpu_ptr_)
      inst->cpu_data();
    SyncedMemPtrEntry entry(inst, inst->cpu_ptr_);
    void* start_p = static_cast<char*>(inst->cpu_ptr_);
    void* end_p = static_cast<char*>(start_p) + inst->size();
    CHECK_EQ(mem_end_table_.erase(end_p), 1);
    CHECK_EQ(mem_start_table_.erase(start_p), 1);
    available_list_.remove(entry);
  }

  SyncedMemPtrSet available_list_;

  // @brief A lookup table to find SyncedMemory* according to end address.
  std::unordered_map<void* , SyncedMemory* > mem_end_table_;

  // @brief A lookup table to find SyncedMemory* according to first address.
  std::unordered_map<void* , SyncedMemory* > mem_start_table_;

  // @brief A unique mapping to map SyncedMemory* to actually
  //        allocated SyncedMemoryPoolEntry.
  std::unordered_map<SyncedMemory*,
        AllocatedSyncedMemoryEntryHandle> allocated_mem_map_;

  /*  @brief inherited members
     @brief Pool that store all allocated memory. For
  std::shared_ptr<ObjectPool<SyncedMemoryPoolEntry> >   objpool_mem_entry_ref_;

    @brief Available memory list. Any malloc request will check this list first to see
          if memory can be reused
  OrderedSyncedMemPtrList available_list_;

   @brief mem_map_table_ store all SyncedMemory* . It is also a hash_map to
     find the EntryType in pool according to SyncedMemory*.
  std::unordered_map<SyncedMemory*, AllocatedSyncedMemoryEntryHandle > mem_map_table_;
  *
  */

  DISABLE_COPY_AND_ASSIGN(ChunkAllocatedSyncedMemoryPool);
};


/**
 * @brief Lazy allocated MemoryBlock that allocated from SyncedMemoryPool(if size > threshold).
 * 				It provided a limited garbage-collection facility and data sharing by using
 * 				reference count and SyncedMemoryPool inside.
 *
 * 				SharedSyncedMemory also can be initialized by providing memory address.
 */
template<size_t Pool_threshold>
class SharedSyncedMemory{
 public:
  SharedSyncedMemory() {
    rwlock_ = new RWLock();
    rwlock_->WriteLock();
    mem_ = new SyncedMemory();
    count_ = new size_t;
    *count_ = 1;
    own_data_ = true;
    rwlock_->WriteUnlock();
  }
  // @brief If additional memory address is provided, just construct from it.
  SharedSyncedMemory(size_t size, void* cpu_data_ptr = nullptr,
      void* gpu_data_ptr = nullptr) {
    rwlock_ = new RWLock();
    rwlock_->WriteLock();
    if (size < Pool_threshold || cpu_data_ptr != nullptr ||
        gpu_data_ptr != nullptr) {
      mem_ = new SyncedMemory(size);
      own_data_ = true;
      if (cpu_data_ptr != nullptr) {
        mem_->set_cpu_data(cpu_data_ptr);
        own_data_ = false;
      }
      if (gpu_data_ptr != nullptr) {
        mem_->set_gpu_data(gpu_data_ptr);
        own_data_ = false;
      }
    } else {
      ChunkAllocatedSyncedMemoryPool* pool =
          ChunkAllocatedSyncedMemoryPool::Get();
//      SyncedMemoryPool * pool = SyncedMemoryPool::Get();
      mem_ = pool->LazyAlloc(size);
      own_data_ = true;
    }
    count_ = new size_t;
    *count_ = 1;
    rwlock_->WriteUnlock();
  }
  SharedSyncedMemory(const SharedSyncedMemory& src) {
    rwlock_ = src.rwlock_;
    rwlock_->WriteLock();
    mem_ = src.mem(); count_ = src.count_;
    (*count_) = (*count_) +1;
    own_data_ = src.own_data_;
    rwlock_->WriteUnlock();
  }
  SharedSyncedMemory& operator=(const SharedSyncedMemory& src) {
    deconstruct();
    rwlock_ = src.rwlock_;
    rwlock_->WriteLock();
    mem_ = src.mem(); count_ = src.count_;
    (*count_) = (*count_) +1;
    own_data_ = src.own_data_;
    rwlock_->WriteUnlock();
    return *this;
  }
  ~SharedSyncedMemory() { deconstruct(); }
  inline SyncedMemory * mem() const { return mem_; }
  inline size_t count() const { return *count_ ; }

 protected:
  void deconstruct() {
    rwlock_->WriteLock();
    (*count_) = (*count_) -1;
    if ((*count_) == 0) {
      delete count_;
      if (mem_->size() < Pool_threshold || own_data_ == false) {
        delete mem_;
      } else {
        ChunkAllocatedSyncedMemoryPool* pool =
            ChunkAllocatedSyncedMemoryPool::Get();
//        SyncedMemoryPool * pool = SyncedMemoryPool::Get();
        pool->Recycle(mem_);
      }
      rwlock_->WriteUnlock();
      delete rwlock_; return;
    }
    rwlock_->WriteUnlock();
  }

  SyncedMemory * mem_;
  size_t * count_;
  RWLock * rwlock_;
  bool own_data_;
};


template<typename EntryType, typename EntryHandleType>
BaseSyncedMemoryPool<EntryType, EntryHandleType>::BaseSyncedMemoryPool() {
  size_in_use_ = size_available_ = 0;
//  available_list_.clear();
  objpool_mem_entry_ref_  = ObjectPool<EntryType>::_GetSharedRef();
  mem_map_table_.clear();
}


template<typename EntryType, typename EntryHandleType>
SyncedMemory* BaseSyncedMemoryPool<EntryType, EntryHandleType>::
    LazyAlloc(size_t size) {
//  ShrinkAvailable(size);
  rwlock.WriteLock();
  SyncedMemory* inst =  GetLowBoundAvailable(size);
  if (inst == nullptr) {
    inst = New(size);
  } else {
//    DBG(LOG(INFO)<<"reuse size "<<size);
  }

  EntryHandleType handle = mem_map_table_[inst];
  handle.entry_handle->ref_count += 1;
  size_t changed_size = inst->size();
  size_available_ -= changed_size;
  size_in_use_ += changed_size;
  RemoveFromAvailableList(inst);
  rwlock.WriteUnlock();
  return inst;
}

template<typename EntryType, typename EntryHandleType>
bool BaseSyncedMemoryPool<EntryType, EntryHandleType>::
  Recycle(SyncedMemory* inst) {
  rwlock.WriteLock();
  DBG(CHECK_EQ(mem_map_table_.count(inst), 1));
  EntryHandleType handle = mem_map_table_[inst];
  handle.entry_handle->ref_count -= 1;
  size_t changed_size = inst->size();
  size_available_ += changed_size;
  size_in_use_ -= changed_size;
//  RecycleSyncedMemory(inst,changed_size );
  RecycleSyncedMemory(inst);
  rwlock.WriteUnlock();
  return true;
}

template<typename EntryType, typename EntryHandleType>
std::string BaseSyncedMemoryPool<EntryType, EntryHandleType>::
  StateString() {
  rwlock.ReadLock();
  std::ostringstream out_stream;
  int64_t _size_in_use = 0;
  int64_t _size_available = 0;
  typename std::unordered_map<SyncedMemory*, EntryHandleType >::iterator
    it = mem_map_table_.begin();
  int id = 0;
  out_stream << std::endl;
  for ( ; it != mem_map_table_.end(); ++it) {
    EntryType* cur = it->second.entry_handle;
    if (cur->ref_count == 0) {
      _size_available += cur->synced_mem->size();
      out_stream << "Block " << id << " (available) has " <<
          cur->synced_mem->size() << " bytes. ref_count: " <<
          cur->ref_count << std::endl;
    } else {
      _size_in_use += cur->synced_mem->size();
      out_stream <<"Block " << id << " (in use) has " <<
          cur->synced_mem->size() << " bytes. ref_count: "
          << cur->ref_count << std::endl;
    }
    ++id;
  }
  rwlock.ReadUnlock();
  out_stream << "Total Memory Usage in SyncedMemoryPool: "<< std::endl <<
      "chunked size_total: " << _size_available + _size_in_use << " (" <<
      size_available_ + size_in_use_ << ")" << std::endl <<
      "chunked size_in_use: " << _size_in_use << " (" << size_in_use_ << ")" <<
      std::endl << "chunked size_available: " << _size_available <<
      " (" << size_available_ << ")";
  return out_stream.str();
}


}  // namespace hbot

#endif  // HBOT_MEMORY_MEM_MANAGEMENT_HPP_
