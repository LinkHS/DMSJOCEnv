/*
 * registry.hpp
 *
 *	@brief Registry utility that helps to build registry singletons.
 *
 *  Originally from DMLC, modified by Alan_Huang
 */

#ifndef HBOT_BASE_REGISTRY_HPP_
#define HBOT_BASE_REGISTRY_HPP_


#include <map>
#include <string>
#include <vector>
#include "hobot_core/base/logging.hpp"

namespace hbot {
/*!
 * \brief Registry class.
 *  Registry can be used to register global singletons.
 *  The most commonly use case are factory functions.
 *
 * \tparam EntryType Type of Registry entries,
 *     EntryType need to name a name field.
 */
template<typename EntryType>
class Registry {
 public:
  /*! \return list of functions in the registry */
  inline static const std::vector<const EntryType*> &List() {
    return Get()->entry_list_;
  }

  /*!
   * \brief Find the entry with corresponding name.
   * \param name name of the function
   * \return the corresponding function, can be NULL
   */
  inline static const EntryType *Find(const std::string &name) {
    const std::map<std::string, EntryType*> &fmap = Get()->fmap_;
    typename std::map<std::string, EntryType*>::const_iterator p =
        fmap.find(name);
    if (p != fmap.end()) {
      return p->second;
    } else {
      return NULL;
    }
  }
  /*!
   * \brief Internal function to register a name function under name.
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  inline EntryType &__REGISTER__(const std::string& name) {
    CHECK_EQ(fmap_.count(name), 0)
        << name << " already registered";
    EntryType *e = new EntryType();
    e->name = name;
    fmap_[name] = e;
    entry_list_.push_back(e);
    return *e;
  }
  /*!
   * \brief Internal function to either register or get registered entry
   * \param name name of the function
   * \return ref to the registered entry, used to set properties
   */
  inline EntryType &__REGISTER_OR_GET__(const std::string& name) {
    if (fmap_.count(name) == 0) {
      return __REGISTER__(name);
    } else {
      return *fmap_.at(name);
    }
  }
  /*!
   * \brief get a singleton of the Registry.
   *  This function can be defined by DMLC_ENABLE_REGISTRY.
   * \return get a singleton
   */
  static Registry *Get();

  inline static std::string ListString() {

    std::string entrys_str;
    bool is_first = true;
    for (auto & inst : Get()->fmap_) {
      if (is_first) {
        is_first = false;
        entrys_str += inst.first;
      } else {
        entrys_str += ", " + inst.first;
      }
    }
    return entrys_str;
  }

 private:
  /*! \brief list of entry types */
  std::vector<const EntryType*> entry_list_;
  /*! \brief map of name->function */
  std::map<std::string, EntryType*> fmap_;

  /*! \brief constructor */
  Registry() {}
  /*! \brief destructor */
  ~Registry() {
    for (typename std::map<std::string, EntryType*>::iterator p = fmap_.begin();
         p != fmap_.end(); ++p) {
      delete p->second;
    }
  }
};


/*!
 * \brief Information about a parameter field in string representations.
 */
struct ParamFieldInfo {
  /*! \brief name of the field */
  std::string name;
  /*! \brief type of the field in string format */
  std::string type;
  /*!
   * \brief detailed type information string
   *  This include the default value, enum constran and typename.
   */
  std::string type_info_str;
  /*! \brief detailed description of the type */
  std::string description;
};


/*!
 * \brief Common base class for function registry.
 *
 * \code
 *  // This example demonstrates how to use Registry to create a factory of trees.
 *  struct TreeFactory :
 *      public FunctionRegEntryBase<TreeFactory, std::function<Tree*()> > {
 *  };
 *
 *  // in a independent cc file
 *  namespace dmlc {
 *  DMLC_REGISTRY_ENABLE(TreeFactory);
 *  }
 *  // register binary tree constructor into the registry.
 *  DMLC_REGISTRY_REGISTER(TreeFactory, TreeFactory, BinaryTree)
 *      .describe("Constructor of BinaryTree")
 *      .set_body([]() { return new BinaryTree(); });
 * \endcode
 *
 * \tparam EntryType The type of subclass that inheritate the base.
 * \tparam FunctionType The function type this registry is registerd.
 */
template<typename EntryType, typename FunctionType>
class FunctionRegEntryBase {
 public:
  /*! \brief name of the entry */
  std::string name;
  /*! \brief description of the entry */
  std::string description;
  /*! \brief additional arguments to the factory function */
  std::vector<ParamFieldInfo> arguments;
  /*! \brief Function body to create ProductType */
  FunctionType body;
  /*! \brief Return type of the function */
  std::string return_type;

  /*!
   * \brief Set the function body.
   * \param body Function body to set.
   * \return reference to self.
   */
  inline EntryType &set_body(FunctionType _body) {
    this->body = _body;
    return this->self();
  }
  /*!
   * \brief Describe the function.
   * \param description The description of the factory function.
   * \return reference to self.
   */
  inline EntryType &describe(const std::string & _description) {
    this->description = _description;
    return this->self();
  }
  /*!
   * \brief Add argument information to the function.
   * \param name Name of the argument.
   * \param type Type of the argument.
   * \param description Description of the argument.
   * \return reference to self.
   */
  inline EntryType &add_argument(const std::string &_name,
                                 const std::string &type,
                                 const std::string & _description) {
    ParamFieldInfo info;
    info.name = _name;
    info.type = type;
    info.type_info_str = info.type;
    info.description = _description;
    arguments.push_back(info);
    return this->self();
  }
  /*!
   * \brief Append list if arguments to the end.
   * \param args Additional list of arguments.
   * \return reference to self.
   */
  inline EntryType &add_arguments(const std::vector<ParamFieldInfo> &args) {
    arguments.insert(arguments.end(), args.begin(), args.end());
    return this->self();
  }
  /*!
  * \brief Set the return type.
  * \param type Return type of the function, could be Symbol or Symbol[]
  * \return reference to self.
  */
  inline EntryType &set_return_type(const std::string &type) {
    return_type = type;
    return this->self();
  }

 protected:
  /*!
   * \return reference of self as derived type
   */
  inline EntryType &self() {
    return *(static_cast<EntryType*>(this));
  }
};

/*!
 * \brief Macro to enable the registry of EntryType.
 * This macro must be used under namespace dmlc, and only used once in cc file.
 * \param EntryType Type of registry entry
 */
#define HBOT_REGISTRY_ENABLE(EntryType)                                 \
  template<>                                                            \
  Registry<EntryType > *Registry<EntryType >::Get() {                   \
    static Registry<EntryType > inst;                                   \
    return &inst;                                                       \
  }                                                                     \

/*!
 * \brief Generic macro to register an EntryType
 *  There is a complete example in FactoryRegistryEntryBase.
 *
 * \param EntryType The type of registry entry.
 * \param EntryTypeName The typename of EntryType, must do not contain namespace :: .
 * \param Name The name to be registered.
 * \sa FactoryRegistryEntryBase
 */
#define HBOT_REGISTRY_REGISTER(EntryType, EntryTypeName, Name)          \
  static EntryType & __make_ ## EntryTypeName ## _ ## Name ## __ =      \
      ::hbot::Registry<EntryType>::Get()->__REGISTER__(#Name)           \

/*!
 * \brief (Optional) Declare a file tag to current file that contains object registrations.
 *
 *  This will declare a dummy function that will be called by register file to
 *  incur a link dependency.
 *
 * \param UniqueTag The unique tag used to represent.
 * \sa DMLC_REGISTRY_LINK_TAG
 */
#define HBOT_REGISTRY_FILE_TAG(UniqueTag)                                \
  int __hbot_registry_file_tag_ ## UniqueTag ## __() { return 0; }

}  // namespace hbot



#endif /* HBOT_BASE_REGISTRY_HPP_ */
