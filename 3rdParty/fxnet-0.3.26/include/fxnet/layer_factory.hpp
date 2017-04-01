/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers could be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef FXNET_LAYER_FACTORY_H_
#define FXNET_LAYER_FACTORY_H_

#include <functional>
#include <map>
#include <string>
#include <vector>
#include "fxnet/common.hpp"
#include "fxnet/proto/fxnet.pb.h"
#include "hobot_core/base/logging.hpp"
#include "hobot_core/base/registry.hpp"
#include "hobot_core/base/thread.hpp"

namespace hbot {
namespace fxnet {

template <typename Dtype>
class Layer;

template <typename Dtype>
class FPGALayer;

#define DEFINE_FXNET_FACTORY(FactoryName, EntryName)  \
template<typename Dtype>  \
struct FactoryName:  \
  public FunctionRegEntryBase<FactoryName<Dtype>, \
  std::function< EntryName<Dtype>*(const LayerParameter&)> > {} \


DEFINE_FXNET_FACTORY(FXNetLayerFactory, Layer);
DEFINE_FXNET_FACTORY(FXNetFPGALayerFactory, FPGALayer);


#define DEFINE_FXNET_REGISTRY(RegistryName, FactoryName, EntryName)  \
template <typename Dtype>  \
class RegistryName { \
 public: \
  typedef EntryName<Dtype>* (*Creator)(const LayerParameter&); \
  static void AddCreator(const string& type, Creator creator) { \
    CHECK(Registry<FactoryName<Dtype> >::Find(type) == NULL)<< \
        "Layer type " << type << " already registered."; \
    string description("construct of layer ");  \
    description = description + type;  \
    Registry<FactoryName<Dtype> >::Get()->__REGISTER__(type) \
        .describe(description) \
        .set_body(creator); \
  } \
  static EntryName<Dtype>* CreateLayer(const LayerParameter& param) { \
    DBG(std::cout << "Creating layer " << param.name() << std::endl;)  \
    const string& type = param.type();   \
    const FactoryName<Dtype>* creator_ptr =  \
      Registry<FactoryName<Dtype> >::Find(type) ;  \
    CHECK(creator_ptr != NULL)<< "Unknown layer type: " << type << \
        " (known types: " <<   Registry<FactoryName<Dtype> >::ListString() \
        << ")";  \
    return creator_ptr->body(param);  \
  } \
 private: \
  RegistryName() {} \
}

DEFINE_FXNET_REGISTRY(FXNetLayerRegistry, FXNetLayerFactory, Layer);
DEFINE_FXNET_REGISTRY(FXNetFPGALayerRegistry, FXNetFPGALayerFactory, FPGALayer);

template<typename Dtype>
struct FPGALayerNameEntry{
  FPGALayerNameEntry() {}
  std::string name;
};


template <typename Dtype>
class FXNetFPGALayerCorres {
 public:
  static void AddCorresMap(const string& type, string fpga_type) {
    const FPGALayerNameEntry<Dtype> * creator_ptr =
        Registry<FPGALayerNameEntry<Dtype> >::Find(type);
    CHECK(creator_ptr == NULL) <<"Layer type " << type <<
        " already has FGPA layer " <<creator_ptr->name <<" .";
    Registry<FPGALayerNameEntry<Dtype> >::Get()->__REGISTER__(type).name =
        fpga_type;
  }
  static string GetFPGACorres(const string& type) {
    const FPGALayerNameEntry<Dtype> * creator_ptr =
        Registry<FPGALayerNameEntry<Dtype> >::Find(type);
    CHECK(creator_ptr != NULL) <<"Unknown corresponding layer type: "
        << type << " (known types: " <<
        Registry<FPGALayerNameEntry<Dtype> >::ListString() << ")";
    return creator_ptr->name;
  }
 private:
  FXNetFPGALayerCorres(){}
};

template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                   Layer<Dtype>* (*creator)(const LayerParameter&)) {
    static RWLock rwlock;
    rwlock.WriteLock();
    FXNetLayerRegistry<Dtype>::AddCreator(type, creator);
    rwlock.WriteUnlock();
  }
};

template <typename Dtype>
class FPGALayerRegisterer {
 public:
  FPGALayerRegisterer(const string& type,
                   FPGALayer<Dtype>* (*creator)(const LayerParameter&)) {
    static RWLock rwlock;
    rwlock.WriteLock();
    FXNetFPGALayerRegistry<Dtype>::AddCreator(type, creator);
    rwlock.WriteUnlock();
  }
};

template <typename Dtype>
class FPGALayerCorresRegisterer{
 public:
  FPGALayerCorresRegisterer(const string& type, const string& fpga_type){
    static RWLock rwlock;
    rwlock.WriteLock();
    FXNetFPGALayerCorres<Dtype>::AddCorresMap(type, fpga_type);
    rwlock.WriteUnlock();
  }
};

#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define FPGA_REGISTER_LAYER_CREATOR(type, creator)                          \
  static FPGALayerRegisterer<float> g_creator_f_##type(#type, creator<float>);\
  static FPGALayerRegisterer<double> g_creator_d_##type(#type, creator<double>)\


#define SET_LAYER_REGISTER_FLAG(type)                                         \
  static bool UNUSED_VAR type##_registed_flag =                               \
    type##Layer<float>::RegisterLayer() && type##Layer<double>::RegisterLayer()

#define LAYER_CREATOR(type, Dtype)                                             \
  static bool RegisterLayer();                                     \
  static LayerRegisterer<Dtype> g_creator_##type

#define FPGA_LAYER_CREATOR(type, Dtype)                                        \
  static bool RegisterLayer();     \
  static FPGALayerRegisterer<Dtype> g_creator_##type

#define REGISTER_LAYER(type)                                             \
  template <typename Dtype>                                                    \
  Layer<Dtype>* Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return (Layer<Dtype>*) (new type##Layer<Dtype>(param));           \
  }                                                                         \
  template <typename Dtype>                                               \
  bool type##Layer<Dtype>::RegisterLayer() {                              \
    UNUSED(g_creator_##type); return true; }                              \
  template <typename Dtype>                                               \
  LayerRegisterer<Dtype> type##Layer<Dtype>::g_creator_##type = \
    LayerRegisterer<Dtype>(#type, Creator_##type##Layer<Dtype>)

#define REGISTER_FPGA_LAYER_CORRES(type, fpgaType)  \
  static FPGALayerCorresRegisterer<float> g_corres_f_##type(                \
    #type, #fpgaType); \
  static FPGALayerCorresRegisterer<double> g_corres_d_##type(               \
    #type, #fpgaType)

#define REGISTER_FPGA_LAYER(type)                                             \
  template <typename Dtype>                                                    \
  FPGALayer<Dtype>* Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return (FPGALayer<Dtype>*) (new type##Layer<Dtype>(param));           \
  }                                                                         \
  template <typename Dtype>                                               \
  bool type##Layer<Dtype>::RegisterLayer() {  \
    UNUSED(g_creator_##type); return true; }       \
  template <typename Dtype>                                               \
  FPGALayerRegisterer<Dtype> type##Layer<Dtype>::g_creator_##type = \
      FPGALayerRegisterer<Dtype>(#type, Creator_##type##Layer<Dtype>)

bool RegisterEngineLayers();

static bool UNUSED_VAR engin_registed_flag = RegisterEngineLayers();

}  // namespace fxnet
}  //  namespace hbot
#endif  // FXNET_LAYER_FACTORY_H_
