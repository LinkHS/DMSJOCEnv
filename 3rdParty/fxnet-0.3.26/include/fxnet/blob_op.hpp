/*
 * blob_op.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef HBOT_BLOB_OP_HPP_
#define HBOT_BLOB_OP_HPP_

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <iostream>  //  NOLINT(readability/streams)
#include "fxnet/blob.hpp"
#include "fxnet/common.hpp"
#include "fxnet/util/math_functions.hpp"
#include "hobot_core/base/transform_expression.hpp"

namespace hbot {
namespace fxnet {

namespace op {

template <int N>
struct Pow2 {
  static const int value = (1 << N);
};

#define REGIST_POW2(N) case N: return Pow2<N>::value;
#define REGIST_POW2_X \
  REGIST_POW2(0) \
  REGIST_POW2(1) \
  REGIST_POW2(2) \
  REGIST_POW2(3) \
  REGIST_POW2(4) \
  REGIST_POW2(5) \
  REGIST_POW2(6) \
  REGIST_POW2(7) \
  REGIST_POW2(8) \
  REGIST_POW2(9)
#define REGIST_POW2_NX(N) \
  REGIST_POW2(N##0) \
  REGIST_POW2(N##1) \
  REGIST_POW2(N##2) \
  REGIST_POW2(N##3) \
  REGIST_POW2(N##4) \
  REGIST_POW2(N##5) \
  REGIST_POW2(N##6) \
  REGIST_POW2(N##7) \
  REGIST_POW2(N##8) \
  REGIST_POW2(N##9)
HBOT_XINLINE int GetPow2(const int n) {
  switch(n) {
  REGIST_POW2_X
  REGIST_POW2_NX(1)
  REGIST_POW2_NX(2)
  default:
    CHECK(0) << "Unsupported pow2_n: "<< n;
  }
  return 1;
}

  struct add {
    template < typename Dtype>
    HBOT_CINLINE static void Do(const Blob<Dtype>& a, const Blob<Dtype>& b,
        Blob<Dtype>& c){    //  NOLINT(runtime/references)
      hbot_add(a.count(), a.cpu_data(), b.cpu_data(),
          c.mutable_cpu_data());
    }
  };

  struct min_and_max {
    template<typename Dtype>
    HBOT_CINLINE static void Do(const Blob<Dtype>&a, Dtype &min,  //  NOLINT
        Dtype& max) {  //  NOLINT(runtime/references)
      const Dtype* data = a.cpu_data();
      for ( int i = 0 ; i < a.count() ; ++i ) {
        max = std::max(data[i], max);
        min = std::min(data[i], min);
      }
    }
  };

  struct axpb {
    template<typename Dtype>
    HBOT_CINLINE static void Do(const Blob<Dtype>&a, Dtype alpha,  //  NOLINT
        Dtype beta){
      const Dtype* data = a.cpu_data();
      for ( int i = 0; i < a.count(); ++i ) {
        Dtype value = data[i]*alpha + beta;
        data[i] = value;
      }
    }
  };

  struct BoundByMinMax {
    template<typename Dtype>
    HBOT_CINLINE static void Do(const Blob<Dtype>&src, Blob<Dtype>& dst,  //  NOLINT
        const Dtype bound_scale){
      Dtype min = FLT_MAX, max = -FLT_MAX;
      min_and_max::Do(src, min, max);
      if ( max - min < 0.001 ) {
        min = 0;
        max = bound_scale;
      }
      const Dtype* in_data = src.cpu_data();
      Dtype* out_data = dst.mutable_cpu_data();
      int count = src.count();
      Dtype alpha = 1.0 / (max-min) * bound_scale;
      Dtype beta = -(min/(max-min) + 0.5) * bound_scale;
      for (int i = 0; i < count; ++i) {
        Dtype temp = Dtype(static_cast<int>(in_data[i]*alpha + beta));
        out_data[i] = (temp - beta)/alpha;
      }
    }
  };

  struct BoundByMaxAbs  {
    template<typename Dtype>
    HBOT_CINLINE static void Do(const Blob<Dtype>&src, Blob<Dtype>& dst, //  NOLINT
        const Dtype bound_scale){
      Dtype min = FLT_MAX, max = -FLT_MAX;
      min_and_max::Do(src, min, max);
      Dtype abs_max = std::max(std::abs(min), std::abs(max));
      if ( abs_max < 0.001 ) {
        abs_max = bound_scale;
      }

      const Dtype* in_data = src.cpu_data();
      Dtype* out_data = dst.mutable_cpu_data();
      int count = src.count();
      for (int i = 0; i < count; ++i) {
        Dtype temp = Dtype (static_cast<int>((in_data[i])/(abs_max)*bound_scale)); //  NOLINT
        out_data[i] = temp / bound_scale * abs_max;
      }
    }
  };


  template<typename Dtype>
  struct LeftShiftParam {
    LeftShiftParam(int n_bit_, Dtype low_bound_, Dtype up_bound_, int size_)
      : n_bit(n_bit_), low_bound(low_bound_), up_bound(up_bound_),
            size(size_), old_n_bit(1000) {
      if (n_bit > 1000 || n_bit < -1000) {
        std::cout << "n_bit_ is invalid " << n_bit_ << std::endl;
        assert_force(0);
      }
    }

    inline friend std::ostream& operator << (std::ostream& out,
        LeftShiftParam<Dtype>& param ) {
      out << "n_bit: " << param.n_bit << " low_bound: " << param.low_bound <<
          " up_bound: " << param.up_bound << " size: " << param.size <<
          " old_n_bit: " << param.old_n_bit;
      return out;
    }

    int n_bit;
    Dtype low_bound;
    Dtype up_bound;
    int size;
    int old_n_bit;
  };

  template<typename Dtype>
  struct PtrConst {
    explicit inline PtrConst(const Dtype* && p): ptr(p) {}
    const Dtype*  ptr;
  };
  template<typename Dtype>
  struct PtrMutable {
    explicit inline PtrMutable(Dtype* && p):ptr(p) {};  //NOLINT
    Dtype*  ptr;
  };

  struct LeftShift {
  template< typename BType, typename InType, typename Dtype>
    HBOT_CINLINE static void Do(const LeftShiftParam<BType>& param,
          const PtrConst<InType>& src_, PtrMutable<Dtype>& dst_ ) {  //NOLINT
      const InType* src = src_.ptr;
      Dtype* dst = dst_.ptr;
      int n_bit = param.n_bit;
      const BType up_bound = param.up_bound;
      const BType low_bound = param.low_bound;
      const int count = param.size;
      if (n_bit > 0) {  // left shift
        for (int i = 0; i < count; ++i) {
          BType res = src[i];
          res *= GetPow2(n_bit);
          res = Dtype(res);
          res = std::max(low_bound, std::min(res, up_bound));
          dst[i] = res;
        }
      } else {   // right shift
        n_bit *= -1;
        for (int i = 0; i < count; ++i) {
          BType res = src[i];
          res /= GetPow2(n_bit);
          res = Dtype(res);
          res = std::max(low_bound, std::min(res, up_bound));
          dst[i] = res;
        }
      }
    }
  HBOT_CINLINE static void Do(const LeftShiftParam<int32_t>& param,
        const PtrConst<int32_t>& src_, PtrMutable<int32_t>& dst_ ) {  //NOLINT
    const int32_t* src = src_.ptr;
    int32_t* dst = dst_.ptr;
    int n_bit = param.n_bit;
    const int32_t up_bound = param.up_bound;
    const int32_t low_bound = param.low_bound;
    const int count = param.size;
    if (n_bit > 0) {  // left shift
      for (int i = 0; i < count; ++i) {
        int32_t res = (src[i] << n_bit);
        res = std::max(low_bound, std::min(res, up_bound));
        dst[i] = res;
      }
    } else {   // right shift
      n_bit *= -1;
      for (int i = 0; i < count; ++i) {
        int32_t res = (src[i] >> n_bit);
        res = std::max(low_bound, std::min(res, up_bound));
        dst[i] = res;
      }
    }
  }

  };

  struct BoundByLeftShift{
    template<typename BType, typename InType, typename Dtype>
      HBOT_CINLINE static void Do(const LeftShiftParam<BType>& param,
        const PtrConst<InType>& src_, PtrMutable<Dtype>& dst_ ) {  //NOLINT
        const InType* src = src_.ptr;
        Dtype* dst = dst_.ptr;
        int n_bit = param.n_bit;
        int old_n_bit = param.old_n_bit;
        if (old_n_bit > 100 || old_n_bit < -100) {
          old_n_bit = n_bit;
        }
        old_n_bit = std::max(n_bit, old_n_bit);
        const BType up_bound = param.up_bound;
        const BType low_bound = param.low_bound;
        const int count = param.size;
        if (n_bit > 0) {  // left shift
          for (int i = 0; i < count; ++i) {
            BType res = src[i];
            res *= GetPow2(old_n_bit);
            res = (static_cast<int>(res) >> (old_n_bit - n_bit));
            res = std::max(low_bound, std::min(res, up_bound));
            res /= GetPow2(n_bit);
            dst[i] = res;
          }
        } else {  // right shift
          n_bit *= -1;
          old_n_bit *= -1;
          for (int i = 0; i < count; ++i) {
            BType res = src[i];
            res /= GetPow2(old_n_bit);
            res = (static_cast<int>(res) >> (n_bit - old_n_bit));
            res = std::max(low_bound, std::min(res, up_bound));
            res *= GetPow2(n_bit);
            dst[i] = res;
          }
        }
      }
  };

  struct QuantizeParam {
    QuantizeParam(const int shift_num_, const int valid_bit_num_) {
      shift_num = shift_num_;
      valid_bit_num = valid_bit_num_;
    }
    int shift_num;
    int valid_bit_num;
  };
  struct QuantizeBlob{
    template<typename Dtype>
    HBOT_CINLINE static void Do(const Blob<Dtype>& src_blob,
        Blob<Dtype>& dst_blob, QuantizeParam& param) {    // NOLINT
      op::LeftShiftParam<Dtype> left_shift_param(param.shift_num,
          n_bit_int_lower_bound(param.valid_bit_num),
          n_bit_int_upper_bound(param.valid_bit_num),
          src_blob.count());
      left_shift_param.old_n_bit = src_blob.shift_num_;
      op::PtrConst<Dtype> src(src_blob.cpu_data());
      op::PtrMutable<Dtype> dst(dst_blob.mutable_cpu_data());
      hbot_transform<op::BoundByLeftShift>(left_shift_param, src, dst);
    }
  };

}  // namespace op


}  // namespace fxnet
}  //  namespace hbot

#endif /* HBOT_BLOB_OP_HPP_ */
