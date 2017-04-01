/*
 * upscale.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef FXNET_UPSCALE_HPP_
#define FXNET_UPSCALE_HPP_

#include "fxnet/blob.hpp"

namespace hbot {
namespace fxnet {

#ifdef __CUDACC__
static __inline__ __device__ float  AtomicAdd(float* address, float val) {
  return atomicAdd(address, val);
}
static __inline__ __device__ double AtomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull =
                            (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +
                             __longlong_as_double(assumed)));
  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#else
template<typename Dtype>
static HBOT_CINLINE Dtype AtomicAdd(Dtype* address, Dtype val) {
  Dtype old = *address;
  *address += val;
  return old;
}

#endif

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &p){
  if(p.size() <= 0){ os << "[ ]"; return os; }
  os <<"["; for (int i = 0; i < p.size() -1 ; ++i) { os << p[i] << ", "; }
  os << p[p.size()-1]; os << "]"; return os;
}


inline bool operator == (const std::vector<int> &a,
    const std::vector<int>&b) {
  if(a.size() != b.size()) {
    return false;
  }
  for(int i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}


struct PointInterpolateForward {
  template <typename Dtype>
  HBOT_XINLINE static void DoEltwise(const Dtype* src,
      const int src_id, Dtype* dst, const int dst_id) {
    dst[dst_id] = src[src_id];
  }

  /* -> w   | h
   * [src_p11, src_p21]    [wX([0,256]) = projected_x - loc_x_src_p11]
   * [src_p12, src_p22]    [wY([0,256]) = projected_y - loc_y_src_p11]
   */
  template <int channel_dim, typename Dtype>
  HBOT_XINLINE static void Do(const Dtype* src_p11,
      const Dtype* src_p21, const Dtype* src_p12, const Dtype* src_p22,
      Dtype* dst_p, const int wX, const int wY) {
    int f24 = (wX * wY) / 256;
    int f14 = wY - f24;
    int f23 = wX - f24;
    int f13 = ((256 - wX) * (256 - wY)) / 256; // this one can be computed faster
  #ifdef __CUDACC__
      // Only CUDA compiler support this pragma. G++ does not.
      #pragma unroll
  #endif
    for (int i = 0; i < channel_dim; ++i) {
      *(dst_p + i) =  ((*(src_p11 + i)) * f13 + (*(src_p21 + i)) * f23 +
          (*(src_p12 + i)) * f14 + (*(src_p22 + i)) * f24) / 256;
    }
  }

};



struct PointInterpolateBackward {
  template <typename Dtype>
  HBOT_XINLINE static void DoEltwise(Dtype* src_diff,
      const int src_id, const Dtype* dst_diff, const int dst_id) {
    AtomicAdd(src_diff + src_id , dst_diff[dst_id]);
  }
  /* -> w   | h
   * [src_p11, src_p21]    [wX([0,256]) = projected_x - loc_x_src_p11]
   * [src_p12, src_p22]    [wY([0,256]) = projected_y - loc_y_src_p11]
   */
  template <typename Dtype, int channel_dim>
  HBOT_XINLINE static void Do(Dtype* src_p11_diff,
      Dtype* src_p21_diff, Dtype* src_p12_diff, Dtype* src_p22_diff,
      const Dtype* dst_p_diff, const int wX, const int wY) {
    int f24 = (wX * wY) / 256;
    int f14 = wY - f24;
    int f23 = wX - f24;
    int f13 = ((256 - wX) * (256 - wY)) / 256; // this one can be computed faster
  #ifdef __CUDACC__
      // Only CUDA compiler support this pragma. G++ does not.
      #pragma unroll
  #endif
    for (int i = 0; i < channel_dim; ++i) {
      const Dtype grad = *(dst_p_diff + i) / 256;
      AtomicAdd(src_p11_diff + i, f13 * grad );
      AtomicAdd(src_p21_diff + i, f23 * grad );
      AtomicAdd(src_p12_diff + i, f14 * grad );
      AtomicAdd(src_p22_diff + i, f24 * grad );
    }
  }
};

/*
 * Up-scale border line around x axis.
 */

template <typename InterpolateAction,typename Dtype, int channel_dim>
HBOT_CINLINE void upscale_2x_line(Dtype* src_line1,
    Dtype* src_line2, Dtype* dst,
    const int dst_w, const int wY) {
  int dst_w_id = 0;
  for ( ; dst_w_id < dst_w -2; ++dst_w_id) {
      int src_w_id_1 = dst_w_id / 2;
      Dtype* src_p11 = src_line1 + src_w_id_1 * channel_dim;
      Dtype* src_p12 = src_line2 + src_w_id_1 * channel_dim;
      InterpolateAction::template Do<channel_dim>(src_p11,
          src_p11 + channel_dim, src_p12, src_p12 + channel_dim,
          dst + dst_w_id * channel_dim,  128 * (dst_w_id % 2), wY);
  }
  for ( ; dst_w_id < dst_w; ++dst_w_id) {
    int src_w_id_1 = dst_w_id / 2;
    Dtype* src_p11 = src_line1 + src_w_id_1 * channel_dim;
    Dtype* src_p12 = src_line2 + src_w_id_1 * channel_dim;
    InterpolateAction::template Do< channel_dim>(src_p11,
        src_p11, src_p12, src_p12, dst + dst_w_id * channel_dim, 0, wY);
  }

}

template <typename InterpolateAction,typename Dtype, int channel_dim>
inline void upscale_2x_cpu(Dtype* src, const int src_h, const int src_w,
    Dtype* dst) {

  const int dst_h = src_h * 2;
  const int dst_w = src_w * 2;

  int dst_h_id = 0;
  for ( ; dst_h_id < dst_h - 2; ++dst_h_id) {
    Dtype* src_line1 = src + dst_h_id /2 * src_w * channel_dim;
    upscale_2x_line<InterpolateAction, Dtype, channel_dim>(
        src_line1, src_line1 + src_w * channel_dim,
        dst + dst_h_id * dst_w * channel_dim, dst_w, 128 * (dst_h_id % 2));
  }
  for ( ; dst_h_id < dst_h; ++dst_h_id) {
    Dtype* src_line1 = src + dst_h_id /2 * src_w * channel_dim;
    upscale_2x_line<InterpolateAction, Dtype, channel_dim>(
        src_line1, src_line1, dst + dst_h_id * dst_w * channel_dim, dst_w, 0);
  }

}

template <typename Dtype>
class Blob2xUpscaler {
 public:
  static void Forward_cpu(const Blob<Dtype>& src_blob, Blob<Dtype>& dst_blob);

 protected:
  static void Check(const Blob<Dtype>& src_blob, const Blob<Dtype>& dst_blob);

};


}  // namespace fxnet
}  // namespace hbot


#endif /* FXNET_UPSCALE_HPP_ */
