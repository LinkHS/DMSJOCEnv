/*
 * blob_transform.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef HBOT_BASE_TRANSFORM_EXPRESSION_HPP_
#define HBOT_BASE_TRANSFORM_EXPRESSION_HPP_

#include "hobot_core/base/base_common.hpp"


namespace hbot {

template<typename Dtype>
struct Expression{
  inline Dtype& self(void) {
    return *static_cast<Dtype*>(this);
  }
  inline Dtype* ptrself(void) {
    return static_cast<Dtype*>(this);
  }
};

template<typename OP, typename Tsrc>
struct UnaryOpExpression : public Expression<UnaryOpExpression<OP, Tsrc> > {
  Tsrc & src_var_;
  explicit UnaryOpExpression(Tsrc& src_var): src_var_(src_var) {}  // NOLINT
};

template<typename OP, typename Tleft, typename Tright>
struct BinaryOpExpression : public
    Expression<BinaryOpExpression<OP, Tleft, Tright> > {
  Tleft & l_var_;
  Tright & r_var_;
  explicit BinaryOpExpression(Tleft& l_var,  // NOLINT
       Tright& r_var):l_var_(l_var), r_var_(r_var) {}   // NOLINT
};

template<typename OP, typename TA, typename TB, typename TC>
struct TripleOpExpression : public
    Expression<TripleOpExpression< OP, TA, TB, TC> > {
  TA & a_var_;
  TB & b_var_;
  TC & c_var_;
  explicit TripleOpExpression(TA& a_var,    // NOLINT
       TB& b_var, TC& c_var):a_var_(a_var), b_var_(b_var), c_var_(c_var) {}   // NOLINT
};


template<typename OPType>
class Trans {
 public:
  HBOT_XINLINE void Do();
};

template< typename OP, typename Tsrc>
class Trans<UnaryOpExpression<OP, Tsrc> > {
 public:
  explicit Trans(const UnaryOpExpression<OP, Tsrc>& src): src_(src) {
  }
  HBOT_XINLINE void Do() {
    OP::Do(src_.src_var_);
  }

 protected:
  UnaryOpExpression<OP, Tsrc> src_;
};

template<typename OP, typename Tleft, typename Tright>
class Trans<BinaryOpExpression<OP, Tleft, Tright> > {
 public:
  explicit Trans(const BinaryOpExpression<OP, Tleft, Tright>& src)
    :src_(src) {}
  HBOT_XINLINE void Do() {
    OP::Do(src_.l_var_, src_.r_var_);
  }

 protected:
  BinaryOpExpression<OP, Tleft, Tright> src_;
};

template<typename OP, typename TA, typename TB, typename TC>
class Trans<TripleOpExpression<OP, TA, TB, TC> >{
 public:
  explicit Trans(const TripleOpExpression<OP, TA, TB, TC>& src): src_(src) {}
  HBOT_XINLINE void Do() {
    OP::Do(src_.a_var_, src_.b_var_, src_.c_var_);
  }
 protected:
  TripleOpExpression<OP, TA, TB, TC> src_;
};

template<typename OP, typename TA, typename TB, typename TC>
inline void hbot_transform(TA & a, TB  & b, TC & c) {  // NOLINT
  Trans<TripleOpExpression<OP, TA, TB, TC> > temp(
      TripleOpExpression<OP, TA, TB, TC>(a, b, c));
  temp.Do();
}

template<typename OP, typename TA, typename TB>
inline void hbot_transform(TA& a, TB& b) {  // NOLINT
  Trans<BinaryOpExpression<OP, TA, TB > > temp(
      BinaryOpExpression<OP, TA, TB >(a, b));
  temp.Do();
}


template<typename OP, typename TA >
inline void hbot_transform(TA& a) {  // NOLINT
  Trans<UnaryOpExpression<OP, TA > > temp((UnaryOpExpression<OP, TA >(a)));
  temp.Do();
}


}  // namespace hbot


#endif /* HBOT_BASE_TRANSFORM_EXPRESSION_HPP_ */
