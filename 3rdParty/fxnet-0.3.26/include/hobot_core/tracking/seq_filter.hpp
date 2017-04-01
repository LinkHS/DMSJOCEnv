#ifndef HBOT_TRACKING_SEQ_FILTER
#define HBOT_TRACKING_SEQ_FILTER

#include <vector>
#include <string>
#include <queue>
#include <iostream>
#include <algorithm>
#include <list>
#include <random>
#include "hobot_core/base/base_common.hpp"
#include "hobot_core/math/math_functions.hpp"

//#define USE_OPENCV


#if defined(USE_OPENCV)
#include "opencv2/video/tracking.hpp"
#endif


namespace hbot{
namespace tracking{


template <typename Particle>
class BaseSeqFilter{
public:
	BaseSeqFilter(){};
	virtual ~BaseSeqFilter(){};
	virtual Particle Predict() =0;
	virtual void Correct(Particle& measurementAndState) =0;
};


class Simple2DParticle{
public:
	Simple2DParticle():weight(0),x(0),y(0), history_len(0){ };
	Simple2DParticle(int max_hist_len, float x_ = 0, float y_ = 0):weight(0){
		x = x_; y=y_;
		history_len = max_hist_len;
		history_x_y.clear();
		for(int i=0; i < max_hist_len; ++i){
			history_x_y.push_back(x_);
			history_x_y.push_back(y_);
		}
	}
	~Simple2DParticle(){};


	void GetHistory(std::vector<float>& hist){
		hist.clear();
		for(std::list<float>::iterator history_iter = history_x_y.begin();
				history_iter != history_x_y.end(); ){
			hist.push_back((*history_iter++));
		}

	}

	friend std::ostream& operator << (std::ostream& out, Simple2DParticle& particle ){
		out<<"( x:"<<particle.x<<" , y:"<<particle.y<< " , weight: "<<particle.weight;

		int history_id=0;
		out<<" history size: "<< particle.history_len;
		for(std::list<float>::iterator history_iter = particle.history_x_y.begin();
				history_iter != particle.history_x_y.end(); ){
			out<<" history "<<history_id<<" :{ x:"<<(*history_iter++)<<" y:"<<(*history_iter++)<<" }";
		}

		out<<" ).";
		return out;
	}



	float weight;
	/**
	 * @bref This function is a interface for particle update( particle proposals).
	 * 			 In this case, std is a 2x1 vector, for std of x and y,
	 * 			 and the size of autoregress_weight is equal to the size of history.
	 */
	void UpdateStates(std::vector<float>& std, std::vector<float>& autoregress_weight);
	/**
	 * @bref This function is a interface for particle to re-weight from measurement.
	 *			 The re_weight_param is a 1x3vector, which denote the variances
	 *			 and bias of bivariate gaussian distribution  along x and y axises (uncertainty of measurement).
	 */
	void ReWeight(Simple2DParticle& measurement, std::vector<float>& re_weight_param);

	/**
	 * @bref This function is a interface for particle to get the expectation of all particles
	 */
	static Simple2DParticle GetExpectation(std::vector<Simple2DParticle> & vec_particle);

	/**
	 * @bref This function is a interface for particle to Update history.
	 */
	void UpdateHistory(Simple2DParticle & previous){
		this->push_history_x_y(previous.x, previous.y);
	}

	float x, y;  // measurement , and state
	std::list<float> history_x_y;
	int history_len;
	static std::random_device rd;
	static std::mt19937 seed;


	inline void push_history_x_y(float _x, float _y){
		history_x_y.pop_front();  history_x_y.pop_front();
		history_x_y.push_back(_x); history_x_y.push_back(_y);
	}

	static inline float gaussian_likelihood(float x, float mean, float std, float bias){
		return std::exp((x - mean)*(x - mean)/(-2*std*std))/std/(std::sqrt(2*HBOT_PI)) + bias;
//		return std::exp((x - mean)*(x - mean)/(-2*std*std)) + bias;
	}

	static inline float independent_bivariate_gaussian_likelihood(float x, float y,
			float mean_a, float mean_b, float std_a, float std_b, float bias){
		return std::exp( ( (x-mean_a)*(x-mean_a)/(std_a*std_a) +
				(y-mean_b)*(y-mean_b)/(std_b*std_b) ) * -0.5f )/(std_a*std_b*2*HBOT_PI) + bias;
//		return std::exp( ( (x-mean_a)*(x-mean_a)/(std_a*std_a) +
//				(y-mean_b)*(y-mean_b)/(std_b*std_b) ) * -0.5f ) + bias;
	}

	static inline void rng_gaussian(const int n,const float mean, const float sigma,
			float* data){
		std::normal_distribution<float> dist(mean, sigma);
	  for (int i = 0; i < n; ++i) {
	  	data[i] = dist(seed);
	  }
	}
	static inline float rng_gaussian(const float mean, const float sigma ){
		std::normal_distribution<float> dist(mean, sigma);
	  return dist(seed);
	}
	static inline float rng_uniform(const float a, const float b){
		std::uniform_real_distribution<float> dist(a,b);
		return dist(seed);
	}
	static inline float random_float(){
		return rng_uniform(0,1);
	}

};

std::random_device Simple2DParticle::rd  ;

std::mt19937 Simple2DParticle::seed = std::mt19937(Simple2DParticle::rd());

template <typename Particle>
class ParticleFilter : public BaseSeqFilter<Particle> {
public:
	ParticleFilter(): BaseSeqFilter<Particle>(),particle_set_idx(0){ };
	virtual ~ParticleFilter(){};
	void SetInitParticles(Particle particle, int n);
	virtual Particle Predict() {
		ReSample();
		UpdateParticles();
		Particle pred = Particle::GetExpectation(particle_vec_set[particle_set_idx]);
//		UpdateParticles();
		UpdateHistory(pred);
		return pred;
	}
	virtual void Correct(Particle& measurementAndState){
		ReWeight(measurementAndState);
	}

	inline void SetUpdateStd(std::vector<float>& std){
		update_std_.clear();
		std::copy(std.begin(), std.end(), std::back_inserter(update_std_));
	}

	inline std::vector<Particle> & GetAllParticle(){
		return particle_vec_set[particle_set_idx];
	}
	inline void SetAutoRegressWeight(std::vector<float>& weight){
		autoregress_weight_.clear();
		std::copy(weight.begin(), weight.end(), std::back_inserter(autoregress_weight_));
	}
	inline void SetReWeightParam(std::vector<float> re_weight_param){
		re_weight_param_.clear();
		std::copy(re_weight_param.begin(), re_weight_param.end(),
				std::back_inserter(re_weight_param_));
	}
protected:

	/**
	 * @brief Update particles according to update_variance_ and autoregress_weight_;
	 */
	void UpdateParticles();
	void ReWeight(Particle& measurementAndState);
	void ReSample();

	void UpdateHistory(Particle& measurementAndState);

	std::vector<Particle> particle_vec_set[2];
	std::vector<float> update_std_;
	std::vector<float> autoregress_weight_;
	std::vector<float> re_weight_param_;
	int particle_set_idx;

};



#if defined(USE_OPENCV)

/**SYSTEM DYNAMICS:
%
% The system evolves according to the following difference equations,
% where quantities are further defined below:
%
% x = Ax + Bu + w  Mx1, meaning the state vector x evolves during one time
%                  step by premultiplying by the "state transition
%                  matrix" A. There is optionally (if nonzero) an input
%                  vector u which affects the state linearly, and this
%                  linear effect on the state is represented by
%                  premultiplying by the "input matrix" B. There is also
%                  gaussian process noise w.
% z = Hx + v       Nx1, meaning the observation vector z is a linear function
%                  of the state vector, and this linear relationship is
%                  represented by premultiplication by "observation
%                  matrix" H. There is also gaussian measurement
%                  noise v.
% where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
%       v ~ N(0,R) meaning v is gaussian noise with covariance R
*/
template <typename Mat>
class FxnetKalmanFilter : public BaseSeqFilter<Mat> {
public:
	FxnetKalmanFilter(): BaseSeqFilter<Mat>(),kf_ptr_(NULL), state_num_(0),
	measure_num_(0),control_num_(0){ };
	virtual ~FxnetKalmanFilter(){if(kf_ptr_) delete kf_ptr_;}
	void SetKalmanFilter(int state_n, int measure_n, int control_n);

	/**
	 * x = Ax + Bu + w   MxM matrix.
	 */
	void SetTransitionMatrix(cv::Mat& A);

	/**
	 * x = Ax + Bu + w   MxK matrix.
	 */
	void SetControlMatrix(cv::Mat& B);

	/**
	 * x = Ax + Bu + w . MxM matrix
	 * w ~ N(0,Q) meaning w is gaussian noise with covariance Q.
	 * This matrix tell Kalman filter how much error is in each action from time.
	 */
	void SetProcessNoiseCov(cv::Mat& Q);

	/**
	 *  z = Hx + v  .   NxM matrix. The model of sensor.
	 */
	void SetMeaturementMatrix(cv::Mat& H);

	/**
	 *  z = Hx + v  .   NxN matrix.
 	 *  v ~ N(0,R) meaning v is gaussian noise with covariance R
	 */
	void SetMeasurementNoiseCov(cv::Mat& R);

	/**
	 * MxM matrix priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q).
	 * The best way is to initialize it as a diagonal matrix. It would converge after
	 * running several times.It could effect the initial several frames .
	 */
	void SetErrorCovPre(cv::Mat& Ppre);

	/**
	 * MxM matrix posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
	 * The best way is to initialize it as a diagonal matrix. It would converge after
	 * running several times.It could effect the initial several frames .
	 */
	void SetErrorCovPost(cv::Mat& Ppost);

	/**
	 * x = Ax + Bu + w   Kx1 matrix.
	 */
	void SetControlU(cv::Mat& u);

	// predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k).  type()=CV_32FC1
	inline cv::Mat& StatePre(){
		return kf_ptr_->statePre;
	}

	// corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))  type()=CV_32FC1
	inline cv::Mat& StatePost(){
		return kf_ptr_->statePost;
	}

	virtual Mat Predict(){
		return kf_ptr_->predict(control_u_);
	}
	virtual void Correct(Mat& measurementAndState){
		kf_ptr_->correct(measurementAndState);
	}
	inline int state_num(){ return state_num_; }
	inline int measure_num(){ return measure_num_; }
	inline int control_num(){ return control_num_; }
protected:
	cv::KalmanFilter* kf_ptr_;
	int state_num_;  // M
	int measure_num_; // N
	int control_num_;  // K
	cv::Mat control_u_;
};


#endif






} // namespace tracking
} // namespace hbot



#endif //HBOT_TRACKING_SEQ_FILTER
