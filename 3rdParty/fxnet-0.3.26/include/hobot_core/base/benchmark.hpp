#ifndef HBOT_UTIL_BENCHMARK_H_
#define HBOT_UTIL_BENCHMARK_H_


#ifdef WINDOWS
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include <iostream>  // NOLINT

namespace hbot {

class Timer {
 public:
  Timer():initted_(false), running_(false), has_run_at_least_once_(false) {
    Init();
  }
  virtual ~Timer() {}
  virtual void Start() {
    if (!running()) {
#ifdef WINDOWS
      LARGE_INTEGER t;
      flag &= QueryPerformanceCounter(&t);
      if (!flag) {
        dStartTime = 0;
        return;
      }
      dStartTime = static_cast<double>(t.QuadPart);
  //   std::cout << "dStartTime: " << dStartTime << std::endl;
#else
      gettimeofday(&this->cpu_Start, NULL);
#endif
      running_ = true;
      has_run_at_least_once_ = true;
    }
  }
  virtual void Stop() {
    if (running()) {
#ifdef WINDOWS
      LARGE_INTEGER t;
      flag &= QueryPerformanceCounter(&t);
      if (!flag) {
        dStartTime = -1;
        return;
      }
      dStopTime = static_cast<double>(t.QuadPart);
#else
      gettimeofday(&this->cpu_Stop, NULL);
#endif
      running_ = false;
    }
  }
  virtual float MilliSeconds() {
    if (!has_run_at_least_once()) {
      std::cout << "Warning: Timer has never been run before reading time."
          << std::endl;
      return 0;
    }
    if (running()) {
      Stop();
    }
#ifdef WINDOWS
    elapsed_milliseconds_ = (dStopTime - dStartTime) * 1000 / dSystemFreq;
#else
    elapsed_milliseconds_ = static_cast<double>(this->cpu_Stop.tv_sec -
        cpu_Start.tv_sec)*1000.0 +
            static_cast<double>(cpu_Stop.tv_usec - cpu_Start.tv_usec)/1000.0;
#endif
    return elapsed_milliseconds_;
  }
  virtual float MicroSeconds() {
    if (!has_run_at_least_once()) {
      std::cout << "Warning: Timer has never been run before reading time."
          << std::endl;
      return 0;
    }
    if (running()) {
      Stop();
    }
#ifdef WINDOWS
    elapsed_microseconds_ = (dStopTime - dStartTime) * 1000000 / dSystemFreq;
#else
    elapsed_microseconds_ =  static_cast<double>(this->cpu_Stop.tv_sec -
        cpu_Start.tv_sec)*1000000.0 + static_cast<double>(cpu_Stop.tv_usec
            - cpu_Start.tv_usec);
#endif
    return elapsed_microseconds_;
  }
  virtual float Seconds() {return MilliSeconds() / 1000.0;}

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init() {
    if (!initted()) {
      initted_ = true;
    }
#ifdef WINDOWS
    LARGE_INTEGER sysFreq;
    flag = QueryPerformanceFrequency(&sysFreq);
    if (flag) {
      dSystemFreq = static_cast<double>(sysFreq.QuadPart);
    }
#endif
  }

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;


#ifdef WINDOWS
  double dSystemFreq;
  bool flag;
  double dStartTime;
  double dStopTime;
#else
  timeval cpu_Start;
  timeval cpu_Stop;
#endif

  float elapsed_milliseconds_;
  float elapsed_microseconds_;
};

class CPUTimer : public Timer {
 public:
  explicit CPUTimer() {
    this->initted_ = true;
    this->running_ = false;
    this->has_run_at_least_once_ = false;
  }
  virtual ~CPUTimer() {}
};

}  // namespace hbot

#endif   // HBOT_UTIL_BENCHMARK_H_
