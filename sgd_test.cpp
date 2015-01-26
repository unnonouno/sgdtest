#include <sys/time.h>

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>

void sgd(
    float eta,
    const std::vector<float>& g,
    std::vector<float>& x) {
  for (std::size_t i = 0, size = x.size(); i < size; ++i) {
    x[i] += eta * g[i];
  }
}

void ada_grad(
    float eta,
    const std::vector<float>& g,
    std::vector<float>& sum,
    std::vector<float>& x) {
  for (std::size_t i = 0, size = x.size(); i < size; ++i) {
    sum[i] += g[i] * g[i];
    x[i] += eta / std::sqrt(sum[i]) * g[i];
  }
}

void adam(
    float eta,
    float& lambda_t,
    float& beta1_t,
    float& beta2_t,
    const std::vector<float>& g,
    std::vector<float>& m1,
    std::vector<float>& m2,
    std::vector<float>& x) {
  float beta1_ = 0.1;
  float beta2_ = 0.001;
  float decay_ = 1.0e-8;

  float b1 = 1 - (1 - beta1_) * lambda_t;
  float k1 = 1.0 / (1.0 - beta1_t);
  float k2_sqrt = std::sqrt(1.0 / (1.0 - beta2_t));
  float eta_k1_k2 = eta * k1 / k2_sqrt;
  for (std::size_t i = 0, size = g.size(); i < size; ++i) {
    m1[i] = b1 * g[i] + (1 - b1) * m1[i];
  }
  float b2 = (1 - beta2_);
  for (std::size_t i = 0, size = g.size(); i < size; ++i) {
    m2[i] = beta2_ * g[i] * g[i] + b2 * m2[i];
  }
  for (std::size_t i = 0, size = g.size(); i < size; ++i) {
    x[i] -= eta_k1_k2 * m1[i] / std::sqrt(m2[i]);
  }

  lambda_t *= decay_;
  beta1_t *= 1 - beta1_;
  beta2_t *= 1 - beta2_;
}

struct timer {
  timer() {
    ::gettimeofday(&begin, NULL);
  }

  ~timer() {
    ::gettimeofday(&end, NULL);
    double elapsed = 1000.0 * (end.tv_sec - begin.tv_sec)
        + (end.tv_usec - begin.tv_usec) / 1000.0;
    std::cout << elapsed << " msec" << std::endl;
  }

  timeval begin, end;
};

int main() {
  std::size_t dimension = 1000;
  float learning_rate = 0.1;
  std::size_t iteration = 1000000;

  srand(0);
  std::vector<float> g;
  for (size_t i = 0; i < dimension; ++i) {
    g.push_back(rand());
  }


  {
    std::cout << "SGD start" << std::endl;
    std::vector<float> r(dimension);
    timer t;
    for (std::size_t i = 0; i < iteration; ++i) {
      sgd(learning_rate, g, r);
    }
    std::cout << r[0] << std::endl;
  }

  {
    std::cout << "Ada-grad start" << std::endl;
    std::vector<float> r(dimension);
    std::vector<float> sum(dimension, 1.0e-10f);
    timer t;
    for (std::size_t i = 0; i < iteration; ++i) {
      ada_grad(learning_rate, g, sum, r);
    }
    std::cout << r[0] << std::endl;
  }

  {
    std::cout << "Adam start" << std::endl;
    std::vector<float> r(dimension);
    std::vector<float> m1(dimension, 1.0e-10f);
    std::vector<float> m2(dimension, 1.0e-10f);
    float lambda_t = 1.0, beta1_t = 1.0, beta2_t = 1.0;

    timer t;
    for (std::size_t i = 0; i < iteration; ++i) {
      adam(learning_rate, lambda_t, beta1_t, beta2_t, g, m1, m2, r);
    }
    std::cout << r[0] << std::endl;
  }
}
