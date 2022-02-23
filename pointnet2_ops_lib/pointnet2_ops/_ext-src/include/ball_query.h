#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);
