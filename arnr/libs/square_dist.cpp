#include <torch/extension.h>

torch::Tensor square_dist(torch::Tensor x1, torch::Tensor x2) {
    auto x1_norm = x1.pow(2).sum(-1, true);
    auto x1_pad = torch::ones_like(x1_norm);
    auto x2_norm = x2.pow(2).sum(-1, true);
    auto x2_pad = torch::ones_like(x2_norm);
    auto x1_ = torch::cat({x1.mul(-2), std::move(x1_norm), std::move(x1_pad)}, -1);
    auto x2_ = torch::cat({x2, std::move(x2_pad), std::move(x2_norm)}, -1);
    auto result = x1_.matmul(x2_.transpose(-2, -1));
    // Removed the clamp and sqrt operations
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square_dist", &square_dist, "Custom Square Distance");
}
