//
// Created by adzie on 05-Oct-18.
//

#include <torch/torch.h>
#include <iostream>

int main() {
  at::Tensor tensor = torch::rand({2, 3});
  std::cout << "tensor: " << tensor << std::endl;
  at::Tensor tensor2 = at::tensor({8.2});
  std::cout << "tensor2: " << tensor2 << std::endl;
}

