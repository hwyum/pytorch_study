import torch
import torch.nn

## shape test
a = torch.tensor([[[1,2,3],[3,4,5]],
                  [[5,6,7],[7,8,9]]])
b = torch.tensor([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

a @ b  # 2 x 2 matrix multiplication (using last two dimensions)

c = torch.tensor([[1,2,3],
                  [1,2,3]])
d = torch.tensor([[2,3,4],
                  [3,4,5]])
c * d # element-wise product