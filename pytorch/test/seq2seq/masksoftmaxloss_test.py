import torch

from pytorch.src.seq_2_seq.masksoftmax import MaskedSoftmaxCELoss

mask_softmax = MaskedSoftmaxCELoss()

X = torch.tensor([[[1, 2, 3, 0, 0],
      [5, 2, 0, 0, 0],
      [1, 0, 0, 0, 0]],
      [[5, 2, 7, 9, 9],
      [1, 0, 0, 0, 0],
      [5, 2, 7, 9, 9]]])

valid_len = [[[3],
      [2],
      [1]],
      [[5],
      [1],
      [5]]]

#print(X.shape, X.size(0), X.size(1), X.size(2))
maxlen = X.size(1)
print(maxlen)

loss = MaskedSoftmaxCELoss()
# a = torch.ones(3, 4, 10)
# b = torch.ones(3, 4, dtype=torch.long)
# l = loss(a, b,  torch.tensor([4, 2, 0]))

l = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
print(l)
