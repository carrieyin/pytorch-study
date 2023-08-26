# by y_dd
# 时间 2023/08/25
import torch
from torch import nn

from pytorch.src.seq_2_seq.masksoftmax import MaskedSoftmaxCELoss


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train(net, data_iter, lr, num_epochs, tgt_vocab, device):
    net.to(device)
    loss = MaskedSoftmaxCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        num_tokens = 0
        total_loss = 0
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input)
            # Y_hat的形状(batch_size,num_steps,vocab_size)
            # Y的形状batch_size,num_steps
            # loss内部permute Y_hat = Y_hat.permute(0, 2, 1)
            l = loss(Y_hat, Y, Y_valid_len)
            # 损失函数的标量进行“反向传播”
            l.sum().backward()
            #梯度裁剪
            grad_clipping(net, 1)
            #梯度更新
            optimizer.step()

            num_tokens = Y_valid_len.sum()
            total_loss = l.sum()

        print('epoch{}, loss{:.3f}'.format(epoch, total_loss/num_tokens))
