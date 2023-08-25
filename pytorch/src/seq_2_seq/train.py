# by y_dd
# 时间 2023/08/25
import torch
from torch import nn


def train(net, data_iter, lr, num_epochs, tgt_vocab, device):
    net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input)
            # Y_hat的形状(batch_size,num_steps,vocab_size)->permute后(batch_size,num_steps,vocab_size)
            # Y的形状batch_size,num_steps
            Y_hat = Y_hat.permute(0, 2, 1)
            l = loss(Y_hat, Y)
            l.sum().backward()	# 损失函数的标量进行“反向传播”
            #grad_clipping(net, 1)
            #num_tokens = Y_valid_len.sum()
            optimizer.step()
        print('epoch{}, loss{}'.format(epoch, l))
