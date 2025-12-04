import torch

gru = torch.nn.GRU(input_size=2049, hidden_size=1024, num_layers=16)
input = torch.randn((3, 2, 2049))

h0 = torch.zeros((3, 2, 1024))

output, hn = gru(input)
print(output.shape)
print(hn.shape)
print((output[-1] - hn[-1]).sum())
