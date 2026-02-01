import torch

decive = "cuda" if torch.cuda.is_available() else "cpu"

layer = torch.nn.TransformerEncoderLayer(
    d_model=1024,
    nhead=8,
    activation=torch.nn.functional.gelu,
    batch_first=True,
    dim_feedforward=512,
    norm_first=True,
).cuda()

encoder = torch.nn.TransformerEncoder(layer, num_layers=12)

input = torch.rand((64, 128, 1024)).cuda()

import time

start = time.time()

encoder_output = encoder(input)

end = time.time() - start
print(f"End time: {end}")
print(f"Shape: {encoder_output.shape}")
print(f"Output diff: {encoder_output - input}")
