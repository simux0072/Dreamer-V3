import torch
from symlog_net import SymLog

BATCH_SIZE = 4096
GENERATION_SIZE = 1_000_000
LOW_BOUND: int = -100_000
HIGH_BOUND: int = 100_000
LR: float = 0.001

TEST_BATCH_SIZE = 1_000_000
TEST_INTERVAL = 1_000

symlog_net = SymLog()
optimizer = torch.optim.AdamW(symlog_net.parameters(), lr=LR)

def test(symlog_net: SymLog) -> None:
    input = torch.rand((TEST_BATCH_SIZE, 1)) + torch.randint(LOW_BOUND, HIGH_BOUND, size=[TEST_BATCH_SIZE, 1])
    true_values = torch.sign(input) * torch.log(torch.abs(input) + 1)
    predicted_values = symlog_net(input)

    loss = torch.mean(((true_values - predicted_values) ** 2)) / 2
    print(f"Test Loss: {loss}")

for generation in range(GENERATION_SIZE):
    input = torch.rand((BATCH_SIZE, 1)) + torch.randint(LOW_BOUND, HIGH_BOUND, size=[BATCH_SIZE, 1])
    true_values = torch.sign(input) * torch.log(torch.abs(input) + 1)
    predicted_values = symlog_net(input)

    loss = torch.mean(((true_values - predicted_values) ** 2)) / 2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if generation % TEST_INTERVAL == 0:
        test(symlog_net)
        print(f"Generation: {generation}")
    else:
        print(f"Generation: {generation}", end='\r')
