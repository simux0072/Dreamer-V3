import torch

class SymLog(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.symlog = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.SELU(),
            torch.nn.Linear(32, 16),
            torch.nn.SELU(),
            torch.nn.Linear(16, 8),
            torch.nn.SELU(),
            torch.nn.Linear(8, 1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        prediction = self.symlog(input)
        return prediction

