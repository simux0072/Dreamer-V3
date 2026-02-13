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
            torch.nn.Linear(8, 1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.size(-1) != 1:
            initial_shape = input.shape
            transformed_input = input.flatten(start_dim=1).unsqueeze(dim=-1)
            prediction = self.symlog(transformed_input).reshape(initial_shape)
            return prediction
        prediction = self.symlog(input)
        return prediction
