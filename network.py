import torch

class Dreamer_V3(torch.nn.Module):
    def __init__(self, symlog_model: torch.nn.Module) -> None:
        super().__init__()
        self.symlog = symlog_model
        #Initial image input size: 20x20
        # Use CNNs for parsing the image into latent space
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            torch.nn.LayerNorm([32, 10, 10]),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=1),
            torch.nn.LayerNorm([64, 6, 6]),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=1),
            torch.nn.LayerNorm([128, 4, 4]),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            torch.nn.LayerNorm([256, 4, 4]),
            torch.nn.SiLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=1),
            torch.nn.LayerNorm([128, 4, 4]),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=1),
            torch.nn.LayerNorm([64, 6, 6]),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=1),
            torch.nn.LayerNorm([32, 10, 10]),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self):
        pass
    def symlog_transform(self, input: torch.Tensor) -> torch.Tensor:
        transformed_input = self.symlog(input)
        return transformed_input

    def test_encoder(self, input: torch.Tensor) -> torch.Tensor:
        transformed_input = self.symlog_transform(input)
        output = self.encoder(transformed_input)
        return output
    
    def test_decoder(self, input: torch.Tensor) -> torch.Tensor:
        symlog_input = self.symlog(input)
        output = self.decoder(symlog_input)
        return output
