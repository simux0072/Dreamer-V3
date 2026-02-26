import torch


class Dreamer_V3(torch.nn.Module):
    def __init__(
        self, symlog_model: torch.nn.Module, device: torch.device, batch_size: int = 16
    ) -> None:
        super().__init__()
        self.device = device
        self.symlog = symlog_model.to(device=self.device)
        self.batch_size = batch_size

        self.current_hidden = torch.zeros((self.batch_size, 1024))
        self.current_action = torch.zeros((self.batch_size, 1))
        self.current_latent = torch.zeros((self.batch_size, 1024))
        self.hidden_cell = torch.nn.GRUCell(
            input_size=1025, hidden_size=1024, device=self.device
        )

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                device=self.device,
            ),
            torch.nn.LayerNorm([32, 10, 10], device=self.device),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=1,
                device=self.device,
            ),
            torch.nn.LayerNorm([64, 6, 6], device=self.device),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=1,
                device=self.device,
            ),
            torch.nn.LayerNorm([128, 4, 4], device=self.device),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=1, device=self.device
            ),
            torch.nn.LayerNorm([256, 4, 4], device=self.device),
            torch.nn.SiLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=1, device=self.device
            ),
            torch.nn.LayerNorm([128, 4, 4], device=self.device),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=1,
                device=self.device,
            ),
            torch.nn.LayerNorm([64, 6, 6], device=self.device),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=2,
                stride=2,
                padding=1,
                device=self.device,
            ),
            torch.nn.LayerNorm([32, 10, 10], device=self.device),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
                device=self.device,
            ),
        )

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(in_features=5120, out_features=8192, device=self.device),
            torch.nn.SELU(),
            torch.nn.Linear(in_features=8192, out_features=4096, device=self.device),
            torch.nn.SELU(),
            torch.nn.Linear(in_features=4096, out_features=2048, device=self.device),
            torch.nn.SELU(),
            torch.nn.Linear(in_features=2048, out_features=1024, device=self.device),
        )

    def forward(self):
        pass

    def symlog_transform(self, input: torch.Tensor) -> torch.Tensor:
        transformed_input = self.symlog(input)
        return transformed_input

    def encoder(self, input_image: torch.Tensor):
        encoder_out: torch.Tensor = self.encoder(input_image)
        return encoder_out

    def hidden_pass(self):
        concat_hidden = torch.concat((self.current_latent, self.current_action))
        self.current_hidden = self.hidden_cell(concat_hidden, self.current_hidden)

    def latent_pass(self, encoder_out: torch.Tensor):
        input = torch.concat((encoder_out, self.current_hidden))
        self.current_latent = self.latent(input)
        self.current_dist = torch.reshape(
            self.current_latent, shape=(self.batch_size, 32, 32)
        )
