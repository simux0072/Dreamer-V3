import torch
from symlog_net import SymLog
from tqdm import tqdm

import wandb


def test(symlog_net: SymLog, run: wandb.Run) -> torch.Tensor:
    input = (
        torch.rand((run.config["test_batch_size"], *run.config["input_shape"]))
        + torch.randint(
            run.config["low_bound"],
            run.config["high_bound"],
            size=[run.config["test_batch_size"], *run.config["input_shape"]],
        )
    ).to(run.config["device"])
    true_values = torch.sign(input) * torch.log(torch.abs(input) + 1)
    predicted_values = symlog_net(input)

    loss = torch.mean(((true_values - predicted_values) ** 2)) / 2
    return loss


def train(wandb_run: wandb.Run, symlog_net: SymLog, optimizer: torch.optim.Optimizer):
    for generation in tqdm(range(wandb_run.config["generation_size"]), desc="Training"):
        input = (
            torch.rand(
                (wandb_run.config["batch_size"], *wandb_run.config["input_shape"])
            )
            + torch.randint(
                wandb_run.config["low_bound"],
                wandb_run.config["high_bound"],
                size=[wandb_run.config["batch_size"], *wandb_run.config["input_shape"]],
            )
        ).to(wandb_run.config["device"])
        true_values = torch.sign(input) * torch.log(torch.abs(input) + 1)
        predicted_values = symlog_net(input)

        loss = torch.mean(((true_values - predicted_values) ** 2)) / 2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        run.log({"loss": loss, "generation": generation})
        if generation % config["test_generation_interval"] == 0:
            test_loss = test(symlog_net, wandb_run)
            run.log({"test loss": test_loss, "generation": generation})

    save_model(symlog_net, wandb_run.config["save_path"], wandb_run)


def save_model(model: torch.nn.Module, path: str, run: wandb.Run):
    torch.save(model.state_dict(), path)
    run.log_artifact(path, name="Symlog", type="model")


if __name__ == "__main__":
    project_name = "Dreamer-V3"
    path = "./symlog/model/symlog.pt"
    config = {
        "lr": 0.001,
        "batch_size": 1024,
        "generation_size": 250_000,
        "low_bound": -1_000,
        "high_bound": 1_000,
        "test_batch_size": 40_000,
        "test_generation_interval": 1_000,
        "save_path": path,
        "input_shape": (20, 20),
        "wandb_mode": "online",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    with wandb.init(
        project=project_name, config=config, mode=config["wandb_mode"]
    ) as run:
        symlog_net = SymLog().to(run.config["device"])
        optimizer = torch.optim.AdamW(symlog_net.parameters(), lr=run.config["lr"])
        train(run, symlog_net, optimizer)
