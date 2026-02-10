import torch
from torch.serialization import save
from symlog_net import SymLog
import wandb
from tqdm import tqdm

def test(symlog_net: SymLog, run: wandb.Run) -> torch.Tensor:
    input = torch.rand((run.config['test_batch_size'], 1)) + torch.randint(run.config['low_bound'], run.config['high_bound'], size=[run.config['test_batch_size'], 1])
    true_values = torch.sign(input) * torch.log(torch.abs(input) + 1)
    predicted_values = symlog_net(input)

    loss = torch.mean(((true_values - predicted_values) ** 2)) / 2
    return loss
def train(wandb_run: wandb.Run, symlog_net: SymLog, optimizer: torch.optim.Optimizer):
    for generation in tqdm(range(wandb_run.config['generation_size']), desc="Training"):
        input = torch.rand((wandb_run.config['batch_size'], 1)) + torch.randint(wandb_run.config['low_bound'], wandb_run.config['high_bound'], size=[wandb_run.config['batch_size'], 1])
        true_values = torch.sign(input) * torch.log(torch.abs(input) + 1)
        predicted_values = symlog_net(input)

        loss = torch.mean(((true_values - predicted_values) ** 2)) / 2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        run.log({"loss": loss, "generation": generation})
        if generation % config['test_generation_interval'] == 0:
            test_loss = test(symlog_net, wandb_run)
            run.log({"test loss": test_loss, "generation": generation})

    save_model(symlog_net, wandb_run.config['save_path'])

def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    project_name = "Dreamer-V3"
    path = './model/symlog.pt'
    config = {
        'lr': 0.001,
        'batch_size': 4096,
        'generation_size': 1_000_000,
        'low_bound': -100_000,
        'high_bound': 100_000,
        'test_batch_size': 1_000_000,
        'test_generation_interval': 1_000,
        'save_path': path,
    }

    with wandb.init(project=project_name, config=config, mode="online") as run:
        symlog_net = SymLog()
        optimizer = torch.optim.AdamW(symlog_net.parameters(), lr=run.config['lr'])
        train(run, symlog_net, optimizer)
