import torch
from symlog.symlog_net import SymLog
from network import Dreamer_V3

import unittest

class TestNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.symlog = SymLog()
        self.symlog.load_state_dict(torch.load('symlog/model/symlog.pt', weights_only=True))
        self.symlog.eval()
        self.dreamer = Dreamer_V3(self.symlog)

    def test_symlog(self) -> None:
        input = torch.rand((30, 20, 20)).flatten(start_dim=1).unsqueeze(dim=-1)
        output: torch.Tensor = self.symlog(input).reshape(, 20, 20)
        print(f"Symlog output shape: {output.shape}")
        self.assertTrue(output.shape == (1, 20, 20))



