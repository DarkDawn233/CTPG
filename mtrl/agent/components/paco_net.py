import torch
from torch import nn
from torch.nn import functional as F

from mtrl.agent.components.base import Component as BaseComponent
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.utils.types import TensorType

class CompositionalFC(nn.Module):
    def __init__(
            self,
            input_shape: int,
            output_shape: int,
            n: int,
            activation: bool = True,
        ):
        
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n = n

        self.weight = nn.Parameter(torch.rand(n, input_shape, output_shape))
        self.bias = nn.Parameter(torch.rand(n, output_shape))
        self.activation = activation

    def forward(self, x, comp_weight):
        x = x.unsqueeze(0).expand(self.n, *x.shape)
        z = torch.baddbmm(self.bias.unsqueeze(1), x, self.weight)
        z = z.transpose(0, 1)

        z = torch.bmm(comp_weight.unsqueeze(1), z).squeeze(1)
        if self.activation:
            z = F.relu(z)

        return z

class PaCoNet(BaseComponent):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            num_of_param_set: int,
            num_layers: int
        ):
        super().__init__()
        self.num_of_param_set = num_of_param_set

        self.layers = nn.ModuleList()
        current_in_features = in_features
        for _ in range(num_layers):
            fc = CompositionalFC(current_in_features, hidden_features, num_of_param_set, True)
            self.layers.append(fc)  
            current_in_features = hidden_features
        self.layers.append(CompositionalFC(current_in_features, out_features, num_of_param_set, False))

    def forward(self, mtobs: MTObs) -> TensorType:
        out = mtobs.env_obs
        task_encode = mtobs.task_info.encoding
        for comp_fc in self.layers:
            out = comp_fc(out, task_encode)
        return out
