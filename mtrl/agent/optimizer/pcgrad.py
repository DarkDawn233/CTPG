import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGrad():
    """
    Source Implementation:
    https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/depreciate/pcgrad_ori.py
    """
    def __init__(self, optimizer):
        self._optim = optimizer
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''
        return self._optim.zero_grad()

    def step(self):
        '''
        update the parameters with the gradient
        '''
        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        input:
        - objectives: a list of objectives
        '''
        grads, shapes = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, shapes=None):
        num_task = len(grads)
        grads = torch.stack(grads, dim=0)
        id_lists = []
        for i in range(num_task):
            id_list = [_ for _ in range(num_task) if _ != i]
            random.shuffle(id_list)
            id_lists.append(torch.tensor(id_list).to(grads[0].device))
        id_lists = torch.stack(id_lists, dim=0)
        pc_grad = copy.deepcopy(grads)
        for i in range(num_task-1):
            g_j = grads[id_lists[:, i]]
            g_i_g_j = (pc_grad *  g_j).sum(dim=-1, keepdim=True)
            pc_grad = pc_grad - ((g_i_g_j < 0).float() * g_i_g_j *  g_j / ( g_j.norm(dim=-1, keepdim=True)**2))
        pc_grad = pc_grad.mean(dim=0)
        return pc_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.requires_grad is False: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        grads, shapes = [], []
        for obj in objectives:
            if torch.isclose(obj, torch.zeros_like(obj)):
                continue
            self._optim.zero_grad()
            obj.backward(retain_graph=True)
            grad, shape = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            shapes.append(shape)
            if obj is not objectives[0]:
                assert shape == shapes[0]
        return grads, shapes

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network.
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        '''
        grad, shape = [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if p.grad is None:
                        shape.append(p.shape)
                        grad.append(torch.zeros_like(p))
                    else:
                        shape.append(p.grad.shape)
                        grad.append(p.grad.clone())
        return grad, shape

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        return self._optim.load_state_dict(state_dict)


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    # seperated shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)