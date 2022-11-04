import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

from typing import Union

# Optimizer wrappers for dealing with optimization, learning rate annealing, and gradient clipping at the same time

# Hyperparameter format must be as below
# "actor_optimizer": {"optimizer": ["Adam", {}], "lr_scheduler": ["Empty", {}], "gradient_clipping": []}

class EmptyScheduler:
    def __init__(self, *args):
        pass
    def step(self):
        return

def get_parameters(network: Union[nn.Module, torch.Tensor, nn.Parameter, list, tuple]):
    if isinstance(network, nn.Module):
        return network.parameters()
    elif isinstance(network, torch.Tensor):
        return [network]
    elif isinstance(network, nn.Parameter):
        return network
    else:
        parameters = []
        for n in network:
            parameters += list(n.parameters())

        return parameters

class Optimizer:
    def __init__(self, network: Union[nn.Module, torch.Tensor, nn.Parameter, list, tuple], hyperparameters):

        self.parameters = get_parameters(network)

        # Optimizer
        optimizer, optimizer_hyperparameter = hyperparameters['optimizer']

        self.optimizer_name = optimizer

        learning_rate = optimizer_hyperparameter['learning_rate']

        if 'weight_decay' in optimizer_hyperparameter.keys():
            weight_decay = optimizer_hyperparameter['weight_decay']
        else:
            weight_decay = 0


        if isinstance(learning_rate, float):
            self.optimizer = getattr(optim, optimizer)(self.parameters, lr=learning_rate, weight_decay=weight_decay)
        elif isinstance(learning_rate, list):
            self.optimizer = getattr(optim, optimizer)([{'params': get_parameters(n),
                                                         'lr': learning_rate[i],
                                                         'weight_decay': weight_decay,
                                                         } for i, n in enumerate(network)])


        # Learning rate scheduler
        scheduler, scheduler_hyperparameter = hyperparameters['lr_scheduler']

        if scheduler == 'Lambda':

            lr_lambda = scheduler_hyperparameter['lr_lambda']
            last_epochs = scheduler_hyperparameter['last_epochs']
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda, last_epochs)
        elif scheduler == 'CosineAnnealing':
            t_max = scheduler_hyperparameter['t_max']
            self.lr_scheduler = CosineAnnealingLR(self.optimizer, t_max)
        elif scheduler == 'CosineAnnealingWarmup':
            t_0 = scheduler_hyperparameter['t_0']
            self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, t_0)
        elif scheduler == 'Plateau':
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer)
        else:
            self.lr_scheduler = EmptyScheduler(self.optimizer)

        # Gradient clipping
        self.clip_gradient = 'gradient_clipping' in hyperparameters.keys()
        if self.clip_gradient:
            self.gradient_clipping_hyperparameters = hyperparameters['gradient_clipping']

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradient:
            nn.utils.clip_grad_norm(self.parameters, *self.gradient_clipping_hyperparameters)
        self.optimizer.step()

        self.lr_scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

class SharedOptimizer(Optimizer):
    def __init__(self,
                 network,

                 hyperparameters):
        self.network = network

        super(SharedOptimizer, self).__init__(network,

                                              hyperparameters)

        # State initialization
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                state['step'] = torch.zeros(1)

                if 'Adam' in self.optimizer_name:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    if self.optimizer_name == 'NAdam':
                        state['mu_product'] = torch.ones(1)
                        state['mu_product'].share_memory_()
                    elif self.optimizer_name == 'Adamax':
                        state['exp_inf'] = torch.zeros_like(p.data)
                        state['exp_inf'].share_memory_()
                    state['exp_avg'].share_memory_()
                    state['exp_avg_sq'].share_memory_()

                elif self.optimizer_name == 'RMSprop':
                    state['square_avg'] = torch.zeros_like(p.data)

                    state['square_avg'].share_memory_()
                elif self.optimizer_name == 'Rprop':
                    state['prev'] = torch.zeros_like(p.data)

                    state['prev'].share_memory_()
                elif self.optimizer_name == 'ASGD':
                    state['mu'] = torch.ones(1)
                    state['ax'] = torch.tensor(group['lr'])
                    state['eta'] = torch.zeros_like(p.data)

                    state['mu'].share_memory_()
                    state['ax'].share_memory_()
                    state['eta'].share_memory_()

                state['step'].share_memory_()

    def cp_grad(self, local_model, global_model):
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            global_param.grad = local_param.grad.cpu()

    def cp_grad_all(self, local_network):
        if isinstance(self.network, list):
            for i, global_network in enumerate(self.network):
                self.cp_grad(local_network[i], global_network)
        else:
            self.cp_grad(local_network, self.network)
    def step(self, loss, local_model):
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradient:
            nn.utils.clip_grad_norm(self.parameters, *self.gradient_clipping_hyperparameters)

        self.cp_grad_all(local_model)

        self.optimizer.step()

        self.lr_scheduler.step()

