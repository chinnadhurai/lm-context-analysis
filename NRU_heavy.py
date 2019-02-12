"""Implementation of the Non-Saturating Recurrent Unit(NRU)."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, memory_size=1152, k=2, use_relu=False, layer_norm=True):
        super(NRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.memory_size = memory_size
        self.k = k
        self._use_relu =  use_relu
        self._layer_norm = layer_norm

        assert math.sqrt(self.memory_size*self.k).is_integer()
        sqrt_memk = int(math.sqrt(self.memory_size*self.k))
        self.hm2v_alpha = nn.Linear(self.memory_size + hidden_size, 2 * sqrt_memk)
        self.hm2v_beta = nn.Linear(self.memory_size + hidden_size, 2 * sqrt_memk)
        self.hm2alpha = nn.Linear(self.memory_size + hidden_size, self.k)
        self.hm2beta = nn.Linear(self.memory_size + hidden_size, self.k)

        if self._layer_norm:
            self._ln_h = nn.LayerNorm(hidden_size)

        self.hmi2h = nn.Linear(self.memory_size + hidden_size + self.input_size, hidden_size)

    def _opt_relu(self, x):
        if self._use_relu:
            return F.relu(x)
        else:
            return x

    def _opt_layernorm(self, x):
        if self._layer_norm:
            return self._ln_h(x)
        else:
            return x

    def forward(self, input, last_hidden):
        last_hidden_h, last_hidden_memory = last_hidden
        c_input = torch.cat((input, last_hidden_h, last_hidden_memory), 1)

        h = F.relu(self._opt_layernorm(self.hmi2h(c_input)))

        # Flat memory equations
        alpha = self._opt_relu(self.hm2alpha(torch.cat((h,last_hidden_memory),1))).clone()
        beta = self._opt_relu(self.hm2beta(torch.cat((h,last_hidden_memory),1))).clone()

        u_alpha = self.hm2v_alpha(torch.cat((h,last_hidden_memory),1)).chunk(2,dim=1)
        v_alpha = torch.bmm(u_alpha[0].unsqueeze(2), u_alpha[1].unsqueeze(1)).view(-1, self.k, self.memory_size)
        v_alpha = self._opt_relu(v_alpha)
        v_alpha = torch.nn.functional.normalize(v_alpha, p=5, dim=2, eps=1e-12)
        add_memory = alpha.unsqueeze(2)*v_alpha

        u_beta = self.hm2v_beta(torch.cat((h,last_hidden_memory),1)).chunk(2, dim=1)
        v_beta = torch.bmm(u_beta[0].unsqueeze(2), u_beta[1].unsqueeze(1)).view(-1, self.k, self.memory_size)
        v_beta = self._opt_relu(v_beta)
        v_beta = torch.nn.functional.normalize(v_beta, p=5, dim=2, eps=1e-12)
        forget_memory = beta.unsqueeze(2)*v_beta

        hidden_memory = last_hidden_memory + torch.mean(add_memory, dim=1) - torch.mean(forget_memory, dim=1)
        hidden_h = h
        return (hidden_h, hidden_memory)

    def reset_hidden(self, batch_size):
        hidden_h = torch.Tensor(np.zeros((batch_size, self.hidden_size))).cuda()
        hidden_memory = torch.Tensor(np.zeros((batch_size, self.memory_size))).cuda()
        return (hidden_h, hidden_memory)


class NRU(nn.Module):
    """Implementation of the Non-Saturating Recurrent Unit(NRU)."""

    def __init__(self, input_size, hidden_size, num_layers=1, 
                 layer_norm=False, use_relu=True, memory_size=256*3, k=3):
        """Initialization parameters for NRU.
        
        Args:
            input_size: Number of expected features in the input.
            hidden_size: Number of expected features in the hidden layer.
            num_layers: Number of stacked recurrent layers.
            use_relu: If true, use ReLU activations over the erase/write memory vectors.
            memory_size: Number of dimensions of the memory vector.
            k: Number of erase/write heads.     
        
        """
        
        super(NRU, self).__init__()

        self._input_size = input_size
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._layer_norm = layer_norm
        self._use_relu = use_relu
        self._memory_size = memory_size
        self._k = k

        self._Cells = []

        self._add_cell(self._input_size, self._hidden_size)
        for i in range(1, num_layers):
            self._add_cell(self._hidden_size, self._hidden_size)

        self._list_of_modules = nn.ModuleList(self._Cells)

        self.print_num_parameters()

    def forward(self, input, init_hidden=None):
        """Implements forward computation of the model.
        Input Args and outputs are in the same format as torch.nn.GRU cell.
        
        Args:
            input of shape (seq_len, batch, input_size): 
            tensor containing the features of the input sequence.
            Support for packed variable length sequence not available yet.
            
            hidden_init of shape (num_layers * num_directions, batch, hidden_size): 
            tensor containing the initial hidden state for each element in the batch. 
            Defaults to zero if not provided.
        
        Returns:
            output of shape (seq_len, batch, hidden_size): 
            tensor containing the output features h_t from the last layer of the GRU, for each t
            
            h_n of shape (num_layers, batch, hidden_size): 
            tensor containing the hidden state for t = seq_len 
        """
        batch_size = input.shape[1]
        seq_len = input.shape[0]

        if init_hidden == None:                                  
            init_hidden = self.reset_hidden(batch_size)

        hiddens = [self.step(input[0], init_hidden)]
        for t in range(1, seq_len):
            hiddens.append(self.step(input[t], hiddens[-1]))
        

        #output contruction
        output = torch.stack([h[-1] for h, _ in hiddens], dim=0) 

        #h_n construction
        h_n = hiddens[-1][0]
        m_n = hiddens[-1][1]
    
        return output, (h_n, m_n)

    def step(self, input, hidden):
        """Implements forward computation of the model for a single recurrent step.

        Args:
            input of shape (batch, input_size): 
            tensor containing the features of the input sequence

        Returns:
            model output for current time step.
        """
        h_n,m_n = hidden
        h = [self._Cells[0](input, (h_n[0], m_n[0]))]
        for i, cell in enumerate(self._Cells[1:]):
                next_input, _ = h[-1]
                h.append(cell(next_input, (h_n[i], m_n[i])))

        #h_n construction
        h_n = torch.stack([h for h, _ in h], dim=0)
        m_n = torch.stack([m for _, m in h], dim=0)

        return (h_n, m_n)

    def reset_hidden(self, batch_size):
        """Resets the hidden state for truncating the dependency."""

        h = []
        for i, cell in enumerate(self._Cells):
            h.append(cell.reset_hidden(batch_size))

        #h_n construction
        h_n = torch.stack([h for h, _ in h], dim=0)
        m_n = torch.stack([m for _, m in h], dim=0)
        return (h_n, m_n)

    def register_optimizer(self, optimizer):
        """Registers an optimizer for the model.

        Args:
            optimizer: optimizer object.
        """

        self.optimizer = optimizer

    def _add_cell(self, input_size, hidden_size):
        """Adds a cell to the stack of cells.

        Args:
            input_size: int, size of the input vector.
            hidden_size: int, hidden layer dimension.
        """

        self._Cells.append(NRUCell(input_size, hidden_size, 
                                   memory_size=self._memory_size, k=self._k,
                                   use_relu=self._use_relu, layer_norm=self._layer_norm))
                                    

    def save(self, save_dir):
        """Saves the model and the optimizer.

        Args:
            save_dir: absolute path to saving dir.
        """

        file_name = os.path.join(save_dir, "model.p")
        torch.save(self.state_dict(), file_name)

        file_name = os.path.join(save_dir, "optim.p")
        torch.save(self.optimizer.state_dict(), file_name)

    def load(self, save_dir):
        """Loads the model and the optimizer.

        Args:
            save_dir: absolute path to loading dir.
        """

        file_name = os.path.join(save_dir, "model.p")
        self.load_state_dict(torch.load(file_name))

        file_name = os.path.join(save_dir, "optim.p")
        self.optimizer.load_state_dict(torch.load(file_name))

    def print_num_parameters(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        print("Num_params : {} ".format(num_params))
        return num_params
