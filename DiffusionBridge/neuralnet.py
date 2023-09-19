"""
A module to approximate score functions with neural networks.
"""

import torch
import torch.nn.functional as F
from torch import nn
import math

def get_timestep_embedding(timesteps, embedding_dim = 128):
    """
    From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py

    Parameters
    ----------
    timesteps : state (N, 1)

    embedding_dim : int specifying dimension of time embedding 
                        
    Returns
    -------    
    emb : time embedding (N, embedding_dim)
    """
    scaling_factor = 100.0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)
    emb = scaling_factor * timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn = F.relu):
        """
        Parameters
        ----------    
        input_dim : int specifying dimension of input 

        layer_widths : list specifying width of each layer 
            (len is the number of layers, and last element is the output dimension)

        activate_final : bool specifying if activation function is applied in the final layer

        activation_fn : activation function for each layer        
        """
        super(MLP, self).__init__()
        layers = []
        norms = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            norms.append(torch.nn.LayerNorm(layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
            
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.norms = torch.nn.ModuleList(norms)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
            x = self.norms[i](x)
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x


class FullScoreNetworkWithLSTM(torch.nn.Module):
    def __init__(self, dimension, lstm_hidden_size, pos_dim=128, decoder_layers=[256, 256]):
        super().__init__()
        self.temb_dim = pos_dim
        self.lstm = torch.nn.LSTM(input_size=pos_dim, hidden_size=lstm_hidden_size, batch_first=True)
        # 其他初始化代码保持不变

    def forward(self, t, x, x0):
        # 其他代码保持不变
        temb = get_timestep_embedding(t, self.temb_dim)
        lstm_out, _ = self.lstm(temb.view(len(temb), 1, -1))
        temb = lstm_out.view(len(temb), -1)
        # 其他代码保持不变


class ScoreNetwork(torch.nn.Module):

    def __init__(self, dimension, encoder_layers = [16], pos_dim = 16, decoder_layers = [128,128]):
        """
        Parameters
        ----------    
        dimension : int specifying dimension of state variable (same as output of network)

        encoder_layers : list specifying width of each encoder layer 

        pos_dim : int specifying dimension of time embedding

        decoder_layers : list specifying width of each decoder layer 
        """
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, dimension]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths = decoder_layers + [dimension],
                       activate_final = False,
                       activation_fn = torch.nn.SiLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths = encoder_layers + [t_enc_dim],
                             activate_final = False,
                             activation_fn = torch.nn.SiLU())

        self.x_encoder = MLP(dimension,
                             layer_widths = encoder_layers + [t_enc_dim],
                             activate_final = False,
                             activation_fn = torch.nn.SiLU())

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : time step (N, 1)

        x : state (N, dimension)
                        
        Returns
        -------    
        out :  score (N, dimension)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim) # size (N, temb_dim)
        temb = self.t_encoder(temb) # size (N, t_enc_dim)
        xemb = self.x_encoder(x) # size (N, t_enc_dim)
        h = torch.cat([xemb, temb], -1) # size (N, 2 * t_enc_dim)
        out = self.net(h) # size (N, dimension)
        return out
        
class FullScoreNetwork(torch.nn.Module):

    def __init__(self, dimension, encoder_layers = [64], pos_dim = 64, decoder_layers = [256,256]):
        """
        Parameters
        ----------    
        dimension : int specifying dimension of state variable (same as output of network)

        encoder_layers : list specifying width of each encoder layer 

        pos_dim : int specifying dimension of time embedding

        decoder_layers : list specifying width of each decoder layer 
        """
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, dimension]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths = decoder_layers + [dimension],
                       activate_final = False,
                       activation_fn = torch.nn.SiLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths = encoder_layers + [t_enc_dim],
                             activate_final = True,
                             activation_fn = torch.nn.SiLU())

        self.x_encoder = MLP(2 * dimension,
                             layer_widths = encoder_layers + [t_enc_dim],
                             activate_final = True,
                             activation_fn = torch.nn.SiLU())
        self.lstm = torch.nn.LSTM(input_size=4 * pos_dim, hidden_size=2 * t_enc_dim, batch_first=True)


    def forward(self, t, x, x0):
        """
        Parameters
        ----------
        t : time step (N, 1)

        x : state (N, dimension)

        x0 : initial state (N, dimension)
                        
        Returns
        -------    
        out :  score (N, dimension)
        """

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim) # size (N, temb_dim)
        temb = self.t_encoder(temb) # size (N, t_enc_dim)
        states = torch.cat([x, x0], -1) # size (N, 2*dimension)
        xemb = self.x_encoder(states) # size (N, t_enc_dim)
        h = torch.cat([xemb, temb], -1) # size (N, 2 * t_enc_dim)
        # print(h.shape)
        lstm_out, _ = self.lstm(h.view(len(h), 1, -1))
        h = lstm_out.view(len(h), -1)
        # print(h.shape)
        out = self.net(h) # size (N, dimension)
        return out
    
class newFullScoreNetwork(torch.nn.Module):

    def __init__(self, dimension, encoder_layers = [64], pos_dim = 64, decoder_layers = [256,256]):
        """
        Parameters
        ----------    
        dimension : int specifying dimension of state variable (same as output of network)

        encoder_layers : list specifying width of each encoder layer 

        pos_dim : int specifying dimension of time embedding

        decoder_layers : list specifying width of each decoder layer 
        """
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, dimension]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths = decoder_layers + [dimension],
                       activate_final = False,
                       activation_fn = torch.nn.SiLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths = encoder_layers + [t_enc_dim],
                             activate_final = False,
                             activation_fn = torch.nn.SiLU())

        self.x_encoder = MLP(3 * dimension,
                             layer_widths = encoder_layers + [t_enc_dim],
                             activate_final = False,
                             activation_fn = torch.nn.SiLU())

    def forward(self, t, x, x0, xT):
        """
        Parameters
        ----------
        t : time step (N, 1)

        x : state (N, dimension)

        x0 : initial state (N, dimension)
                        
        Returns
        -------    
        out :  score (N, dimension)
        """

        # print(x.shape, x0.shape, xT.shape)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(0)
            
        if len(xT.shape) == 1:
            xT = xT.unsqueeze(0)
        temb = get_timestep_embedding(t, self.temb_dim) # size (N, temb_dim)
        temb = self.t_encoder(temb) # size (N, t_enc_dim)
        states = torch.cat([x, x0, xT], -1) # size (N, 2*dimension)
        xemb = self.x_encoder(states) # size (N, t_enc_dim)
        h = torch.cat([xemb, temb], -1) # size (N, 2 * t_enc_dim)
        out = self.net(h) # size (N, dimension)
        return out
