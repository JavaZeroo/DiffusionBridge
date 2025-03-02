"""
A module to simulate approximations of diffusion and diffusion bridge processes.
"""

import torch
import torch.nn.functional as F
from DiffusionBridge.neuralnet import ScoreNetwork, FullScoreNetwork, newFullScoreNetwork
from DiffusionBridge.ema import ema_register, ema_update, ema_copy
from DiffusionBridge.utils import normal_logpdf
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, OneCycleLR

from rich.panel import Panel
from rich.pretty import Pretty
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from utils.data import bridgeBackwardsDataset
from torch.utils.data import DataLoader


def construct_time_discretization(terminal_time, num_steps):
    stepsizes = (terminal_time / num_steps) * torch.ones(num_steps)
    time = torch.linspace(0.0, terminal_time, num_steps + 1)
    return (time, stepsizes)


class model(torch.nn.Module):

    def __init__(self, f, sigma, dimension, terminal_time, num_steps):
        """
        Parameters
        ----------    
        f : drift function
        sigma : diffusion coefficient (assume constant for now)
        dimension : dimension of diffusion
        terminal_time : length of time horizon
        num_steps : number of time-discretization steps        
        """
        super().__init__()
        self.f = f
        self.sigma = sigma
        self.Sigma = sigma * sigma
        self.invSigma = 1.0 / self.Sigma
        self.d = dimension
        self.T = terminal_time
        self.num_steps = num_steps
        self.debug = False
        (self.time, self.stepsizes) = construct_time_discretization(
            terminal_time, num_steps)

    def gen_drift(self, time, now_states, terminate_states):
        # dydt = torch.tensor(0)
        dydt = terminate_states - now_states
        return dydt
    
    def gen_bridge(self, time, now_states, terminate_states):
        # dydt = torch.tensor(0)
        dydt = (terminate_states - now_states) / (self.T - time)
        return dydt

    def simulate_process(self, initial_states):
        """
        Simulate diffusion process using Euler-Maruyama discretization.

        Parameters
        ----------    
        initial_states : initial condition of size (N, d)

        Returns
        -------   
        output : dict containing 
            trajectories : realizations of time-discretized process (N, M+1, d)
            scaled_brownian : scaled brownian increments (N, M, d)
        """
        # initialize and preallocate
        N = initial_states.shape[0]
        M = self.num_steps
        X = initial_states.clone()  # size (N ,d)
        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        scaled_brownian = torch.zeros(N, M, self.d)

        # simulate process forwards in time
        for m in range(M):
            stepsize = self.stepsizes[m]
            t = self.time[m]
            drift = 0
            euler = X + stepsize * drift
            brownian = torch.sqrt(stepsize) * \
                torch.randn(X.shape)  # size (N x d)
            X = euler + self.sigma * brownian
            trajectories[:, m+1, :] = X
            scaled_brownian[:, m, :] = - \
                (self.invSigma / stepsize) * self.sigma * brownian

        # output
        output = {'trajectories': trajectories,
                  'scaled_brownian': scaled_brownian}

        return output

    def my_simulate_bridge(self, initial_states, terminate_states):
        """
        Simulate diffusion process using Euler-Maruyama discretization.

        Parameters
        ----------    
        initial_states : initial condition of size (N, d)

        Returns
        -------   
        output : dict containing 
            trajectories : realizations of time-discretized process (N, M+1, d)
            scaled_brownian : scaled brownian increments (N, M, d)
        """
        # initialize and preallocate
        N = initial_states.shape[0]
        M = self.num_steps
        X = initial_states.clone()  # size (N ,d)
        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        scaled_brownian = torch.zeros(N, M, self.d)

        # simulate process forwards in time
        for m in range(M):
            stepsize = self.stepsizes[m]
            t = self.time[m]
            drift = (terminate_states - trajectories[:, m, :]) / (self.T - t)
            euler = X + stepsize * drift
            brownian = torch.sqrt(stepsize) * \
                torch.randn(X.shape)  # size (N x d)
            X = euler + self.sigma * brownian
            trajectories[:, m+1, :] = X
            scaled_brownian[:, m, :] = - \
                (self.invSigma / stepsize) * self.sigma * brownian

        # output
        output = {'trajectories': trajectories,
                  'scaled_brownian': scaled_brownian}

        return output
    
    def my_simulate_process(self, initial_states, terminate_states):
        """
        Simulate diffusion process using Euler-Maruyama discretization.

        Parameters
        ----------    
        initial_states : initial condition of size (N, d)

        Returns
        -------   
        output : dict containing 
            trajectories : realizations of time-discretized process (N, M+1, d)
            scaled_brownian : scaled brownian increments (N, M, d)
        """
        # initialize and preallocate
        N = initial_states.shape[0]
        M = self.num_steps
        X = initial_states.clone()  # size (N ,d)
        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        scaled_brownian = torch.zeros(N, M, self.d)

        # simulate process forwards in time
        for m in range(M):
            stepsize = self.stepsizes[m]
            t = self.time[m]
            drift = self.gen_drift(t, X, terminate_states)
            euler = X + stepsize * drift
            brownian = torch.sqrt(stepsize) * \
                torch.randn(X.shape)  # size (N x d)
            X = euler + self.sigma * brownian
            trajectories[:, m+1, :] = X
            scaled_brownian[:, m, :] = - \
                (self.invSigma / stepsize) * self.sigma * brownian

        # output
        output = {'trajectories': trajectories,
                  'scaled_brownian': scaled_brownian}

        return output

    def my_simulate_bridge_backwards(self, score_net, initial_state, terminal_state, epsilon, num_samples=1, modify=False, full_score=False, full_model=False, new_num_steps=None, device='cpu'):
        """
        Simulate diffusion bridge process backwards using Euler-Maruyama discretization.

        Parameters
        ----------
        score_net : neural network approximation of score function of transition density

        initial_state : initial condition of size d or (N, d)

        terminal_state : terminal condition of size d or (N, d)

        epsilon : positive constant to enforce initial constraint 

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 

        full_score : bool specifying if the full score function is employed

        new_num_steps : new number of time-discretization steps        

        Returns
        -------    
        output : dict containing
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
            scaled_brownian : scaled brownian increments (N, M, d)
            score_evaluations : evaluations of score network (N, M, d)
        """

        # initialize and preallocate
        if len(initial_state.shape) == 1:
            X0 = initial_state.repeat(num_samples, 1)  # size (N, d)
            N = num_samples
        else:
            X0 = initial_state.clone()
            N = initial_state.shape[0]
        X0 = X0.to('cpu')
        if len(terminal_state.shape) == 1:
            Z = terminal_state.repeat(num_samples, 1)  # size (N, d)
            N = num_samples
        else:
            Z = terminal_state.clone()
            N = terminal_state.shape[0]
        if full_model:
            XT = Z.clone()
        
        if new_num_steps is None:
            M = self.num_steps
            timesteps = self.time
            stepsizes = self.stepsizes
        else:
            M = new_num_steps
            (timesteps, stepsizes) = construct_time_discretization(self.T, M)
        trajectories = torch.zeros(N, M+1, self.d).to(device)
        trajectories[:, M, :] = Z.to(device)
        scaled_brownian = torch.zeros(N, M, self.d).to(device)
        score_evaluations = torch.zeros(N, M, self.d).to(device)
        timesteps = timesteps.to(device)
        stepsizes = stepsizes.to(device)
        sigma = self.sigma.to(device)
        Sigma = self.Sigma.to(device)
        invSigma = self.invSigma.to(device)

        # simulate process backwards in time
        for m in range(M, 0, -1):
            stepsize = stepsizes[m-1]
            t = timesteps[m]
            t_next = timesteps[m-1]
            score_net.to(device)
            t = t.repeat((N, 1)).to(device)
            Z = Z.to(device)
            # XT = Z.copy_()
            if full_score and not full_model:
                X0 = X0.to(device)
                score = score_net(t, Z, X0)  # size (N, d)
            elif full_model:
                X0 = X0.to(device)
                XT = XT.to(device)
                score = score_net(t, Z, X0, XT)  # size (N, d
            else:
                score = score_net(t, Z)  # size (N, d)
            # score = score.cpu()
            if self.debug:
                print(score.shape)
            score_evaluations[:, m-1, :] = score
            drift = Sigma * score + epsilon * (X0 - Z) / t - self.gen_drift(t,Z, terminal_state)

            euler = Z + stepsize * drift
            if (m > 1):
                if modify:
                    scaling = stepsize * t_next / t
                else:
                    scaling = stepsize
                brownian = torch.sqrt(
                    scaling) * torch.randn(Z.shape).to(device)  # size (N x d)
                Z = euler + sigma * brownian
                trajectories[:, m-1, :] = Z
                scaled_brownian[:, m-1, :] = - \
                    (invSigma / scaling) * sigma * brownian
            else:
                # terminal constraint
                if modify:
                    # fudging a little here because of singularity
                    scaling = stepsize * 0.25 * stepsize / t
                else:
                    scaling = stepsize
                trajectories[:, 0, :] = X0
                scaled_brownian[:, 0, :] = - \
                    (invSigma / scaling) * (X0 - euler)

        # output
        output = {'trajectories': trajectories, 'scaled_brownian': scaled_brownian,
                  'score_evaluations': score_evaluations}

        return output

    def simulate_bridge_backwards(self, score_net, initial_state, terminal_state, epsilon, num_samples=1, modify=False, full_score=False, new_num_steps=None):
        """
        Simulate diffusion bridge process backwards using Euler-Maruyama discretization.

        Parameters
        ----------
        score_net : neural network approximation of score function of transition density

        initial_state : initial condition of size d or (N, d)

        terminal_state : terminal condition of size d or (N, d)

        epsilon : positive constant to enforce initial constraint 

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 

        full_score : bool specifying if the full score function is employed

        new_num_steps : new number of time-discretization steps        

        Returns
        -------    
        output : dict containing
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
            scaled_brownian : scaled brownian increments (N, M, d)
            score_evaluations : evaluations of score network (N, M, d)
        """

        # initialize and preallocate
        if len(initial_state.shape) == 1:
            X0 = initial_state.repeat(num_samples, 1)  # size (N, d)
            N = num_samples
        else:
            X0 = initial_state.clone()
            N = initial_state.shape[0]

        if len(terminal_state.shape) == 1:
            Z = terminal_state.repeat(num_samples, 1)  # size (N, d)
            N = num_samples
        else:
            Z = terminal_state.clone()
            N = terminal_state.shape[0]

        if new_num_steps is None:
            M = self.num_steps
            timesteps = self.time
            stepsizes = self.stepsizes
        else:
            M = new_num_steps
            (timesteps, stepsizes) = construct_time_discretization(self.T, M)

        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, M, :] = Z
        logdensity = torch.zeros(N)
        scaled_brownian = torch.zeros(N, M, self.d)
        score_evaluations = torch.zeros(N, M, self.d)

        # simulate process backwards in time
        for m in range(M, 0, -1):
            stepsize = stepsizes[m-1]
            t = timesteps[m]
            t_next = timesteps[m-1]
            if full_score:
                score = score_net(t.repeat((N, 1)), Z, X0)  # size (N, d)
            else:
                score = score_net(t.repeat((N, 1)), Z)  # size (N, d)
            score_evaluations[:, m-1, :] = score
            drift = -self.f(t, Z) + self.Sigma * score + epsilon * (X0 - Z) / t
            euler = Z + stepsize * drift
            if (m > 1):
                if modify:
                    scaling = stepsize * t_next / t
                else:
                    scaling = stepsize
                brownian = torch.sqrt(scaling) * \
                    torch.randn(Z.shape)  # size (N x d)
                Z = euler + self.sigma * brownian
                logdensity += normal_logpdf(Z, euler, scaling * self.Sigma)
                trajectories[:, m-1, :] = Z
                scaled_brownian[:, m-1, :] = - \
                    (self.invSigma / scaling) * self.sigma * brownian
            else:
                # terminal constraint
                if modify:
                    # fudging a little here because of singularity
                    scaling = stepsize * 0.25 * stepsize / t
                else:
                    scaling = stepsize
                trajectories[:, 0, :] = X0
                scaled_brownian[:, 0, :] = - \
                    (self.invSigma / scaling) * (X0 - euler)

        # output
        output = {'trajectories': trajectories, 'logdensity': logdensity,
                  'scaled_brownian': scaled_brownian, 'score_evaluations': score_evaluations}

        return output

    def simulate_bridge_forwards(self, score_transition_net, score_marginal_net, initial_state, terminal_state, epsilon, num_samples=1, modify=False, full_score=False, new_num_steps=None):
        """
        Simulate diffusion bridge process fowards using Euler-Maruyama discretization.

        Parameters
        ----------
        score_transition_net : neural network approximation of score function of transition density

        score_marginal_net : neural network approximation of score function of marginal (diffusion bridge) density

        initial_state : initial condition of size d or (N, d)

        terminal_state : terminal condition of size d or (N, d)

        epsilon : positive constant to enforce initial constraint 

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 

        full_score : bool specifying if the full score function is employed

        new_num_steps : new number of time-discretization steps        

        Returns
        -------   
        output : dict containing  
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
        """

        # initialize and preallocate
        T = self.T
        if len(initial_state.shape) == 1:
            X0 = initial_state.repeat(num_samples, 1)  # size (N, d)
            X = initial_state.repeat(num_samples, 1)  # size (N, d)
            N = num_samples
        else:
            X0 = initial_state.clone()
            X = initial_state.clone()
            N = initial_state.shape[0]

        if len(terminal_state.shape) == 1:
            XT = terminal_state.repeat(num_samples, 1)  # size (N, d)
            N = num_samples
        else:
            XT = terminal_state.clone()
            N = terminal_state.shape[0]

        if new_num_steps is None:
            M = self.num_steps
            timesteps = self.time
            stepsizes = self.stepsizes
        else:
            M = new_num_steps
            (timesteps, stepsizes) = construct_time_discretization(T, M)

        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        logdensity = torch.zeros(N)

        # simulate process forwards in time
        for m in range(M-1):
            stepsize = stepsizes[m]
            if m == 0:
                # fudging a little here because of singularity
                t = timesteps[m] + 0.5 * stepsize
            else:
                t = timesteps[m]
            t_next = timesteps[m+1]
            if full_score:
                score_marginal = score_marginal_net(
                    t.repeat(N, 1), X)  # size (N, d)
                # score_marginal = score_marginal_net(t.repeat(N,1), X, X0, XT) # size (N, d)
                score_transition = score_transition_net(
                    t.repeat(N, 1), X, X0)  # size (N, d)
            else:
                score_marginal = score_marginal_net(
                    t.repeat(N, 1), X)  # size (N, d)
                score_transition = score_transition_net(
                    t.repeat(N, 1), X)  # size (N, d)
            drift = self.f(t, X) + self.Sigma * (score_marginal -
                                                 score_transition) + epsilon * ((XT - X) / (T - t) - (X0 - X) / t)
            euler = X + stepsize * drift
            if modify:
                scaling = stepsize * (T - t_next) / (T - t)
            else:
                scaling = stepsize
            brownian = torch.sqrt(scaling) * \
                torch.randn(X.shape)  # size (N x d)
            X = euler + self.sigma * brownian
            logdensity += normal_logpdf(X, euler, scaling * self.Sigma)
            trajectories[:, m+1, :] = X

        # terminal constraint
        trajectories[:, M, :] = XT

        # output
        output = {'trajectories': trajectories, 'logdensity': logdensity}

        return output

    def my_simulate_bridge_forwards(self, score_transition_net, score_marginal_net, initial_state, terminal_state, epsilon, num_samples=1, modify=False, full_score=False, new_num_steps=None):
        """
        Simulate diffusion bridge process fowards using Euler-Maruyama discretization.

        Parameters
        ----------
        score_transition_net : neural network approximation of score function of transition density

        score_marginal_net : neural network approximation of score function of marginal (diffusion bridge) density

        initial_state : initial condition of size d or (N, d)

        terminal_state : terminal condition of size d or (N, d)

        epsilon : positive constant to enforce initial constraint 

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 

        full_score : bool specifying if the full score function is employed

        new_num_steps : new number of time-discretization steps        

        Returns
        -------   
        output : dict containing  
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
        """

        # initialize and preallocate
        T = self.T
        if len(initial_state.shape) == 1:
            X0 = initial_state.repeat(num_samples, 1)  # size (N, d)
            X = initial_state.repeat(num_samples, 1)  # size (N, d)
            N = num_samples
        else:
            X0 = initial_state.clone()
            X = initial_state.clone()
            N = initial_state.shape[0]

        if len(terminal_state.shape) == 1:
            XT = terminal_state.repeat(num_samples, 1)  # size (N, d)
            N = num_samples
        else:
            XT = terminal_state.clone()
            N = terminal_state.shape[0]

        if new_num_steps is None:
            M = self.num_steps
            timesteps = self.time
            stepsizes = self.stepsizes
        else:
            M = new_num_steps
            (timesteps, stepsizes) = construct_time_discretization(T, M)

        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        logdensity = torch.zeros(N)

        # simulate process forwards in time
        for m in range(M-1):
            stepsize = stepsizes[m]
            if m == 0:
                # fudging a little here because of singularity
                t = timesteps[m] + 0.5 * stepsize
            else:
                t = timesteps[m]
            t_next = timesteps[m+1]
            if full_score:
                score_marginal = score_marginal_net(
                    t.repeat(N, 1), X, X0, XT)  # size (N, d)
                score_transition = score_transition_net(
                    t.repeat(N, 1), X, X0)  # size (N, d)
            else:
                score_marginal = score_marginal_net(
                    t.repeat(N, 1), X)  # size (N, d)
                score_transition = score_transition_net(
                    t.repeat(N, 1), X)  # size (N, d)
            # + epsilon * ((XT - X) / (T - t) - (X0 - X) / t) #+ self.f(t, X)
            drift = self.Sigma * (score_marginal - score_transition)
            euler = X + stepsize * drift
            if modify:
                scaling = stepsize * (T - t_next) / (T - t)
            else:
                scaling = stepsize
            brownian = torch.sqrt(scaling) * \
                torch.randn(X.shape)  # size (N x d)
            X = euler + self.sigma * brownian
            logdensity += normal_logpdf(X, euler, scaling * self.Sigma)
            trajectories[:, m+1, :] = X

        # terminal constraint
        trajectories[:, M, :] = XT

        # output
        output = {'trajectories': trajectories, 'logdensity': logdensity}

        return output

    def simulate_proposal_bridge(self, drift, initial_state, terminal_state, num_samples, modify=False, new_num_steps=None):
        """
        Simulate proposal diffusion bridge process fowards using Euler-Maruyama discretization.

        Parameters
        ----------
        drift : proposal drift function

        initial_state : initial condition of size d

        terminal_state : terminal condition of size d

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 

        new_num_steps : new number of time-discretization steps        

        Returns
        -------   
        output : dict containing  
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
        """

        # initialize and preallocate
        N = num_samples
        T = self.T

        if new_num_steps is None:
            M = self.num_steps
            timesteps = self.time
            stepsizes = self.stepsizes
        else:
            M = new_num_steps
            (timesteps, stepsizes) = construct_time_discretization(T, M)

        X = initial_state.repeat(N, 1)  # size (N, d)
        XT = terminal_state.repeat(N, 1)  # size (N, d)
        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        logdensity = torch.zeros(N)

        # simulate process forwards in time
        for m in range(M-1):
            stepsize = stepsizes[m]
            t = timesteps[m]
            t_next = timesteps[m+1]
            euler = X + stepsize * drift(t, X)
            if modify:
                scaling = stepsize * (T - t_next) / (T - t)
            else:
                scaling = stepsize
            brownian = torch.sqrt(scaling) * \
                torch.randn(X.shape)  # size (N x d)
            X = euler + self.sigma * brownian
            logdensity += normal_logpdf(X, euler, scaling * self.Sigma)
            trajectories[:, m+1, :] = X

        # terminal constraint
        trajectories[:, M, :] = XT

        # output
        output = {'trajectories': trajectories, 'logdensity': logdensity}

        return output

    def law_bridge(self, trajectories, new_num_steps=None):
        """
        Evaluate law of (time-discretized) diffusion bridge process.

        Parameters
        ----------
        trajectories : realizations of time-discretized proposal bridge process satisfying initial and terminal constraints (N, M+1, d)

        new_num_steps : new number of time-discretization steps

        Returns
        -------    
        logdensity : log-density values of size N
        """

        N = trajectories.shape[0]
        logdensity = torch.zeros(N)

        if new_num_steps is None:
            M = self.num_steps
            timesteps = self.time
            stepsizes = self.stepsizes
        else:
            M = new_num_steps
            (timesteps, stepsizes) = construct_time_discretization(self.T, M)

        for m in range(M):
            stepsize = stepsizes[m]
            t = timesteps[m]
            X_current = trajectories[:, m, :]
            drift = self.f(t, X_current)
            euler = X_current + stepsize * drift
            X_next = trajectories[:, m+1, :]
            logdensity += normal_logpdf(X_next, euler, stepsize * self.Sigma)

        return logdensity

    def gradient_transition(self, trajectories, scaled_brownian, epsilon):
        """
        Evaluate gradient function needed in score matching to learn score of transition density.

        Parameters
        ----------
        trajectories : realizations of time-discretized process (N, M+1, d)

        scaled_brownian : scaled brownian increments (N, M, d)

        epsilon : positive constant to enforce initial constraint 

        Returns
        -------    
        grad : gradient function evaluations (N, M, d)
        """
        N = trajectories.shape[0]
        M = self.num_steps
        grad = torch.zeros(N, M, self.d)
        XT = trajectories[:, -1, :]

        for m in range(M):
            X_next = trajectories[:, m, :]
            # if (m == (M-1)):
            #     # fudging a little here because of singularity
            #     t_next = self.time[m+1] - 0.25 * self.stepsizes[m]
            # else:
            t_next = self.time[m]
            grad[:, m, :] = scaled_brownian[:, m, :] - \
                self.invSigma * self.gen_drift(t_next, X_next, XT)

        return grad

    def learn_score_transition(self, initial_state, terminal_state, epsilon, minibatch, num_iterations, learning_rate, ema_momentum):
        """
        Learn approximation of score transition using score matching.

        Parameters
        ----------
        initial_state : initial condition of size d

        terminal_state : terminal condition of size d

        epsilon : positive constant to enforce initial constraint 

        minibatch : number of mini-batch samples desired

        num_iterations : number of optimization iterations (divisible by num_batches)

        learning_rate : learning rate of Adam optimizer

        ema_momentum : momentum parameter of exponential moving average update

        Returns
        -------
        output : dict containing    
            net : neural network approximation of score function of transition density
            loss : value of loss function during learning
        """

        M = self.num_steps
        N = minibatch
        timesteps = self.time[1:(M+1)].reshape((1, M, 1)
                                               ).repeat((N, 1, 1))  # size (N, M, 1)
        timesteps_flatten = timesteps.flatten(
            start_dim=0, end_dim=1)  # size (N*M, 1)
        loss_values = torch.zeros(num_iterations)

        # create score network
        score_net = ScoreNetwork(dimension=self.d)
        ema_parameters = ema_register(score_net)

        # optimization
        optimizer = torch.optim.Adam(score_net.parameters(), lr=learning_rate)
        num_batches = 10
        num_samples = num_batches * N
        num_repeats = int(num_iterations / num_batches)
        iteration = 1
        for i in range(num_repeats):
            # simulate trajectories from diffusion process
            initial_states = initial_state.repeat(
                num_samples, 1)  # size (N, d)
            simulation_output = self.simulate_process(initial_states)
            trajectories = simulation_output['trajectories']
            scaled_brownian = simulation_output['scaled_brownian']

            for j in range(num_batches):
                # get minibatch of trajectories
                traj = trajectories[(j * N):((j+1) * N), :,
                                    :]  # size (N, M+1, d)
                scaled = scaled_brownian[(j * N):((j+1) * N), :, :]

                # evaluate gradient function
                grad = self.gradient_transition(
                    traj, scaled, epsilon)  # size (N, M, d)
                grad_flatten = grad.flatten(
                    start_dim=0, end_dim=1)  # size (N*M, d)

                # evaluate score network
                # size (N*M, d)
                traj_flatten = traj[:, 1:(
                    M+1), :].flatten(start_dim=0, end_dim=1)
                score = score_net(timesteps_flatten,
                                  traj_flatten)  # size (N*M, d)

                # compute loss function
                # need to extend this for non-uniform stepsizes
                loss = F.mse_loss(score, grad_flatten)

                # backpropagation
                loss.backward()

                # optimization step and zero gradient
                optimizer.step()
                optimizer.zero_grad()

                # update parameters using exponential moving average
                ema_update(ema_parameters, score_net, ema_momentum)

                # iteration counter
                current_loss = loss.item()
                loss_values[iteration-1] = current_loss
                if (iteration == 1) or (iteration % 50 == 0):
                    print("Optimization iteration:",
                          iteration, "Loss:", current_loss)
                iteration += 1

        # use exponential moving average parameters in score network
        ema_copy(ema_parameters, score_net)

        # output
        output = {'net': score_net, 'loss': loss_values}

        return output

    def learn_full_score_transition(self, simulate_initial_state, terminal_state, epsilon, minibatch, num_initial_per_batch, num_iterations, learning_rate, ema_momentum):
        """
        Learn full approximation of score transition using score matching.

        Parameters
        ----------
        simulate_initial_state : function returning simulated initial conditions

        terminal_state : terminal condition of size d

        epsilon : positive constant to enforce initial constraint 

        minibatch : number of mini-batch samples desired

        num_initial_per_batch : number of initial states per batch

        num_iterations : number of optimization iterations (divisible by num_batches)

        learning_rate : learning rate of Adam optimizer

        ema_momentum : momentum parameter of exponential moving average update

        Returns
        -------
        output : dict containing    
            net : neural network approximation of full score function of transition density
            loss : value of loss function during learning
        """

        d = self.d
        M = self.num_steps
        N = minibatch * num_initial_per_batch
        timesteps = self.time[1:(M+1)].reshape((1, M, 1)
                                               ).repeat((N, 1, 1))  # size (N, M, 1)
        timesteps_flatten = timesteps.flatten(
            start_dim=0, end_dim=1)  # size (N*M, 1)
        loss_values = torch.zeros(num_iterations)

        # create score network
        score_net = FullScoreNetwork(dimension=self.d)
        ema_parameters = ema_register(score_net)

        # optimization
        optimizer = torch.optim.Adam(score_net.parameters(), lr=learning_rate)
        for i in range(num_iterations):
            # simulate initial states
            initial_states = simulate_initial_state(
                num_initial_per_batch).repeat((minibatch, 1))  # size (N, d)
            initial_states_repeated = initial_states.reshape(
                (N, 1, d)).repeat((1, M, 1))  # size (N, M, d)
            initial_states_flatten = initial_states_repeated.flatten(
                start_dim=0, end_dim=1)  # size (N*M, d)

            # simulate trajectories from diffusion process
            simulation_output = self.simulate_process(initial_states)
            traj = simulation_output['trajectories']  # size (N, M+1, d)
            scaled = simulation_output['scaled_brownian']

            # evaluate gradient function
            grad = self.gradient_transition(
                traj, scaled, epsilon)  # size (N, M, d)
            grad_flatten = grad.flatten(
                start_dim=0, end_dim=1)  # size (N*M, d)

            # evaluate score network
            # size (N*M, d)
            traj_flatten = traj[:, 1:(M+1), :].flatten(start_dim=0, end_dim=1)
            score = score_net(timesteps_flatten, traj_flatten,
                              initial_states_flatten)  # size (N*M, d)

            # compute loss function
            # need to extend this for non-uniform stepsizes
            loss = F.mse_loss(score, grad_flatten)

            # backpropagation
            loss.backward()

            # optimization step and zero gradient
            optimizer.step()
            optimizer.zero_grad()

            # update parameters using exponential moving average
            ema_update(ema_parameters, score_net, ema_momentum)

            # iteration counter
            current_loss = loss.item()
            loss_values[i] = current_loss
            if (i == 0) or ((i+1) % 50 == 0):
                print("Optimization iteration:", i+1, "Loss:", current_loss)

        # use exponential moving average parameters in score network
        ema_copy(ema_parameters, score_net)

        # output
        output = {'net': score_net, 'loss': loss_values}

        return output

    def my_learn_full_score_transition(self, simulate_initial_state, simulate_terminal_state, epsilon, minibatch, num_initial_per_batch, num_iterations, learning_rate, ema_momentum, scheduler=None, device='cpu', drift=None, full_model=False, checkpoint=None):
        """
        Learn full approximation of score transition using score matching.

        Parameters
        ----------
        simulate_initial_state : function returning simulated initial conditions

        simulate_terminal_state : function returning simulated terminal conditions

        epsilon : positive constant to enforce initial constraint 

        minibatch : number of mini-batch samples desired

        num_initial_per_batch : number of initial states per batch

        num_iterations : number of optimization iterations (divisible by num_batches)

        learning_rate : learning rate of Adam optimizer

        ema_momentum : momentum parameter of exponential moving average update

        Returns
        -------
        output : dict containing    
            net : neural network approximation of full score function of transition density
            loss : value of loss function during learning
        """

        d = self.d
        M = self.num_steps
        N = minibatch * num_initial_per_batch
        timesteps = self.time[1:(M+1)].reshape((1, M, 1)
                                               ).repeat((N, 1, 1))  # size (N, M, 1)
        timesteps_flatten = timesteps.flatten(
            start_dim=0, end_dim=1)  # size (N*M, 1)
        loss_values = torch.zeros(num_iterations)

        if checkpoint is not None:
            score_net = self.load_checkpoint_score(checkpoint, full_model=full_model)['net']
        else:
            score_net = newFullScoreNetwork(self.d)

        # create score network
        # score_net = FullScoreNetwork(dimension=self.d) if not full_model else newFullScoreNetwork(dimension=self.d)
        score_net.to(device).train()
        ema_parameters = ema_register(score_net)

        # optimization
        optimizer = torch.optim.AdamW(score_net.parameters(), lr=learning_rate)

        if scheduler == 'cos':
            scheduler = OneCycleLR(
                optimizer, max_lr=learning_rate, total_steps=num_iterations)
        printt = True
        save = True
        with Progress(
            SpinnerColumn(spinner_name='moon'),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task1 = progress.add_task(
                "[red]Training full_score_transition (lr: X) (loss=X)", total=num_iterations)
            while not progress.finished:
                for i in range(num_iterations):
                    # simulate initial states
                    initial_states = simulate_initial_state(
                        num_initial_per_batch).repeat((minibatch, 1))  # size (N, d)
                    initial_states_repeated = initial_states.reshape(
                        (N, 1, d)).repeat((1, M, 1))  # size (N, M, d)
                    initial_states_flatten = initial_states_repeated.flatten(
                        start_dim=0, end_dim=1)  # size (N*M, d)

                    if drift is not None or full_model:
                        terminal_states = simulate_terminal_state(num_initial_per_batch).repeat((minibatch,1)) # size (N, d)
                    if full_model:
                        terminal_states_repeated = terminal_states.reshape((N,1,d)).repeat((1,M,1)) # size (N, M, d)
                        terminal_states_flatten = terminal_states_repeated.flatten(start_dim = 0, end_dim = 1) # size (N*M, d)

                    # simulate trajectories from diffusion process
                    if drift is not None:
                        simulation_output = self.my_simulate_bridge(initial_states, terminal_states) if drift=='bridge' else self.my_simulate_process(initial_states, terminal_states)
                        # simulation_output = self.my_simulate_process(initial_states, terminal_states)
                        if printt:
                            print(f'learn_{drift}')
                            printt=False
                    else:
                        simulation_output = self.simulate_process(initial_states)
                    # simulation_output = self.my_simulate_process(initial_states, terminal_states)
                    # size (N, M+1, d)
                    traj = simulation_output['trajectories']
                    scaled = simulation_output['scaled_brownian']

                    # evaluate gradient function
                    
                    
                    grad = self.gradient_transition(
                        traj, scaled, epsilon)  # size (N, M, d)
                    if save:
                        import numpy as np
                        np.save('grad.npy', grad.detach().cpu().numpy())
                    
                    grad_flatten = grad.flatten(
                        start_dim=0, end_dim=1)  # size (N*M, d)

                    # evaluate score network
                    # size (N*M, d)
                    traj_flatten = traj[:, 1:(
                        M+1), :].flatten(start_dim=0, end_dim=1)
                    timesteps_flatten = timesteps_flatten.to(device)
                    traj_flatten = traj_flatten.to(device)
                    initial_states_flatten = initial_states_flatten.to(device)
                    grad_flatten = grad_flatten.to(device)
                    # size (N*M, d)
                    if full_model:
                        terminal_states_flatten = terminal_states_flatten.to(device)
                        score = score_net(timesteps_flatten,
                                            traj_flatten, initial_states_flatten, terminal_states_flatten)
                    else:
                        score = score_net(timesteps_flatten,
                                        traj_flatten, initial_states_flatten)

                    # compute loss function
                    # need to extend this for non-uniform stepsizes
                    loss = F.mse_loss(score, grad_flatten)

                    # backpropagation
                    loss.backward()

                    # optimization step and zero gradient
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    # update parameters using exponential moving average
                    ema_update(ema_parameters, score_net, ema_momentum)

                    # iteration counter
                    current_loss = loss.item()
                    loss_values[i] = current_loss
                    # if (i == 0) or ((i+1) % 50 == 0):
                    #     print("Optimization iteration:", i+1, "Loss:", current_loss)
                    cur_lr = optimizer.param_groups[-1]['lr']
                    progress.update(task1, advance=1, description="[red]Training whole dataset (lr: %2.5f) (loss=%2.5f)" % (
                        cur_lr, current_loss))

        # use exponential moving average parameters in score network
        ema_copy(ema_parameters, score_net)

        # output
        output = {'net': score_net, 'loss': loss_values}

        return output

    def gradient_marginal(self, trajectories, scaled_brownian, epsilon):
        """
        Evaluate gradient function needed in score matching to learn score of marginal (diffusion bridge) density.

        Parameters
        ----------
        trajectories : realizations of time-discretized process (N, M+1, d)

        scaled_brownian : scaled brownian increments (N, M, d)

        epsilon : positive constant to enforce initial constraint 

        Returns
        -------    
        grad : gradient function evaluations (N, M, d)
        """
        N = trajectories.shape[0]
        M = self.num_steps
        grad = torch.zeros(N, M, self.d)
        XT = trajectories[:, M, :]

        for m in range(M, 0, -1):
            Z_next = trajectories[:, m-1, :,]
            if (m == 1):
                # fudging a little here because of singularity
                t_next = 0.25 * self.stepsizes[m-1]
            else:
                t_next = self.time[m-1]
            grad[:, m-1, :] = scaled_brownian[:, m-1, :] - \
                epsilon * self.invSigma * (XT - Z_next) / t_next

        return grad

    def my_learn_score_marginal(self, score_transition_net, simulate_initial_state, simulate_terminal_state, epsilon, minibatch, num_initial_per_batch, num_iterations, learning_rate, ema_momentum, num_workers=0, device='cpu', scheduler=None, checkpoint=None):
        """
        Learn approximation of score of marginal (diffusion bridge) density using score matching.

        Parameters
        ----------
        score_transition_net : neural network approximation of score function of transition density

        initial_state : initial condition of size d

        terminal_state : terminal condition of size d

        epsilon : positive constant to enforce initial constraint 

        minibatch : number of mini-batch samples desired

        num_iterations : number of optimization iterations (divisible by num_batches)

        learning_rate : learning rate of Adam optimizer

        ema_momentum : momentum parameter of exponential moving average update

        Returns
        -------    
        score_marginal_net : neural network approximation of score function of marginal (diffusion bridge) density

        loss_values : value of loss function during learning
        """
        d = self.d
        M = self.num_steps
        N = num_initial_per_batch
        timesteps = self.time[0:M].reshape(
            (1, M, 1)).repeat((N, 1, 1))  # size (N, M, 1)
        timesteps_flatten = timesteps.flatten(
            start_dim=0, end_dim=1)  # size (N*M, 1)
        loss_values = torch.zeros(num_iterations)

        # create score network
        if checkpoint is not None:
            score_net = self.load_checkpoint_marginal(checkpoint)['net']
        else:
            score_net = newFullScoreNetwork(self.d)
        score_net = score_net.to(device)
        ema_parameters = ema_register(score_net)

        # optimization
        optimizer = torch.optim.Adam(score_net.parameters(), lr=learning_rate)
        num_batches = 10
        num_samples = num_batches * N
        num_repeats = int(num_iterations / num_batches)
        # iteration = 1
        if scheduler == 'cos':
            scheduler = OneCycleLR(
                optimizer, max_lr=learning_rate, total_steps=num_iterations*num_batches)

        score_transition_net = score_transition_net.to(device)

        ds = bridgeBackwardsDataset(self, score_transition_net, simulate_initial_state, simulate_terminal_state,
                                    num_repeats, num_initial_per_batch, minibatch, device, epsilon, num_samples, N, d, M)
        dl = DataLoader(ds, batch_size=1, num_workers=num_workers)
        iter_nums = 0
        with Progress(
            SpinnerColumn(spinner_name='moon'),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task1 = progress.add_task(
                "[red]Training score marginal (lr: X) (loss=X)", total=num_iterations)
            while not progress.finished:
                for trajectories, scaled_brownian in dl:
                    trajectories = trajectories[0]
                    scaled_brownian = scaled_brownian[0]
                    for j in range(num_batches):
                        # get minibatch of trajectories
                        # size (N, M+1, d)
                        traj = trajectories[(j * N):((j+1) * N), :, :]
                        scaled = scaled_brownian[(j * N):((j+1) * N), :, :]

                        # evaluate gradient function
                        grad = self.gradient_marginal(
                            traj, scaled, epsilon)  # size (N, M, d)
                        grad_flatten = grad.flatten(
                            start_dim=0, end_dim=1)  # size (N*M, d)

                        # evaluate score network
                        traj_flatten = traj[:, 0:M, :].flatten(
                            start_dim=0, end_dim=1)  # size (N*M, d)

                        initial_states = traj[:, :1, :]  # size (N, d)
                        initial_states_repeated = initial_states.reshape(
                            (N, 1, d)).repeat((1, M, 1))  # size (N, M, d)
                        initial_states_flatten = initial_states_repeated.flatten(
                            start_dim=0, end_dim=1)  # size (N*M, d)

                        terminal_states = traj[:, -1:, :]  # size (N, d)
                        terminal_states_repeated = terminal_states.reshape(
                            (N, 1, d)).repeat((1, M, 1))  # size (N, M, d)
                        terminal_states_flatten = terminal_states_repeated.flatten(
                            start_dim=0, end_dim=1)  # size (N*M, d)

                        timesteps_flatten = timesteps_flatten.to(device)
                        traj_flatten = traj_flatten.to(device)
                        grad_flatten = grad_flatten.to(device)
                        initial_states_flatten = initial_states_flatten.to(
                            device)
                        terminal_states_flatten = terminal_states_flatten.to(
                            device)

                        score = score_net(
                            timesteps_flatten, traj_flatten, initial_states_flatten, terminal_states_flatten)  # size (N*M, d)
                        # score = score_net(timesteps_flatten, traj_flatten, initial_states_flatten) # size (N*M, d)

                        # compute loss function
                        # need to extend this for non-uniform stepsizes
                        loss = F.mse_loss(score, grad_flatten)

                        # backpropagation
                        loss.backward()

                        # optimization step and zero gradient
                        optimizer.step()
                        optimizer.zero_grad()

                        if scheduler is not None:
                            scheduler.step()

                        # update parameters using exponential moving average
                        ema_update(ema_parameters, score_net, ema_momentum)

                        # iteration counter
                        current_loss = loss.item()
                        # loss_values[iteration-1] = current_loss
                        # iteration += 1

                    cur_lr = optimizer.param_groups[-1]['lr']
                    progress.update(task1, advance=1, description="[red]Training score marginal (lr: %2.5f) (loss=%2.5f)" % (
                        cur_lr, current_loss))
                    # if iter_nums % 10 == 0:
                    #     print("Optimization iteration:", iter_nums, "Loss:", current_loss)
                    #     torch.save(score_net.state_dict(), f'temp/score_marginal_net_{iter_nums}.pth')
                    iter_nums += 1

        # use exponential moving average parameters in score network
        ema_copy(ema_parameters, score_net)

        # output
        output = {'net': score_net, 'loss': loss_values}

        return output

    def learn_score_marginal(self, score_transition_net, initial_state, terminal_state, epsilon, minibatch, num_iterations, learning_rate, ema_momentum, full_score=False):
        """
        Learn approximation of score of marginal (diffusion bridge) density using score matching.

        Parameters
        ----------
        score_transition_net : neural network approximation of score function of transition density

        initial_state : initial condition of size d

        terminal_state : terminal condition of size d

        epsilon : positive constant to enforce initial constraint 

        minibatch : number of mini-batch samples desired

        num_iterations : number of optimization iterations (divisible by num_batches)

        learning_rate : learning rate of Adam optimizer

        ema_momentum : momentum parameter of exponential moving average update

        Returns
        -------    
        score_marginal_net : neural network approximation of score function of marginal (diffusion bridge) density

        loss_values : value of loss function during learning
        """

        M = self.num_steps
        N = minibatch
        timesteps = self.time[0:M].reshape(
            (1, M, 1)).repeat((N, 1, 1))  # size (N, M, 1)
        timesteps_flatten = timesteps.flatten(
            start_dim=0, end_dim=1)  # size (N*M, 1)
        loss_values = torch.zeros(num_iterations)

        # create score network
        score_net = ScoreNetwork(dimension=self.d)
        ema_parameters = ema_register(score_net)

        # optimization
        optimizer = torch.optim.Adam(score_net.parameters(), lr=learning_rate)
        num_batches = 10
        num_samples = num_batches * N
        num_repeats = int(num_iterations / num_batches)
        iteration = 1
        for i in range(num_repeats):
            # simulate trajectories from approximate diffusion bridge process backwards
            with torch.no_grad():
                simulation_output = self.simulate_bridge_backwards(
                    score_transition_net, initial_state, terminal_state, epsilon, num_samples, modify=True, full_score=full_score)
            trajectories = simulation_output['trajectories']
            scaled_brownian = simulation_output['scaled_brownian']

            for j in range(num_batches):
                # get minibatch of trajectories
                traj = trajectories[(j * N):((j+1) * N), :,
                                    :]  # size (N, M+1, d)
                scaled = scaled_brownian[(j * N):((j+1) * N), :, :]

                # evaluate gradient function
                grad = self.gradient_marginal(
                    traj, scaled, epsilon)  # size (N, M, d)
                grad_flatten = grad.flatten(
                    start_dim=0, end_dim=1)  # size (N*M, d)

                # evaluate score network
                traj_flatten = traj[:, 0:M, :].flatten(
                    start_dim=0, end_dim=1)  # size (N*M, d)
                score = score_net(timesteps_flatten,
                                  traj_flatten)  # size (N*M, d)

                # compute loss function
                # need to extend this for non-uniform stepsizes
                loss = F.mse_loss(score, grad_flatten)

                # backpropagation
                loss.backward()

                # optimization step and zero gradient
                optimizer.step()
                optimizer.zero_grad()

                # update parameters using exponential moving average
                ema_update(ema_parameters, score_net, ema_momentum)

                # iteration counter
                current_loss = loss.item()
                loss_values[iteration-1] = current_loss
                if (iteration == 1) or (iteration % 50 == 0):
                    print("Optimization iteration:",
                          iteration, "Loss:", current_loss)
                iteration += 1

        # use exponential moving average parameters in score network
        ema_copy(ema_parameters, score_net)

        # output
        output = {'net': score_net, 'loss': loss_values}

        return output

    def load_checkpoint_score(self, filename, full_model=False):
        """
        Load checkpoint of score network.

        Parameters
        ----------
        filename : filename of checkpoint

        Returns
        -------    
        net : neural network approximation of score function of transition density
        """

        # load checkpoint
        checkpoint = torch.load(filename)

        # create score network
        if full_model:
            score_net = newFullScoreNetwork(dimension=self.d)
        else:
            score_net = FullScoreNetwork(dimension=self.d)
        score_net.load_state_dict(checkpoint)

        # output
        output = {'net': score_net}

        return output

    def load_checkpoint_marginal(self, filename):
        """
        Load checkpoint of score network.

        Parameters
        ----------
        filename : filename of checkpoint

        Returns
        -------    
        net : neural network approximation of score function of marginal (diffusion bridge) density
        """

        # load checkpoint
        checkpoint = torch.load(filename)

        # create score network
        score_net = newFullScoreNetwork(dimension=self.d)
        score_net.load_state_dict(checkpoint)

        # output
        output = {'net': score_net}

        return output
