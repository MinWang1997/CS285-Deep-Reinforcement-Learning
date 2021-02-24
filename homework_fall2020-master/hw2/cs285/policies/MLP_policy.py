import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

#add
from cs285.infrastructure import utils

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    
    # query the policy with observation(s) to get selected action(s)
    # TODO: get this from hw1
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
    

        #Args:obs (numpy.ndarray): Observation from environment.
        #Returns:numpy.ndarray: Predicted action by forward NN. Note:return numpy array instead of tensor, may because np is more general
        
        action_distribution = self.forward(ptu.from_numpy(observation))
        action = action_distribution.sample()  
        
        return ptu.to_numpy(action)
     

    
    
    
# update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

        
        
        
        
        
    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate through it. For example, you can return a torch.FloatTensor. 
    #You can also return more flexible objects, such as a `torch.distributions.Distribution` object(and then sample action in get_action function). It's up to you!
    # TODO: get this from hw1
    
    def forward(self, observation: torch.FloatTensor): 
    # TODO: get this from hw1
    # building a feedforward neural network policy and return action distribution 
    #logits_na and mean_net (for discrete/continues case), are the same network and both are from build_mlp method, to construct log-probabilities and probabilities for actions, and then we can use the get_action function samples actions based on it.
    
        if self.discrete: # discrete -> categorical policy
            action_distribution = distributions.Categorical(logits=self.logits_na(observation))
       
        else:  # continues -> Multivariate Normal
            action_distribution = distributions.MultivariateNormal( loc = self.mean_net(observation), scale_tril = torch.diag(torch.exp(self.logstd)))
        return action_distribution
    
    


    
    
    
#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages=None, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)#advantages=(Q_t - b_t)

        # TODO: compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE is the expectation over collected trajectories of: sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        #We max gradient of Cumulative rewards J(to take a step towards steepest direction) instead of J itself because it is hard to max J directly.
                 
        # HINT2: you will want to use the `log_prob` method on the distribution returned # by the `self.forward` method above
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss
        
        #compute log pi(a_t|s_t) 
        log_pi = self.forward(observations).log_prob(actions) 
       

        #use Back propagation tool to help us to compute policy gradient: the pseudo-loss is policy gradient without gradient -> the gradient of pseudo-loss is equal to the policy gradient.
        #the pseudo-loss is a weighted maximum likelihood, where the weight is advantages (reward to go with baseline), i.e., Q = q_value-baseline
        # use Minus - transform Gradient decent -> accent
        

        #it doesn't matter to use double mean or sum for tensor instead of mean and sum, because optimaser will adapt to it.
        #compute pseudo-loss sum_{t=0}^{T-1} [log pi(a_t|s_t) * (q_t - b_t)]
        loss = torch.neg(torch.mean(torch.mul(log_pi, advantages)))


        # TODO: optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            #most common choice of baseline is the on-policy value function V^pi(s_t) i.e., average return an agent gets if it starts in state s_t (Reward to go, i.e.,q_value)
            
            # TODO: normalize the q_values to have a mean of zero and a standard deviation of one
            '''
            why normalize q_values first as targets of baseline?
            '''
            ## HINT: there is a `normalize` function in `infrastructure.utils`
            targets = utils.normalize(q_values, np.mean(q_values), np.std(q_values))
            targets = ptu.from_numpy(targets)

            # TODO: use the `forward` method of `self.baseline` to get baseline predictions 
            
            #self.baseline is approximated by a neural network, which is updated concurrently with the policy
            baseline_predictions = self.baseline.forward(observations).squeeze()
            
            ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            ## [ N ] versus shape [ N x 1 ]
            ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
            assert baseline_predictions.shape == targets.shape
            
            # TODO: compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
            # HINT: use `F.mse_loss`
            
            #simplest method for learning baseline is minimize MSE.
            baseline_loss = self.baseline_loss(baseline_predictions, targets)

            # TODO: optimize `baseline_loss` using `self.baseline_optimizer`
            # HINT: remember to `zero_grad` first
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }
        return train_log

    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]

