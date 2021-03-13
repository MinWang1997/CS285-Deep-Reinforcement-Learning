# presentation

## Outline problem + solution

### policy gradient 
-> REINFORCE

### variance reduction 
#### reward-to-go 
    the policy cannot affect rewards in the past. When sum up the rewards, doesn't include the rewards achieved prior to the time step at which the policy is being queried.


    
#### discounting
    Multiplying a discount factor γ to the rewards can be interpreted as encouraging the agent to focus more on the rewards that are closer in time, and less on the rewards that are further in the future.

#### baseline
    subtract a baseline (that is a constant with respect to τ ) from the sum of rewards



## Difficulties you encountered + how you solved them
- Feel lost to implement an algorithm in a lastest paper -> start with simple, try to finish assignment of CS285, which provide a structure way to learn reinforcement learning. 
- Linux server hard drive limit for students and MuJoCo configuration failure on Linux -> use my own laptop
- The policy doesn't perform as expected: In experiment 4, the run with both reward-to-go and the baseline should achieve an average score close to 200, but it is only about 150 -> debug and run experiments over multiple random seeds to compute average
- When I run experiment 3 and specify -n 100 (Number of iterations), my program iterates 199 times and then raise error zsh command not found --discount -> use simple environment and different hyper parameters to test if implemention is correct.
- The learning curve of baseline doesn't increase average return -> run experiments on different random seeds and plot the mean
- It is hard to get the optimal hyper-parameter in experiment 2 by simple grid search -> focus on how batch size and learning rate affect return instead and try to get a reasonable hyper-parameter setting instead of perfect one
- After data virtualization, there are something wrong or I need more data so that I have to run all experiments again and again -> simplify code by define commonly used function and automate several processes such as runing experiments, naming, reading data and plot.
 

## Future work
- check and debug the correctness of implementation of advantages standardization andreward-to-go.
- automate hyper-parameter search
- Implement paralleling of sample collectionacross multiple threads and compare the difference in training time.
- Implement GAE-λ for advantage estimation to speed up training
- Compare single-step PG and multi-step PG
- Implement   more   complex   algorithm  i.e., SRVR-PG-PE
- Integrate policy gradient into the ADVISER framework



## Short demo video of the implementation


### Exp 1
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

### Exp 2
Due to Mujoco error, there is no camera for track. Therefore, I cannot make demo video for this experiment.

### Exp 3
This environment requires the half-cheetah to learn to run forward or backward.

At each time step the half-cheetah receives a signal composed of a control cost and a reward equal to its average velocity in the direction of the plane. The tasks are Bernoulli samples on {-1, 1} with probability 0.5, where -1 indicates the half-cheetah should move backward and +1 indicates the half-cheetah should move forward. The velocity is calculated as the distance (in the target direction) of the half-cheetah's torso position before and after taking the specified action divided by a small value dt.

### Exp 4

This environment requires the half-cheetah to learn to run forward or backward. At each time step the half-cheetah receives a signal composed of a control cost and a reward equal to its average velocity in the direction of the plane. The tasks are Bernoulli samples on {-1, 1} with probability 0.5, where -1 indicates the half-cheetah should move backward and +1 indicates the half-cheetah should move forward. The velocity is calculated as the distance (in the target direction) of the half-cheetah's torso position before and after taking the specified action divided by a small value dt.