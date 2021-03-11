# 10-minute presentation

## Outline problem + solution
### policy gradient -> REINFORCE


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
- The policy doesn't perform as expected: In experiment 4, the run with both reward-to-go and the baseline should achieve an average score close to 200, but it is only about 150 -> debug
- When I run experiment 3 and specify -n 100 (Number of iterations), my program iterates 199 times and then raise error zsh command not found --discount -> use simple environment and different hyper parameters to test if implemention is correct.
- The learning curve of baseline doesn't increase average return -> run experiments on different random seeds and plot the mean
- After data virtualization, there are something wrong or I need more data so that I have to run all experiments again and again -> simplify code by define commonly used function and automate several processes such as runing experiments, naming, reading data and plot.
 

## Future work
- check and debug the correctness ofimplementation of advantages standardization andreward-to-go.
- automate hyper-parameter search
- Implement paralleling of sample collectionacross multiple threads and compare the dif-ference in training time.
- Implement GAE-λ for advantage estimationto speed up training•Compare single-step PG and multi-step PG
- Implement   more   complex   algorithm  i.e., SRVR-PG-PE
- Integrate policy gradient into the ADVISER framework



## Short demo video of the implementation
