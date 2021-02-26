# Experiments 

## Flags in command line
### hyper parameters 
when running the following commands:
- -n: number of policy training iterations

- -b: batch size (number of state-action pairs sampled while acting according to the current policy at each iteration).
- -lr: learning rate

###  flags
- -rtg : Flag: if present, sets reward_to_go=True. Otherwise, reward_to_go=False by default.

- -dsa: do not standardize the advantage values.  If present, sets standardize_advantages to False. Otherwise, by default, standardizes advantages to have a mean of zero and standard deviation of one.

- --video log freq -1 Don't generate video (by default) while debugging; To generate videos of the policy, remove this flag



- ep_len : Episode Length 
    
    Note: eval batch size shouldbe greater than ep len, such that you’re collecting multiple rollouts when evaluating the performance of your trained policy. For example, if ep len is 1000 and eval batch size is 5000, then you’ll be collecting approximately 5 trajectories (maybe more if any of them terminate early).



### others
- --exp_name : Name for experiment, which goes into the name for the data logging directory.




# Experiment 1 (CartPole)
## command line
### 1.1 small batch
`python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
    -dsa --exp_name q1_sb_no_rtg_dsa`
 
`python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
    -rtg -dsa --exp_name q1_sb_rtg_dsa`
    
`python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
    -rtg --exp_name q1_sb_rtg_na`
### 1.2 large batch   
`python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
    -dsa --exp_name q1_lb_no_rtg_dsa`
    
`python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
    -rtg -dsa --exp_name q1_lb_rtg_dsa`
    
`python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
    -rtg --exp_name q1_lb_rtg_na`
   
## questions
– Which value estimator has better performance without advantage-standardization: the trajectory-centric one, or the one using reward-to-go?

    using reward-to-go converges much faster
     
– Did advantage standardization help?
    
    yes, during the convergence process, the reward is more stable

– Did the batch size make an impact?
    
    yes, large batch can converge to maxinum score 200 


# Experiment 2 (InvertedPendulum)
## command line


### (if prefer larger learning rate)
`python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 1000 -lr 0.08 -rtg \
--exp_name q2_b1000_r0.08`

###  (if prefer smaller batch size)
`python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 300 -lr 0.01 -rtg \
--exp_name q2_b300_r0.01`

Note: it is hard to achieve largest lr and smallest batch size at the same time.


add baseline 
`python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 300 -lr 0.01 -rtg \
--nn_baseline --exp_name q2_b300_r0.01_rtg_nnbaseline`



# Experiment 3 (LunarLander)
## command line

### this one leads to error ??
`python cs285/scripts/run_hw2.py \
--env_name LunarLanderContinuous-v2 --ep_len 1000
--discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 \
--reward_to_go --nn_baseline --exp_name q3_b40000_r0.005`


### work
`python cs285/scripts/run_hw2.py --env_name LunarLanderContinuous-v2 \
--ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 -rtg \
--nn_baseline --exp_name q3_b40000_r0.005`






# Experiment 4 (HalfCheetah)
## command line
Search over batch sizes b ∈ [10000, 30000, 50000]
and learning rates r ∈ [0.005, 0.01, 0.02] to replace <b> and <r> below.
    
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b <b> -lr <r> -rtg --nn_baseline \
--exp_name q4_search_b<b>_lr<r>_rtg_nnbaseline
    

        
    
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.005 -rtg --nn_baseline \
--exp_name q4_search_b10000_lr0.005_rtg_nnbaseline`
    
    
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.01 -rtg --nn_baseline \
--exp_name q4_search_b10000_lr0.01_rtg_nnbaseline`
    
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 -rtg --nn_baseline \
--exp_name q4_search_b10000_lr0.02_rtg_nnbaseline`
    
    
    
    
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.005 -rtg --nn_baseline \
--exp_name q4_search_b30000_lr0.005_rtg_nnbaseline`
    
    
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.01 -rtg --nn_baseline \
--exp_name q4_search_b30000_lr0.01_rtg_nnbaseline`
    
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --nn_baseline \
--exp_name q4_search_b30000_lr0.02_rtg_nnbaseline`

    
    
    
    
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.005 -rtg --nn_baseline \
--exp_name q4_search_b50000_lr0.005_rtg_nnbaseline`
    
    
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.01 -rtg --nn_baseline \
--exp_name q4_search_b50000_lr0.01_rtg_nnbaseline`
    
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline \
--exp_name q4_search_b50000_lr0.02_rtg_nnbaseline`

    
    
    

The optimal values b* and r* that I found are -b 50000 -lr 0.02 , and use them to run the following commands:
    
    
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 \
--exp_name q4_b50000_r0.02
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg \
--exp_name q4_b50000_r0.02_rtg
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 --nn_baseline \
--exp_name q4_b50000_r0.02_nnbaseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline \
--exp_name q4_b50000_r0.02_rtg_nnbaseline`


    
### questions
Describe in words how the batch size and learning rate affected task performance
    
    In general, larger batch size and learning rate leads to higher average return in the end.
    
    
    Note: 
- small batch size with large learning rate is unstable.
- the performance is affected by batch size and learning rate at the same time. If one of Hyper parameter is not tuned properly, it will limit the overall performance even though the other parameter is tuned properly.
 
       
  
