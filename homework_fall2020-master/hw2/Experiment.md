# Experiment 1

* `python cs285/experiments/run.py run_exp_1`


# Experiment 2

* `python cs285/experiments/run.py run_exp_2`


# Experiment 3

* `python cs285/experiments/run.py run_exp_3`


# Experiment 4

* `python cs285/experiments/run.py run_exp_4`


# demo
## Exp 1
`python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
    -rtg --video_log_freq 5 --exp_name demo_q1_lb_rtg_na`
    
## Exp 2  mujoco error  
`python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 4096 -lr 0.01 -rtg \
--video_log_freq 5 --exp_name demo_q2_b4096_r0.01`
## Exp 3    
`python cs285/scripts/run_hw2.py --env_name LunarLanderContinuous-v2 \
--ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 -rtg \
--nn_baseline --video_log_freq 5 --exp_name demo_q3_b40000_r0.005`
## Exp 4
`python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline \
--video_log_freq 5 --exp_name demo_q4_b50000_r0.02_rtg_nnbaseline`