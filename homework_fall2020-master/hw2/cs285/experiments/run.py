import argparse
import random
import json
import os
from cycler import cycler
import sys
import os.path as osp
import subprocess
import shlex
import time
import numpy as np
from collections import defaultdict

import pickle


#_______helper function_______________________________________________________

def run_multiple_commands(commands, logpaths, metalogpath):
    import subprocess
    import shlex
    import time
    assert len(commands) == len(logpaths)
    logfiles = []
    processes = []
    try:
        metalogfile = open(metalogpath, 'w', buffering=1)
        for command, logpath in zip(commands, logpaths):
            logfile = open(logpath, 'w', buffering=1)
            logfiles.append(logfile)
            p = subprocess.Popen(shlex.split(command), stdout=logfile)
            processes.append(p)
            print(f'{command}\n=> {logpath}', file=metalogfile)

        while len(processes) > 0:
            terminated = []
            for i in range(len(processes)):
                return_code = processes[i].poll()
                if return_code is not None:
                    terminated.append(i)
            for i in reversed(terminated):
                print('TERMINATED: {}'.format(commands[i]), file=metalogfile)
                del processes[i]
                del commands[i]
                del logpaths[i]
                logfiles[i].close()
                del logfiles[i]
            time.sleep(5)
        print('All processes have terminated. Quitting...')
    finally:
        for file in logfiles:
            file.close()

            
#_______Experiments___________________________________________________________

#_______Experiments_1___________________
def run_exp_1():
    seeds = range(5)
    
    templates = [
        "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa_seed_{seed} --seed {seed}",
        "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa_seed_{seed} --seed {seed}",
        "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na_seed_{seed} --seed {seed}",
        "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa_seed_{seed} --seed {seed}",
        "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa_seed_{seed} --seed {seed}",
        "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na_seed_{seed} --seed {seed}",
    ]
    commands = []
    for template in templates:
        for seed in seeds:
            command = template.format(seed=seed)
            commands.append(command)
    os.makedirs('cs285/scripts/logs', exist_ok=True)
    logfiles = [f'cs285/experiments/logs/{i}.log' for i in range(len(commands))]
    metalogfile = 'cs285/experiments/logs/meta.log'
    run_multiple_commands(commands, logfiles, metalogfile)

    
#_______Experiments__2_________________________________

#get lr and bs from log file name


def parse_exp_2(filename):
    #e.g., q2_b300_r0.02
    split = filename.split('_')
    bs = int(split[1][1:])
    lr = float(split[2][1:])
    return lr, bs



def get_tested_hyper():
    results = []
    for logdir in os.listdir('data'):
        if logdir.startswith('q2'):
            #print(logdir)
            lr, bs = parse_exp_2(logdir)
            results.append((lr, bs))
            print(results)
    
    return results

    

def run_exp_2(SEARCH = True):
    seeds = range(3)
    #search hyper parameters or not :learning rate -lr and batch size -b
    if SEARCH:
        bs_list = [ 2**j for j in range(8,14) ]
        lr_list = [6e-3,7e-3,8e-3,9e-3, 3e-2, 4e-2, 6e-2, 8e-2]
        #bs_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


        
        tested = get_tested_hyper()#Don't run tested hyper parameters 
        print(tested)
        combinations = [(lr, bs) for lr in lr_list for bs in bs_list if (lr, bs) not in tested]
        num = 30 #max number of conbination
        #print('combinations',combinations)
        print('length of combinations',len(combinations))
        
        assert num <= len(combinations)
        

    
        template = "python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b {bs} -lr {lr} -rtg --exp_name q2_b{bs}_r{lr}_seed_{seed} --seed {seed}"
        
        #sample combinations of hyper
        configs_indices = np.random.choice(len(combinations), num, replace=False)
        configs = [combinations[i] for i in configs_indices]
        
        commands = []
        for lr, bs in configs:
            for seed in seeds:
                command = template.format(lr=lr, bs=bs,seed=seed)
                commands.append(command)
        #print(commands)
        
        
        os.makedirs('cs285/scripts/logs', exist_ok=True)
        logfiles = [f'cs285/experiments/logs/{i}.log' for i in range(len(commands))]
        metalogfile = 'cs285/experiments/logs/meta.log'
        run_multiple_commands(commands, logfiles, metalogfile)
        
        
    else:# run optimal hyper
        #seeds = [1] #default seeds is 1
        template = "python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b {bs} -lr {lr} -rtg --exp_name ip_b{bs}_r{lr}_seed_{seed} --seed {seed}"
        lr = 2e-2
        bs = 300
        commands = []
        for seed in seeds:
            command = template.format(lr=lr, bs=bs, seed=seed)
            commands.append(command)
        #subprocess.run(shlex.split(commands[0]))
        
        logfiles = [f'cs285/experiments/logs/{i}.log' for i in range(len(commands))]
        metalogfile = 'cs285/experiments/logs/meta.log'
        run_multiple_commands(commands, logfiles, metalogfile)

        
# _______Experiments__3_________________________________    

def run_exp_3():
    command = 'python cs285/scripts/run_hw2.py --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005 -eb 5000'
    subprocess.run(shlex.split(command))
    

#_______Experiments__4_________________________________


def run_exp_4(SEARCH = False):
    seeds = range(5)
    if SEARCH:
        print('SEARCH')
      
        '''
        
        lr_list = [0.005, 0.01, 0.02]
        bs_list = [10000, 30000, 50000]
        
        
        template = 'python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} --video_log_freq -1 -rtg --nn_baseline --exp_name q4_serach_b{bs}_lr{lr}_rtg_nnbaseline_seed_{seed}'
        
        commands = []
        for lr in lr_list:
            for bs in bs_list:
                for seed in seeds:
                    command = template.format(lr=lr, bs=bs,seed=seed)
                    commands.append(command)
                
        os.makedirs('cs285/scripts/logs', exist_ok=True)
        logfiles = [f'cs285/experiments/logs/{i}.log' for i in range(len(commands))]
        metalogfile = 'cs285/experiments/logs/meta.log'
        run_multiple_commands(commands, logfiles, metalogfile)
       '''
            
    #optimal hyper parameter
    else:
        print('not SEARCH')
        lr = 0.02
        bs = 50000
        '''
        
        templates = ['python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} --seed {seed} \
--exp_name q4_b{bs}_lr{lr}_seed_{seed}',           
'python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} -rtg --seed {seed} \
--exp_name q4_b{bs}_lr{lr}_rtg_seed_{seed}',
'python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} --nn_baseline --seed {seed} \
--exp_name q4_b{bs}_lr{lr}_nnbaseline_seed_{seed}',
'python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} -rtg --nn_baseline --seed {seed} \
--exp_name q4_b{bs}_lr{lr}_rtg_nnbaseline_seed_{seed}'    
        ]
        '''
        
        templates = ['python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} -rtg --seed {seed} \
--exp_name q4_b{bs}_lr{lr}_rtg_seed_{seed}',
'python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} --nn_baseline --seed {seed} \
--exp_name q4_b{bs}_lr{lr}_nnbaseline_seed_{seed}'   
        ]
        
        

        commands = []
        for template in templates:
            for seed in seeds:
                command = template.format(seed=seed,lr=lr, bs=bs)
                commands.append(command)
        
        print(len(commands))
        print(commands)
        
        
        
        logfiles = [f'cs285/experiments/logs/{i}.log' for i in range(len(commands))]
        metalogfile = 'cs285/experiments/logs/meta.log'
        run_multiple_commands(commands, logfiles, metalogfile)







def main():
    assert os.path.isdir('cs285/scripts'), 'Please run this from hw2 root'
    parser = argparse.ArgumentParser()
    parser.add_argument('exp')
    args = parser.parse_args()
    
    exp = {
        'run_exp_1': run_exp_1,
        'run_exp_2': run_exp_2,
        'run_exp_3': run_exp_3,
        'run_exp_4': run_exp_4,
    }
    assert args.exp in exp
    exp[args.exp]()


if __name__ == '__main__':
    main()

