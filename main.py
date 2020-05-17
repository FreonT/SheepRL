import hydra
from omegaconf import DictConfig

import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch

from env_wrapper import create_env
from run_threads import train, test


def init_flags(flags):
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    
    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    flags.device = None
    if flags.cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = str(torch.device("cuda"))
    else:
        logging.info("Not using CUDA.")
        flags.device = str(torch.device("cpu"))
    
    return flags

@hydra.main(config_path="config.yaml")
def main(flags : DictConfig):
    flags = init_flags(flags)
    env = create_env(flags)
    flags.observation_shape = env.observation_space.shape

    if flags.model == "Vanilla":
        from algos.vanilla.learn import learn, create_buffers, optimizer
        flags.action_shape = env.action_space.n
        if flags.ObsType == "State":
            from algos.vanilla.model import FCNet
            Net = FCNet
        elif flags.ObsType == "Image" or flags.ObsType == "Atari":
            from algos.vanilla.model import AtariNet
            Net = AtariNet

    elif flags.model == "SAC":
        if flags.ActionType == "Continuous":
            from algos.sac.learn import learn, create_buffers, optimizer
            from algos.sac.model import SACNet
            flags.action_shape = env.action_space.shape[0]
            Net = SACNet
        elif flags.ActionType == "Discrete":
            from algos.sac_discrete.learn import learn, create_buffers, optimizer
            from algos.sac_discrete.model import SACNet
            flags.action_shape = env.action_space.n
            Net = SACNet

    elif flags.model == "SLAC":
        if flags.ActionType == "Continuous":
            from algos.slac.learn import learn, create_buffers, optimizer
            from algos.slac.model import SACNet
            flags.action_shape = env.action_space.shape[0]
            Net = SACNet
        elif flags.ActionType == "Discrete":
            from algos.slac_discrete.learn import learn, create_buffers, optimizer
            from algos.slac_discrete.model import SACNet
            flags.action_shape = env.action_space.n
            flags.observation_shape = env.observation_space.shape
            Net = SACNet
    
    elif flags.model == "Random":
        if flags.ActionType == "Continuous":
            print("TOOD")

        elif flags.ActionType == "Discrete":
            from algos.random_agant.model import RandomAgent_discrete
            from algos.random_agant.learn import learn, create_buffers, optimizer
            
            flags.action_shape = env.action_space.n
            flags.observation_shape = env.observation_space.shape
            Net = RandomAgent_discrete

    env.close()

    if flags.mode == "train":
        train(flags, Net, learn, create_buffers, optimizer)
    else:
        test(flags, Net)


if __name__ == "__main__":
    main()
