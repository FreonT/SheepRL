import threading
import typing

import torch

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

import pickle

def save_data(flags, batch, step):
    file_name = flags.save_data_name + flags.env + "_step_" + str(step) + ".pickle"
    with open(file_name, "wb") as f:
        pickle.dump(batch, f)

def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optim,
    step,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        save_data(flags, batch, step)
        return {}

def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

class optimizer:
    def __init__(self, flags, model):
        self.optimizer = None
        self.scheduler = None
        