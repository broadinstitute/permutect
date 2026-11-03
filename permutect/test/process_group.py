import os
import socket
import tempfile

import torch
import torch.distributed as dist


def run_in_process_group(world_size: int, fn, *args) -> list:
    """
    Run fn(*args) in world_size new processes that form a gloo process group, as under torchrun, and return their
    results in order of rank.  fn, args and the results must be picklable, so fn must be defined at the top level of a
    module.
    """
    with tempfile.TemporaryDirectory() as results_dir:
        port = find_free_port()
        torch.multiprocessing.spawn(run_rank, args=(world_size, port, results_dir, fn, args), nprocs=world_size)
        return [torch.load(os.path.join(results_dir, f"{rank}.pt"), weights_only=False) for rank in range(world_size)]


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_rank(rank: int, world_size: int, port: int, results_dir: str, fn, args):
    # the environment variables that torchrun sets
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo")
    try:
        result = fn(*args)
    finally:
        dist.destroy_process_group()
    torch.save(result, os.path.join(results_dir, f"{rank}.pt"))
