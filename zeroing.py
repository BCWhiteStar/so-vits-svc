import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = optimizer.state_dict()
    new_state_dict["param_groups"][0]["lr"] = learning_rate
    new_state_dict["param_groups"][0]["initial_lr"] = learning_rate
    optimizer.load_state_dict(new_state_dict)
    print(optimizer.state_dict()["param_groups"][0]["lr"])
    # torch.save({'model': state_dict,
    #             'iteration': iteration,
    #             'optimizer': optimizer.state_dict(),
    #             'learning_rate': learning_rate}, checkpoint_path)


def run(rank, n_gpus, hps):
    # for pytorch on win, backend use gloo
    dist.init_process_group(backend='gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus,
                            rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank])  # , find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank])

    print(optim_g)
    try:
        utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                              optim_g, False)
        utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                              optim_d, False)
    except:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    print(optim_g)
    epoch_str = 1
    global_step = 0
    save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch_str,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
    save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch_str,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))


if __name__ == "__main__":
    main()
