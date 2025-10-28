import logging
import os
import sys

import torch.distributed as dist

from rllava.utils.logger.aggregate_logger import print_rank_0


root_logger = None


def logger_setting(save_dir=None, level=logging.INFO):
    global root_logger
    if root_logger is not None:
        return root_logger
    else:
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, 'log.txt')
            if not os.path.exists(save_file):
                os.system(f"touch {save_file}")
            fh = logging.FileHandler(save_file, mode='a')
            fh.setLevel(level)
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)
            return root_logger
        
def log(*args):
    global root_logger
    local_rank = dist.get_rank()
    if local_rank == 0:
        root_logger.info(*args)

def log_trainable_params(model):
    def _global_numel(param):
        ds_numel = getattr(param, "ds_numel", None)
        if ds_numel is not None:
            return int(ds_numel)
        return param.numel()

    try:
        total_params = sum(_global_numel(p) for p in model.parameters())
        total_trainable_params = sum(_global_numel(p) for p in model.parameters() if p.requires_grad)
    except Exception:
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log(f'Total Parameters: {total_params}, Total Trainable Parameters: {total_trainable_params}')
    log(f'Trainable Parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank_0(f"{name}: {_global_numel(param)} parameters")
