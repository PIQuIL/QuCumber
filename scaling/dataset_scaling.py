"""
Run TFIM Scaling experiments on GRAHAM
"""

import click
import numpy as np
from observables import TFIMChainEnergy, TFIMChainMagnetization
from rbm.callbacks import (ModelSaver,
                           ComputeMetrics,
                           VarianceBasedEarlyStopping)
from rbm import RBM
import torch
from pathlib import Path
import re
import os.path


def load_train(N, h):
    N = int(N)
    h = float(h)
    data = np.loadtxt(
        f"/home/ejaazm/projects/def-rgmelko/ejaazm/"
        f"tfim1d/datasets/tfim1d_N{N}_h{h:.2f}_train.txt")
    return data.astype('float32')


CB_PERIOD = 100
SAVE_PATH = ("/home/ejaazm/scratch/"
             "tfim1d_dataset_scaling/model_snapshots")
RBM_NAME = "tmp"  # should get overwritten


NUM_GIBBS_STEPS_OBS = 100
NUM_SAMPLES = int(1e5)


POSSIBLE_CONFIGS = []
for N in [16, 32, 64, 128, 256, 512]:
    for h in [0.5, 1.0, 1.5]:
        for dataset_size in range(10000, 210000, 10000):
            for rep in range(5):
                POSSIBLE_CONFIGS.append({
                    "N": N,
                    "h": h,
                    "dataset_size": dataset_size,
                    "rep": rep
                })


def extract_config(path):
    px = Path(path)
    config = {
        "N": None,
        "h": None,
        "dataset_size": None,
        "rep": None
    }

    for part in px.parts:
        st = part.split("_")[-1]
        for k in config.keys():
            if part.startswith(k):
                config[k] = st
                break
    
    config["N"] = int(config["N"])
    config["h"] = float(config["h"])
    config["dataset_size"] = int(config["dataset_size"])
    config["rep"] = int(config["rep"])

    return config


def make_rbm_path(N, h, dataset_size, rep):
    return f"N_{N}/h_{h:.2f}/dataset_size_{dataset_size}/rep_{rep}/"


def scan_for_incomplete():
    px = Path(SAVE_PATH)
    all_runs = set(map(lambda p: p.parent, px.glob("./**/epoch_*.params")))
    all_complete = set(map(lambda p: p.parent, px.glob("./**/done")))
    return all_runs - all_complete


GET_NUMS_REGEX = re.compile(r'\d+')  # match all numerical chars


def get_epoch_num(p: Path):
    return int(GET_NUMS_REGEX.findall(p.name)[0])


def get_last_epoch_params(path):
    epoch_paths = Path(path).glob("./epoch_*.params")
    return max(epoch_paths, key=get_epoch_num)


def get_all_epoch_params_except_last(path):
    epoch_paths = Path(path).glob("./epoch_*.params")
    last_epoch_path = max(epoch_paths, key=get_epoch_num)
    return set(epoch_paths) - {last_epoch_path}


def init(N: int, h: float, dataset_size: int, rep: int):
    data = load_train(N, h)

    rs = np.random.RandomState(int(N * h * dataset_size * rep))
    data = data[rs.choice(len(data), dataset_size, replace=False)]

    model = RBM(num_visible=N, num_hidden=N, seed=rep)

    rbm_path = make_rbm_path(N, h, dataset_size, rep)
    save_dir_path = os.path.join(SAVE_PATH, rbm_path)

    mag = TFIMChainMagnetization()
    energy = TFIMChainEnergy(h)

    cm = ComputeMetrics(CB_PERIOD,
                        {"mag": mag.statistics,
                         "energy": energy.statistics},
                        num_samples=NUM_SAMPLES,
                        k=NUM_GIBBS_STEPS_OBS,
                        batch_size=100)

    es = VarianceBasedEarlyStopping(CB_PERIOD, 0.01, 10, cm,
                                    "energy_mean", "energy_variance")

    metric_saver = ModelSaver(CB_PERIOD, save_dir_path, "epoch_{}.metrics",
                              lambda rbm, ep: cm.last,
                              metadata_only=True)

    model_saver = ModelSaver(CB_PERIOD, save_dir_path, "epoch_{}.params")

    return model, data, [cm, es, metric_saver, model_saver], save_dir_path


TRAINING_CONFIG = {
    "epochs": int(1e5),
    "batch_size": 100,
    "k": 10,
    "persistent": False,
    "lr": 1e-3,
    "momentum": 0.0,
    "l1_reg": 0.0,
    "l2_reg": 1e-5,
    "progbar": False
}


def delete_old_params(path):
    # Delete all param files except for the last one
    for f in get_all_epoch_params_except_last(path):
        if f.is_file():
            f.unlink()


def dataset_scaling(N: int, h: float, dataset_size: int, rep: int):
    print(f"Beginning evaluation for system size N={N} and h={h}")
    print(f"Dataset Size: {dataset_size}; Repetition: {rep}.")
    model, data, callbacks, save_dir_path = init(N, h, dataset_size, rep)
    model.train(data, callbacks=callbacks, **TRAINING_CONFIG)
    delete_old_params(save_dir_path)

    # create file to show that training finished
    open(os.path.join(save_dir_path, "done"), "w").close()


def continue_dataset_scaling(path, starting_epoch_path):
    config = extract_config(path)
    starting_epoch_path = Path(starting_epoch_path)
    starting_epoch = get_epoch_num(starting_epoch_path)
    N, h, dataset_size, rep = (config["N"], config["h"],
                               config["dataset_size"], config["rep"])

    print(f"Continuing evaluation for system size N={N} and "
          f"h={h} at epoch {starting_epoch}.")
    print(f"Dataset Size: {dataset_size}; Repetition: {rep}.")

    model, data, callbacks, save_dir_path = init(N, h, dataset_size, rep)

    # Populate metric history with old metric values
    epoch_paths = Path(path).glob("./epoch_*.metrics")
    for ep in epoch_paths:
        stats = torch.load(ep)
        callbacks[0].metric_values.append((ep, stats))
        callbacks[0].last = stats.copy()
    
    print("Begin training...")
    model.train(data,
                starting_epoch=starting_epoch,
                callbacks=callbacks,
                **TRAINING_CONFIG)

    delete_old_params(save_dir_path)
    open(os.path.join(save_dir_path, "done"), "w").close()


@click.command()
@click.option("--task-id", type=int, envvar="SLURM_ARRAY_TASK_ID")
@click.option("--rerun", is_flag=True)
def main(task_id, rerun):
    if not rerun and task_id is not None:
        print(f"Task id is: {task_id}")
        config = POSSIBLE_CONFIGS[task_id]
        dataset_scaling(**config)
    else:
        incomplete_runs = list(scan_for_incomplete())
        index = task_id if task_id is not None and task_id < len(incomplete_runs) else 0
        if task_id > len(incomplete_runs):
            print("No more incomplete runs")
            return
        print(f"Found {len(incomplete_runs)} incomplete runs.")
        if len(incomplete_runs) > 0:
            continue_dataset_scaling(
                incomplete_runs[index],
                starting_epoch_path=get_last_epoch_params(incomplete_runs[index]))


if __name__ == '__main__':
    main()
