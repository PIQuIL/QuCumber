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
import pathlib
import re
# from itertools import product
import signal
import os.path


def load_train(N, h):
    data = np.loadtxt(
        f"/home/ejaazm/projects/def-rgmelko/ejaazm/"
        f"tfim1d/datasets/tfim1d_N{N}_h{h:.2f}_train.txt")
    return data.astype('float32')


CB_PERIOD = 100
SAVE_PATH = ("/home/ejaazm/projects/def-rgmelko/ejaazm/"
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
    px = pathlib.Path(path)
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

    return config


def make_rbm_name(N, h, dataset_size, rep):
    return f"N_{N}/h_{h:.2f}/dataset_size_{dataset_size}/rep_{rep}/"


def scan_for_incomplete():
    px = pathlib.Path(SAVE_PATH)
    all_incomplete = set(map(lambda p: p.parent, px.glob("./**/checkpoint")))
    return all_incomplete


GET_NUMS_REGEX = re.compile(r'\d+')  # match all numerical chars


def get_epoch_num(p):
    return int(GET_NUMS_REGEX.findall(p.name)[0])


def get_last_epoch_path(path):
    epoch_paths = pathlib.Path(path).glob("./epoch*")
    return max(epoch_paths, key=get_epoch_num)


@click.command()
@click.option("--task-id", type=int, envvar="SLURM_ARRAY_TASK_ID")
@click.option("--rerun", is_flag=True)
def main(task_id, rerun):
    if not rerun:
        print(f"Task id is: {task_id}")
        config = POSSIBLE_CONFIGS[task_id]
        run(**config)
    else:
        incomplete_runs = scan_for_incomplete()
        print(f"Found {len(incomplete_runs)} incomplete runs.")
        if len(incomplete_runs) > 0:
            continue_run(
                incomplete_runs[0],
                starting_epoch_path=get_last_epoch_path(incomplete_runs[0]))


def run(N, h, dataset_size, rep):
    print(f"Beginning evaluation for system size N={N} and h={h}")
    print(f"Dataset Size: {dataset_size}; Repetition: {rep}.")

    data = load_train(N, h)

    rs = np.random.RandomState(int(N * h * dataset_size * rep))
    data = data[rs.choice(len(data), dataset_size, replace=False)]

    model = RBM(num_visible=N, num_hidden=N, seed=rep)

    rbm_name = make_rbm_name(N, h, dataset_size, rep)

    current_epoch = 0

    def sigterm_handler(signal, frame):
        print(f"Job killed while evaluating {rbm_name}")
        model.save(os.path.join(SAVE_PATH, rbm_name,
                                f"checkpoint_{current_epoch}"))

    # catch any sigterms that show up
    signal.signal(signal.SIGTERM, sigterm_handler)

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

    ms = ModelSaver(CB_PERIOD, SAVE_PATH, rbm_name,
                    lambda rbm, ep: cm.last,
                    metadata_only=True)

    def update_epoch_cb(rbm, epoch):
        global current_epoch
        current_epoch = epoch

    model.train(data,
                epochs=int(1e5), batch_size=100,
                k=10, persistent=False,
                lr=1e-3, momentum=0.0,
                l1_reg=0.0, l2_reg=1e-5,
                progbar=False,
                callbacks=[cm, es, ms, update_epoch_cb])

    # Save most recent observable values
    # along with trained model params
    ms.metadata_only = False
    ms.metadata = cm.last
    ms(model, es.last_epoch, last=True)


def continue_run(path, starting_epoch_path):
    config = extract_config(path)
    starting_epoch_path = pathlib.Path(starting_epoch_path)
    starting_epoch = get_epoch_num(starting_epoch_path)
    N, h, dataset_size, rep = (config["N"], config["h"],
                               config["dataset_size"], config["rep"])
    print(f"Continuing evaluation for system size N={N} and "
          f"h={h} at epoch {starting_epoch}.")
    print(f"Dataset Size: {dataset_size}; Repetition: {rep}.")

    data = load_train(N, h)

    rs = np.random.RandomState(int(N * h * dataset_size * rep))
    data = data[rs.choice(len(data), dataset_size, replace=False)]

    model = RBM(num_visible=N, num_hidden=N, seed=rep)
    model.load(str(starting_epoch_path.absolute()))

    rbm_name = make_rbm_name(N, h, dataset_size, rep)

    global RBM_NAME
    RBM_NAME = rbm_name

    mag = TFIMChainMagnetization()
    energy = TFIMChainEnergy(h)

    cm = ComputeMetrics(CB_PERIOD,
                        {"mag": mag.statistics,
                         "energy": energy.statistics},
                        num_samples=NUM_SAMPLES,
                        k=NUM_GIBBS_STEPS_OBS,
                        batch_size=100)

    # Populate cm history with old metric values
    epoch_paths = pathlib.Path(path).glob("./epoch*")
    for ep in epoch_paths:
        stats = torch.load(ep)
        cm.metric_values.append(stats)
        cm.last = stats.copy()

    es = VarianceBasedEarlyStopping(CB_PERIOD, 0.01, 10, cm,
                                    "energy_mean",
                                    "energy_variance")

    ms = ModelSaver(CB_PERIOD, SAVE_PATH, rbm_name,
                    lambda rbm, ep: cm.last,
                    metadata_only=True)

    model.train(data,
                epochs=int(1e5),
                starting_epoch=starting_epoch,
                batch_size=100,
                k=10, persistent=False,
                lr=1e-3, momentum=0.0,
                l1_reg=0.0, l2_reg=1e-5,
                progbar=False,
                callbacks=[cm, es, ms])

    # Save most recent observable values
    # along with trained model params
    ms.metadata_only = False
    ms.metadata = cm.last
    ms(model, es.last_epoch, last=True)


if __name__ == '__main__':
    main()
