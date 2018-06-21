"""
Run TFIM Scaling experiments on GRAHAM
"""

import click
import numpy as np
from observables import TFIMChainEnergy, TFIMChainMagnetization
from rbm.callbacks import ModelSaver, ComputeMetrics, EarlyStopping
from rbm import RBM
from itertools import product
import signal
import os.path


def load_train(N, h):
    data = np.loadtxt(
        f"/home/ejaazm/projects/def-rgmelko/ejaazm/"
        f"tfim1d/datasets/tfim1d_N{N}_h{h:.2f}_train.txt")
    return data.astype('float32')


CB_PERIOD = 100
SAVE_PATH = ("/home/ejaazm/projects/def-rgmelko/ejaazm/"
             "tfim1d_scaling/model_snapshots")
RBM_NAME = "tmp"  # should get overwritten


NUM_GIBBS_STEPS_OBS = 100
NUM_SAMPLES = int(1e5)


def sigterm_handler(signal, frame):
    print(f"Job killed while evaluating {RBM_NAME}")
    open(os.path.join(SAVE_PATH, RBM_NAME, "job_killed"), 'a').close()


# catch any sigterms that show up
signal.signal(signal.SIGTERM, sigterm_handler)

POSSIBLE_CONFIGS = list(product(
    # [16, 32, 64, 128, 256],          # N : do 512 later
    # [0.5, 1.0, 1.5],                 # h
    range(10000, 210000, 10000),     # dataset_size
    [1 / 8., 1 / 4., 1 / 2., 1, 2],  # alpha
    range(5)                         # run number
))


@click.command()
@click.option("--task-id", type=int, envvar="SLURM_ARRAY_TASK_ID")
@click.option("-n", type=int)
@click.option("-h", type=float)
def main(task_id, n, h):
    print(f"Task id is: {task_id}")
    config = POSSIBLE_CONFIGS[task_id]
    run(n, h, *config)


def run(N, h, dataset_size, alpha, run):
    print(f"Beginning evaluation for system size N={N} and h={h}")
    print(f"Dataset Size: {dataset_size}; Alpha: {alpha}; Run: {run}.")

    data = load_train(N, h)
    data = data[np.random.choice(len(data), dataset_size, replace=False)]

    num_hidden = int(alpha * N)

    model = RBM(num_visible=N, num_hidden=num_hidden)

    rbm_name = (f"N_{N}/h_{h:.2f}/data_size_{dataset_size}/"
                f"alpha_{alpha}/run_{run}/")

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

    es = EarlyStopping(CB_PERIOD, 0.1/100, 10,
                       lambda rbm, **kwargs: cm.last["Energy"])

    ms = ModelSaver(CB_PERIOD, SAVE_PATH, rbm_name,
                    lambda rbm, ep: cm.last,
                    metadata_only=True)

    model.train(data,
                epochs=int(1e5), batch_size=100,
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
