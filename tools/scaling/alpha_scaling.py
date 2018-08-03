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
# import signal
# import os.path
# import pickle


def load_train(N, h):
    data = np.loadtxt(
        f"/home/ejaazm/projects/def-rgmelko/ejaazm/"
        f"tfim1d/datasets/tfim1d_N{N}_h{h:.2f}_train.txt")
    return data.astype('float32')


CB_PERIOD = 100
SAVE_PATH = ("/home/ejaazm/projects/def-rgmelko/ejaazm/"
             "tfim1d_alpha_scaling/model_snapshots")
RBM_NAME = "tmp"  # should get overwritten
CURRENT_CONFIG = {}

NUM_GIBBS_STEPS_OBS = 100
NUM_SAMPLES = int(1e5)

ALPHA_STEP_SIZES = {
    16: 2,
    32: 4,
    64: 4,
    128: 4,
    256: 8,
    512: 16
}


# def sigterm_handler(signal, frame):
#     print(f"Job killed while evaluating {RBM_NAME}")
#     pickle.dump(CURRENT_CONFIG,
#                 os.path.join(SAVE_PATH, "job_killed.cpt"))


# # catch any sigterms that show up
# signal.signal(signal.SIGTERM, sigterm_handler)

POSSIBLE_CONFIGS = []
for N, step in ALPHA_STEP_SIZES.items():
    for num_hidden in range(step, N+step, step):
        for h in [0.5, 1.0, 1.5]:
            for rep in range(5):
                POSSIBLE_CONFIGS.append({
                    "N": N,
                    "h": h,
                    "num_hidden": num_hidden,
                    "rep": rep
                })


@click.command()
@click.option("--task-id", type=int, envvar="SLURM_ARRAY_TASK_ID")
def main(task_id):
    print(f"Task id is: {task_id}")
    config = POSSIBLE_CONFIGS[task_id]

    global CURRENT_CONFIG
    CURRENT_CONFIG = config

    run(**CURRENT_CONFIG)


def make_rbm_name(N, h, num_hidden, rep):
    return f"N_{N}/h_{h:.2f}/num_hidden_{num_hidden}/rep_{rep}/"


def run(N, h, num_hidden, rep):
    alpha = N / float(num_hidden)
    print(f"Beginning evaluation for system size N={N} and h={h}")
    print(f"Alpha: {alpha}; Repetition: {rep}.")

    data = load_train(N, h)

    model = RBM(num_visible=N, num_hidden=num_hidden, seed=rep)

    rbm_name = make_rbm_name(N, h, num_hidden, rep)

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

    es = VarianceBasedEarlyStopping(CB_PERIOD, 0.01, 10, cm,
                                    "energy_mean",
                                    "energy_variance")

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
