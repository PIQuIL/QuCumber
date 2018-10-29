import os.path
import numpy as np
import matplotlib.pyplot as plt

from qucumber.nn_states import PositiveWavefunction
from qucumber.callbacks import MetricEvaluator
from qucumber.callbacks import ObservableEvaluator
from qucumber.callbacks import Timer
from qucumber.observables import Heisenberg1DEnergy

import qucumber.utils.training_statistics as ts
import qucumber.utils.data as data

def trainEnergy(numQubits,numSamples1 = "All",numSamples2 = 1000,burn_in = 100,steps = 100,mT = 60):
    '''
    Trains RBM on samples using energy observable as metric.

    :param numQubits: Number of qubits.
    :type numQubits: int
    :param numSamples1: Number of samples to use from training file.
                        Default is "All".
    :type numSamples1: int or "All"
    :param numSamples2: Number of samples to generate internally.
                        Default is 1000.
    :type numSamples2: int
    :param burn_in: Number of Gibbs steps to perform before recording
                    any samples. Default is 100.
    :type burn_in: int
    :param steps: Number of Gibbs steps to perform between each sample.
                  Default is 100.
    :type steps: int
    :param mT: Maximum time elapsed during training.
    :type mT: int or float

    :returns: None
    '''

    # Load the data corresponding to the amplitudes and samples
    # of the quantum system
    psi_path = r"Samples/{0}Q/Amplitudes.txt".format(numQubits)
    train_path = r"Samples/{0}Q/Samples.txt".format(numQubits)
    train_data, true_psi = data.load_data(train_path, psi_path,
                                          numSamples=numSamples1)

    nv = train_data.shape[-1]
    nh = nv

    nn_state = PositiveWavefunction(num_visible=nv, num_hidden=nh)

    epochs = 10
    pbs = 2
    nbs = 2
    lr = 0.001
    k = 1

    log_every = 1
    h1d_energy = Heisenberg1DEnergy()
    space = nn_state.generate_hilbert_space(nv)

    callbacks = [
        ObservableEvaluator(
            log_every,
            [h1d_energy],
            verbose=True,
            num_samples=numSamples2,
            burn_in=burn_in,
            steps=steps,
        ),
        MetricEvaluator(
            log_every,
            {"Fidelity": ts.fidelity},
            target_psi=true_psi,
            verbose=True,
            space=space
        ),
        Timer(mT,log_every,verbose = True)
    ]

    nn_state.fit(
        train_data,
        epochs=epochs,
        pos_batch_size=pbs,
        neg_batch_size=nbs,
        lr=lr,
        k=k,
        callbacks=callbacks,
    )

    energies = callbacks[0].Heisenberg1DEnergy.mean
    errors = callbacks[0].Heisenberg1DEnergy.std_error
    variance = callbacks[0].Heisenberg1DEnergy.variance

    epoch = np.arange(log_every, epochs + 1, log_every)

    obsFile = open("Samples/{0}Q/Observables.txt".format(numQubits))
    obsFile.readline()
    line = obsFile.readline()
    H = round(float(line.strip("\n").split(" ")[1]),2)

    files = os.listdir("Data/Energy/Q{0}".format(numQubits))
    prevFile = files[-1]
    trial = 0
    for char in prevFile:
        if char.isdigit():
            trial = int(char) + 1

    ax = plt.axes()
    ax.plot(epoch, energies, color = "red")
    ax.set_xlim(log_every, epochs)
    ax.axhline(H,color = "black")
    ax.fill_between(epoch, energies - errors, energies + errors, alpha = 0.2, color = "black")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy")
    ax.grid()
    plt.tight_layout()
    plt.savefig("Data/Energy/Q{0}/Trial{1}".format(numQubits,trial))

    fidelities = callbacks[1].Fidelity
    runtimes = callbacks[2].epochTimes
    relativeErrors = []
    for i in range(len(energies)):
        relativeErrors.append(abs(energies[i] - H)/abs(H))

    resultsfile = open("Data/Energy/Q{0}/Trial{1}.txt".format(numQubits,trial),"w")
    resultsfile.write("samples: " + str(numSamples2) + "\n")
    resultsfile.write("burn_in: " + str(burn_in) + "\n")
    resultsfile.write("steps: " + str(steps) + "\n")
    resultsfile.write("\n")
    for i in range(len(fidelities)):
        resultsfile.write(str(round(float(fidelities[i]),3)) + "  ")
        resultsfile.write(str(round(float(relativeErrors[i]),6)) + "  ")
        resultsfile.write(str(round(float(runtimes[i]),3)) + "  \n")
    resultsfile.close()

trainEnergy(10,numSamples1 = 1500,numSamples2 = 1500,burn_in = 1000,steps = 1000)
