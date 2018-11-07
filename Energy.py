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

def trainEnergy(numQubits,
                nh,
                numSamples1 = 10000,
                numSamples2 = 5000,
                burn_in = 500,
                steps = 100,
                mT = 500,
                trial = 1,
                storeFidelities = False,
                plotError = False):
    '''
    Trains RBM on samples using energy observable as metric.

    :param numQubits: Number of qubits.
    :type numQubits: int
    :param nh: Number of hidden units.
    :type nh: int
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
    :param trial: Trial number. Default is 1.
    :type trial: int
    :param storeFidelities: Store fidelities.
    :type storeFidelities: bool
    :param plotError: Plot error.
    :type plotError: bool

    :returns: None
    '''

    # Load the data corresponding to the amplitudes and samples
    # of the quantum system
    psi_path = r"Samples/{0}Q/Amplitudes.txt".format(numQubits)
    train_path = r"Samples/{0}Q/Samples.txt".format(numQubits)
    train_data, true_psi = data.load_data(train_path, psi_path,
                                          numSamples=numSamples1)

    nv = train_data.shape[-1]
    nn_state = PositiveWavefunction(num_visible=nv, num_hidden=nh)

    epochs = 1000
    pbs = 2
    nbs = 2
    lr = 0.001
    k = 1

    log_every = 1
    h1d_energy = Heisenberg1DEnergy()
    space = nn_state.generate_hilbert_space(nv)

    if storeFidelities:
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
    else:
        callbacks = [
            ObservableEvaluator(
                log_every,
                [h1d_energy],
                verbose=True,
                num_samples=numSamples2,
                burn_in=burn_in,
                steps=steps,
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
    median = callbacks[0].Heisenberg1DEnergy.median
    lls = []
    uls = []
    mmRatio = []
    minError = []
    maxError = []

    obsFile = open("Samples/{0}Q/Observables.txt".format(numQubits))
    obsFile.readline()
    line = obsFile.readline()
    H = round(float(line.strip("\n").split(" ")[1]),2)

    C = 2.576
    for i in range(len(energies)):
        mean = energies[i]
        med = median[i]
        mmRatio.append(mean/med)
        std = np.sqrt(variance[i])
        ll = mean - C * std / np.sqrt(numSamples2)
        ul = mean + C * std / np.sqrt(numSamples2)
        lls.append(ll)
        uls.append(ul)
        lld = abs(ll - H)/abs(H)
        uld = abs(ul - H)/abs(H)
        if lld > uld:
            minError.append(uld)
            maxError.append(lld)
        else:
            minError.append(lld)
            maxError.append(uld)

    epoch = np.arange(log_every, len(energies) + 1, log_every)
    epoch.astype(int)
    nn_state.save("Data/Energy/Q{0}/Trial{1}.pt".format(numQubits,trial))

    if plotError:
        ax = plt.axes()
        ax.plot(epoch, energies, color = "red")
        ax.set_xlim(left = 1)
        ax.axhline(H,color = "black")
        ax.fill_between(epoch, energies - errors, energies + errors, alpha = 0.2, color = "black")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Energy")
        ax.grid()
        plt.title("Samples = {0}".format(numSamples2) +
                  " & Burn In = {0}".format(burn_in) +
                  " & Steps = {0} for N = {1}".format(steps,numQubits))
        plt.tight_layout()
        plt.savefig("Data/Energy/Q{0}/Trial{1}".format(numQubits,trial))

    if storeFidelities:
        fidelities = callbacks[1].Fidelity
        runtimes = callbacks[2].epochTimes
    else:
        runtimes = callbacks[1].epochTimes
    relativeErrors = []
    stdErrors = []
    for i in range(len(energies)):
        relativeErrors.append(abs(energies[i] - H)/abs(H))
        stdErrors.append(C * np.sqrt(variance[i])/np.sqrt(numSamples2))

    resultsfile = open("Data/Energy/Q{0}/Trial{1}.txt".format(numQubits,trial),"w")
    resultsfile.write("samples: " + str(numSamples2) + "\n")
    resultsfile.write("burn_in: " + str(burn_in) + "\n")
    resultsfile.write("steps: " + str(steps) + "\n")
    resultsfile.write("Exact H:" + str(H) + "\n")
    resultsfile.write("   Fidelity  ROE       Mean      Median    CI Width LL on CI  UL on CI  mmRatio  Min Err  Max Err\n")
    for i in range(len(energies)):
        resultsfile.write(str(epoch[i]) + "  ")
        if storeFidelities:
            resultsfile.write("%.6f" % round(float(fidelities[i]),6) + "  ")
        else:
            resultsfile.write("   N/A  " + "  ")
        resultsfile.write("%.6f" % round(float(relativeErrors[i]),6) + "  ")
        resultsfile.write("%.5f" % round(float(energies[i]),5) + "  ")
        resultsfile.write("%.5f" % round(float(median[i]),5) + "  ")
        resultsfile.write("%.5f" % round(float(stdErrors[i]),5) + "  ")
        resultsfile.write("%.5f" % round(float(lls[i]),5) + "  ")
        resultsfile.write("%.5f" % round(float(uls[i]),5) + "  ")
        resultsfile.write("%.5f" % round(float(mmRatio[i]),5) + "  ")
        resultsfile.write("%.5f" % round(float(minError[i]),5) + "  ")
        resultsfile.write("%.5f" % round(float(maxError[i]),5) + "  \n")

    resultsfile.close()

trainEnergy(10,1,numSamples1 = 10000,numSamples2 = 5000,burn_in = 500,steps = 100,mT = 30,trial = 1)
