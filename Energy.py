import os.path
import numpy as np
import matplotlib.pyplot as plt

from qucumber.nn_states import PositiveWavefunction
from qucumber.callbacks import MetricEvaluator
from qucumber.callbacks import ObservableEvaluator
from qucumber.callbacks import Timer
from qucumber.observables import Heisenberg1DEnergy
from Ising import TFIMChainEnergy

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
                plotError = False,
                model = "Heisenberg1D"):
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
    :param model: Model that sampling is based on.
                  Default is Heisenberg1D.
                  Other option is TFIM1D.
    :type model: str

    :returns: None
    '''

    # Load the data corresponding to the amplitudes and samples
    # of the quantum system
    if storeFidelities:
        psi_path = r"Samples/{0}/{1}Q/Amplitudes.txt".format(model,numQubits)
        train_path = r"Samples/{0}/{1}Q/Samples.txt".format(model,numQubits)
        train_data, true_psi = data.load_data(train_path, psi_path,
                                              numSamples=numSamples1)
    else:
        train_path = r"Samples/{0}/{1}Q/Samples.txt".format(model,numQubits)
        train_data = data.load_data(train_path, numSamples=numSamples1)[0]

    nv = train_data.shape[-1]
    nn_state = PositiveWavefunction(num_visible=nv, num_hidden=nh)

    if model == "Heisenberg1D":
        epochs = 10000
        pbs = 2
        nbs = 2
        lr = 0.001
        k = 1
        log_every = 1
    elif model == "TFIM1D":
        epochs = 10000
        pbs = 100
        nbs = 100
        lr = 0.01
        k = 1
        log_every = 10

    if model == "Heisenberg1D":
        modelEnergy = Heisenberg1DEnergy()
    elif model == "TFIM1D":
        modelEnergy = TFIMChainEnergy(1)

    if storeFidelities:
        space = nn_state.generate_hilbert_space(nv)

    if storeFidelities:
        callbacks = [
            ObservableEvaluator(
                log_every,
                [modelEnergy],
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
                [modelEnergy],
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

    if model == "Heisenberg1D":
        energies = callbacks[0].Heisenberg1DEnergy.mean
        errors = callbacks[0].Heisenberg1DEnergy.std_error
        variance = callbacks[0].Heisenberg1DEnergy.variance
        median = callbacks[0].Heisenberg1DEnergy.median
    elif model == "TFIM1D":
        energies = callbacks[0].TFIMChainEnergy.mean
        errors = callbacks[0].TFIMChainEnergy.std_error
        variance = callbacks[0].TFIMChainEnergy.variance
        median = callbacks[0].TFIMChainEnergy.median

    lls = []
    uls = []
    mmRatio = []
    minError = []
    maxError = []

    obsFile = open("Samples/{0}/{1}Q/Observables.txt".format(model,numQubits))
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

    epoch = np.arange(log_every, len(energies) * log_every + 1, log_every)
    epoch.astype(int)
    nn_state.save("Data/{0}/Energy/Q{1}/Trial{2}.pt".format(model,numQubits,trial))

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
        plt.savefig("Data/{0}/Energy/Q{1}/Trial{2}".format(model,numQubits,trial))

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

    resultsfile = open("Data/{0}/Energy/Q{1}/Trial{2}.txt".format(model,numQubits,trial),"w")
    resultsfile.write("nh: " + str(nh) + "\n")
    resultsfile.write("samples1: " + str(numSamples1) + "\n")
    resultsfile.write("samples2: " + str(numSamples2) + "\n")
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

def confidence(numQubits,numSamples,burn_in = 500,steps = 100,trial = 1,model = "Heisenberg1D"):
    '''
    Returns 99% confidence interval on trained RBM. Used to determine
    whether or not low relative error was obtained by chance.
    Essentially a tool for validating reconstruction accuracy.

    :param numQubits: Number of qubits.
    :type numQubits: int
    :param numSamples: Number of samples to generate.
    :type numSamples: int
    :param burn_in: Number of Gibbs steps to perform before recording
                    any samples. Default is 500.
    :type burn_in: int
    :param steps: Number of Gibbs steps to perform between each sample.
                  Default is 100.
    :type steps: int
    :param trial: Trial number.
    :type trial: int
    :param model: Model that sampling is based on.
                  Default is Heisenberg1D.
                  Other option is TFIM1D.
    :type model: str

    :returns: None
    '''

    trainedRBM = PositiveWavefunction.autoload("Data/{0}/Energy/Q{1}/Trial{2}.pt".format(model,numQubits,trial))

    if model == "Heisenberg1D":
        modelEnergy = Heisenberg1DEnergy()
    elif model == "TFIM1D":
        modelEnergy = TFIMChainEnergy(1)
    modelStats = modelEnergy.statistics(trainedRBM,numSamples,burn_in = burn_in,steps = steps)

    mean = modelStats["mean"]
    variance = modelStats["variance"]

    obsFile = open("Samples/{0}/{1}Q/Observables.txt".format(model,numQubits))
    obsFile.readline()
    line = obsFile.readline()
    H = round(float(line.strip("\n").split(" ")[1]),2)
    ROE = abs(mean - H)/abs(H)

    C = 2.576
    std = np.sqrt(variance)
    ll = mean - C * std / np.sqrt(numSamples)
    ul = mean + C * std / np.sqrt(numSamples)
    lld = abs(ll - H)/abs(H)
    uld = abs(ul - H)/abs(H)
    if lld > uld:
        minError = uld
        maxError = lld
    else:
        minError = lld
        maxError = uld

    confidenceFile = open("Data/{0}/Energy/Q{1}/Trial{2}CI.txt".format(model,numQubits,trial),"w")
    confidenceFile.write("ROE: {0}\n".format(ROE))
    confidenceFile.write("Min Error for 99% CI: {0}\n".format(minError))
    confidenceFile.write("Max Error for 99% CI: {0}\n".format(maxError))
    confidenceFile.close()

    print("ROE: {0}".format(ROE))
    print("Min Error for 99% CI: {0}".format(minError))
    print("Max Error for 99% CI: {0}".format(maxError))

trainEnergy(10,1,numSamples1 = 10000,numSamples2 = 5000,burn_in = 500,steps = 100,mT = 30,trial = 1,storeFidelities = True,model = "TFIM1D")
confidence(10,10000,1000,100,trial = 1,model = "TFIM1D")
