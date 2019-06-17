import os.path
import numpy as np
import matplotlib.pyplot as plt
import torch

from qucumber.nn_states import PositiveWavefunction
from qucumber.callbacks import MetricEvaluator
from qucumber.callbacks import ObservableEvaluator
from qucumber.callbacks import Timer
from qucumber.observables import Heisenberg1DEnergy
from Ising import TFIMChainEnergy

import qucumber.utils.training_statistics as ts
import qucumber.utils.data as data

def makeDir(path):
    '''
    Make directories to avoid forcing user to do so.

    :param path: Folder to create
    :type path: str

    :returns: None
    '''

    try:
        os.mkdir(path)
    except:
        pass

def earlyStopping(ROEs,tol,pat,req):
    '''
    Stops training early is defined criteria is met or desired
    improvement over certain number of epochs is not achieved.

    :param resultsfile: Name of file containing results
    :type resultsfile: str
    :param tol: Minimum required improvement
    :type tol: float
    :param pat: Maximum number of epochs allowed for minimum
                required improvement to be achieved.
    :type pat: float
    :param req: Goal for upper limit on ROE
    :type req: float

    :return: Status string if early stopping conditions met
             False otherwise
    :rtype: str or bool
    '''

    # Loop through all ROEs in case we simply want to look
    # at a results file where early stopping was not called
    for i in range(len(ROEs)):
        stop = True
        passed = False

        # If requirement met then stop
        if ROEs[i] < req:
            passed = True

        # If improvement requirement met then do not stop
        if not passed and len(ROEs) - i > pat:
            for j in range(1,pat + 1):
                if ROEs[i] - ROEs[i + j] > tol:
                    stop = False

        # If epoch is less than patience tnen continue
        elif not passed and len(ROEs) - i <= pat:
            stop = False

        if stop:
            if passed:
                string = "Stopped at Upper ROE = " + str(ROEs[i])
                string += " (Passed!)"
            else:
                string = "Stopped at Upper ROE = " + str(ROEs[i + pat])
                string += " (Early Stopping)"
            return string

    return False

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
                model = "Heisenberg1DFM",
                earlyStoppingParams = [0.0005,50,0.0005],
                seeds = [777,888,999],
                study = "Nh"):
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
                  Default is Heisenberg1DFM.
                  Other options are Heisenberg1DAFM and TFIM1D.
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
        log_every = 20

    if model == "Heisenberg1D":
        modelEnergy = Heisenberg1DEnergy()
    elif model == "TFIM1D":
        modelEnergy = TFIMChainEnergy(1)

    if storeFidelities:
        nn_state = PositiveWavefunction(num_visible=nv, num_hidden=nh)
        space = nn_state.generate_hilbert_space(nv)

    tol = earlyStoppingParams[0]
    pat = earlyStoppingParams[1]
    req = earlyStoppingParams[2]

    for seed in seeds:

        torch.manual_seed(seed)
        currentNh = nh
        currentM = numSamples1
        passed = False

        # Each iteration will correspond to an increment in Nh or M
        while not passed:

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

            nn_state = PositiveWavefunction(num_visible=nv,num_hidden=currentNh)
            roes = []
            lls = []
            uls = []
            mmRatio = []
            minError = []
            maxError = []
            fidelities = []

            # Run over several epochs until passed or converged
            for i in range(10000):

                nn_state.fit(
                    train_data,
                    epochs=log_every,
                    pos_batch_size=pbs,
                    neg_batch_size=nbs,
                    lr=lr,
                    k=k,
                    callbacks=callbacks,
                )

                if storeFidelities:
                    fidelity = callbacks[1].Fidelity
                    fidelities.append(fidelity[0])

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

                obsFile = open("Samples/{0}/{1}Q/Observables.txt".format(model,numQubits))
                obsFile.readline()
                line = obsFile.readline()
                H = round(float(line.strip("\n").split(" ")[1]),2)

                C = 2.576
                mean = energies[i]
                med = median[i]
                mmRatio.append(mean/med)
                std = np.sqrt(variance[i])
                roe = abs(mean - H)/abs(H)
                roes.append(roe)
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

                esStatus = earlyStopping(maxError,tol,pat,req)
                if esStatus != False:
                    print("#" * 52)
                    print(esStatus)
                    print("#" * 52)
                    if esStatus[-2] == "!":
                        passed = True
                    else:
                        if study == "Nh":
                            currentNh += 1
                        elif study == "M":
                            currentM += 1000
                    break

            path1 = "Data/{0}Study".format(study)
            path2 = path1 + "/Q{0}".format(numQubits)
            path3 = path2 + "/{0}".format(seed)
            if study == "Nh":
                path4 = path3 + "/{0}{1}".format(study,currentNh)
            else:
                path4 = path3 + "/{0}{1}".format(study,currentM)

            makeDir(path1)
            makeDir(path2)
            makeDir(path3)
            makeDir(path4)

            nn_state.save(path4 + "/model.pt")
            epochs = np.arange(log_every, len(energies) * log_every + 1, log_every)
            epochs.astype(int)

            numParams = numQubits * currentNh + numQubits + currentNh
            rfile = open(path4 + "/Results.txt","w")
            rfile.write("Number of Qubits: " + str(numQubits) + "\n")
            rfile.write("Number of Samples: " + str(currentM) + "\n")
            rfile.write("Hidden Units: " + str(currentNh) + "\n")
            rfile.write("Number of Parameters: " + str(numParams) + "\n")
            rfile.write("Energy Samples: " + str(numSamples2) + "\n")
            rfile.write("Burn In: " + str(burn_in) + "\n")
            rfile.write("Steps: " + str(steps) + "\n")

            for i in range(len(epochs)):
                rfile.write("Epoch {0}:   ".format(epochs[i]))
                rfile.write("ROE: {0:.8f}   ".format(roes[i]))
                rfile.write("ROE Upper Bound: {0:.8f}   ".format(maxError[i]))
                rfile.write("MM Ratio: {0:.4f}   ".format(mmRatio[i]))
                rfile.write("\n")

            rfile.close()

def confidence(numQubits,numSamples,burn_in = 500,steps = 100,trial = 1,model = "Heisenberg1DFM"):
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
                  Other options are Heisenberg1DAFM and TFIM1D.
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

trainEnergy(10,10,numSamples1 = 10000,numSamples2 = 5000,burn_in = 500,steps = 100,mT = 30,trial = 1,storeFidelities = True,model = "TFIM1D")
# confidence(10,10000,1000,100,trial = 1,model = "TFIM1D")
