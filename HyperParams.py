import numpy as np
import matplotlib.pyplot as plt
import torch

from qucumber.nn_states import PositiveWavefunction
from qucumber.callbacks import MetricEvaluator
from qucumber.callbacks import Timer

import qucumber.utils.training_statistics as ts
import qucumber.utils.data as data



listOptimizers = [
    "Adadelta",
    "Adam",
    "Adamax",
    "SGD",
    "SGD $\gamma$ = 0.9",
    "NAG $\gamma$ = 0.9"
]

def trainRBM(numQubits,epochs,pbs,nbs,lr,k,numSamples,optimizer,mT,log_every,**kwargs):
    '''
    Takes amplitudes and samples file as input and runs an RBM in order
    to reconstruct the quantum state. Returns a dictionary containing
    the fidelities and runtimes corresponding to certain epochs.

    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param epochs: Total number of epochs to train.
    :type epochs: int
    :param pbs: Positive batch size.
    :type pbs: int
    :param nbs: Negative batch size.
    :type nbs: int
    :param lr: Learning rate.
    :type lr: float
    :param k: Number of contrastive divergence steps in training.
    :type k: int
    :param numSamples: Number of samples to use from sample file. Can use "All"
    :type numSamples: int
    :param optimizer: The constructor of a torch optimizer.
    :type optimizer: torch.optim.Optimizer
    :param mT: Maximum time elapsed during training.
    :type mT: int or float
    :param log_every: Update callbacks every this number of epochs.
    :type log_every: int
    :param kwargs: Keyword arguments to pass to the optimizer

    :returns: Dictionary of fidelities and runtimes at various epochs.
    :rtype: dict["epochs"]
            dict["fidelities"]
            dict["times"]
    '''

    # Load the data corresponding to the amplitudes and samples
    # of the quantum system
    psi_path = r"Samples/{0}Q/AmplitudesP.txt".format(numQubits)
    train_path = r"Samples/{0}Q/Samples.txt".format(numQubits)
    train_data, true_psi = data.load_data(train_path, psi_path,
                                          numSamples=numSamples)

    # Specify the number of visible and hidden units and
    # initialize the RBM
    nv = train_data.shape[-1]
    nh = nv
    nn_state = PositiveWavefunction(num_visible = nv,num_hidden = nh,
                                    gpu = False)

    space = nn_state.generate_hilbert_space(nv)

    # And now the training can begin!
    callbacks = [
        MetricEvaluator(
            log_every,
            {"Fidelity": ts.fidelity, "KL": ts.KL},
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
        optimizer=optimizer,
        **kwargs
    )

    results = {"epochs": np.arange(log_every, epochs + 1, log_every),
               "fidelities": callbacks[0].Fidelity,
               "times": callbacks[1].epochTimes}

    return results

def produceDataB(epochs,k,numQubits,numSamples,mT,batchSizes,log_every,trial):
    '''
    Writes a datafile containing lists of fidelities and runtimes for
    several epochs for various batch sizes.

    :param epochs: Total number of epochs to train.
    :type epochs: int
    :param k: Number of contrastive divergence steps in training.
    :type k: int
    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param numSamples: Number of samples to use from sample file.
    :type numSamples: int
    :param mT: Maximum time elapsed during training.
    :type mT: int or float
    :param batchSizes: List of batch sizes to try.
    :type batchSizes: listof int
    :param log_every: Update callbacks every this number of epochs.
    :type log_every: int
    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''

    results = []
    for b in batchSizes:
        results.append(trainRBM(numQubits,epochs,b,b,0.01,k,numSamples,torch.optim.SGD,mT,log_every))

    datafile = open("Data/BatchSizes/Q{0}/Trial{1}.txt".format(numQubits,trial),"w")
    counter = 0
    for result in results:
        datafile.write("Batch size is {0}\n".format(batchSizes[counter]))
        datafile.write("Epoch & Fidelity & Runtime" + " \n")
        for i in range(len(result["times"])):
            datafile.write(str(result["epochs"][i]) + " " +
                           str(round(result["fidelities"][i].item(),6)) + " " +
                           str(round(result["times"][i],6)) + "\n")
        datafile.write("\n")
        counter += 1
    datafile.close()

def graphDataB(numQubits,trial):
    '''
    Graphs a plot of fidelity vs runtime

    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''

    f = open("Data/BatchSizes/Q{0}/Trial{1}.txt".format(numQubits,trial))
    lines = []
    line = f.readline()
    fidelities = []
    runtimes = []
    batchsizes = []

    counter = 0
    while line != "":
        if line == "\n":
            plt.plot(runtimes,fidelities,"-o",label = batchsizes[counter],markersize = 2)
            counter += 1
            fidelities = []
            runtimes = []
        elif line[0] == "E":
            line = f.readline()
            continue
        elif line[0] == "B":
            line = line.strip("\n")
            line = line.split(" ")
            batchsizes.append(int(line[3]))
        else:
            line = line.strip("\n")
            line = line.split(" ")
            fidelities.append(float(line[1]))
            runtimes.append(float(line[2]))
        line = f.readline()

    plt.xlabel("Runtime (Seconds)")
    plt.ylabel("Fidelity")
    plt.title("Learning Curve for Various Batch Sizes with SGD")
    plt.legend()
    plt.savefig("Data/BatchSizes/Q{0}/Trial{1}".format(numQubits,trial),dpi = 200)
    plt.clf()
    f.close()

def produceData(epochs,b,k,numQubits,numSamples,lrs,opt,mT,log_every,trial):
    '''
    Writes a datafile containing lists of fidelities and runtimes for
    several epochs for various optimizers.

    :param epochs: Total number of epochs to train.
    :type epochs: int
    :param b: Batch size.
    :type b: int
    :param k: Number of contrastive divergence steps in training.
    :type k: int
    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param numSamples: Number of samples to use from sample file.
    :type numSamples: int
    :param lrs: List of learning rates.
    :type lrs: listof float
    :param opt: Type of optimizer.
    :type opt: str
    :param mT: Maximum time elapsed during training.
    :type mT: int or float
    :param batchSizes: List of batch sizes to try.
    :type batchSizes: listof int
    :param log_every: Update callbacks every this number of epochs.
    :type log_every: int
    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''

    results = []
    if opt == "Adadelta":
        for lr in lrs:
            results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.Adadelta,mT,log_every))
    if opt == "Adam":
        for lr in lrs:
            results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.Adam,mT,log_every))
    if opt == "Adamax":
        for lr in lrs:
            results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.Adamax,mT,log_every))
    if opt == "SGD":
        for lr in lrs:
            results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.SGD,mT,log_every))
    if opt == "SGDM":
        for lr in lrs:
            results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.SGD,mT,log_every,momentum=0.9))
    if opt == "NAG":
        for lr in lrs:
            results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.SGD,mT,log_every,momentum=0.9,nesterov=True))

    datafile = open("Data/LearningRates/Q{0}/{1}{2}.txt".format(numQubits,opt,trial),"w")
    datafile.write("Optimizer is {0}\n".format(opt))
    datafile.write("\n")
    counter = 0
    for result in results:
        datafile.write("LR is " + str(lrs[counter]) + "\n")
        datafile.write("Epoch & Fidelity & Runtime" + " \n")
        for i in range(len(result["times"])):
            datafile.write(str(result["epochs"][i]) + " " +
                           str(round(result["fidelities"][i].item(),6)) + " " +
                           str(round(result["times"][i],6)) + "\n")
        datafile.write("\n")
        counter += 1
    datafile.close()

def graphData(numQubits,opt,trial):
    '''
    Graphs a plot of fidelity vs runtime

    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param opt: Type of optimizer.
    :type opt: str
    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''

    f = open("Data/LearningRates/Q{0}/{1}{2}.txt".format(numQubits,opt,trial))
    lines = []
    f.readline()
    f.readline()
    line = f.readline()
    fidelities = []
    runtimes = []

    counter = 0
    while line != "":
        if line == "\n":
            plt.plot(runtimes,fidelities,"-o",label = lr,markersize = 2)
            counter += 1
            fidelities = []
            runtimes = []
        elif line[0] == "E":
            line = f.readline()
            continue
        elif line[0:2] == "LR":
            line = line.strip("\n")
            line = line.split(" ")
            lr = line[2]
        else:
            line = line.strip("\n")
            line = line.split(" ")
            fidelities.append(float(line[1]))
            runtimes.append(float(line[2]))
        line = f.readline()

    plt.xlabel("Runtime (Seconds)")
    plt.ylabel("Fidelity")
    plt.title("Learning Curve for {0} with Various Learning Rates".format(opt))
    plt.legend()
    plt.savefig("Data/LearningRates/Q{0}/{1}{2}".format(numQubits,opt,trial),dpi = 200)
    plt.clf()
    f.close()

def graphLR(numQubits,trial):
    '''
    Graphs a plot of fidelity vs runtime

    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''

    f = open("Data/CompareOpt/Q{0}/T{1}.txt".format(numQubits,trial))
    lines = []
    line = f.readline()
    fidelities = []
    runtimes = []

    counter = 0
    while line != "":
        if line == "\n":
            label = r"{0} ($\alpha = {1}$)".format(opt,lr)
            plt.plot(runtimes,fidelities,"-o",label = label,markersize = 2)
            counter += 1
            fidelities = []
            runtimes = []
        elif line[0] == "E":
            line = f.readline()
            continue
        elif line[0] == "O":
            line = line.strip("\n")
            line = line.split(" ")
            opt = line[2]
        elif line[0:2] == "LR":
            line = line.strip("\n")
            line = line.split(" ")
            lr = line[2]
        else:
            line = line.strip("\n")
            line = line.split(" ")
            fidelities.append(float(line[1]))
            runtimes.append(float(line[2]))
        line = f.readline()

    plt.xlabel("Runtime (Seconds)")
    plt.ylabel("Fidelity")
    plt.title("Learning Curves for Various Optimizers with N = {0}".format(numQubits))
    plt.legend()
    plt.savefig("Data/CompareOpt/Q{0}/T{1}".format(numQubits,trial),dpi = 200)
    plt.clf()
    f.close()

def produceDataK(epochs,b,numQubits,numSamples,mT,kValues,log_every,trial):
    '''
    Writes a datafile containing lists of fidelities and runtimes for
    several epochs for various k values.

    :param epochs: Total number of epochs to train.
    :type epochs: int
    :param b: Batch size.
    :type b: int
    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param numSamples: Number of samples to use from sample file.
    :type numSamples: int
    :param mT: Maximum time elapsed during training.
    :type mT: int or float
    :param kValues: List of k values to try.
    :type kValues: listof int
    :param log_every: Update callbacks every this number of epochs.
    :type log_every: int
    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''

    results = []
    for k in kValues:
        results.append(trainRBM(numQubits,epochs,4,4,0.01,k,numSamples,torch.optim.SGD,mT,log_every))

    datafile = open("Data/kValues/Q{0}/Trial{1}.txt".format(numQubits,trial),"w")
    counter = 0
    for result in results:
        datafile.write("k value is {0}\n".format(kValues[counter]))
        datafile.write("Epoch & Fidelity & Runtime" + " \n")
        for i in range(len(result["times"])):
            datafile.write(str(result["epochs"][i]) + " " +
                           str(round(result["fidelities"][i].item(),6)) + " " +
                           str(round(result["times"][i],6)) + "\n")
        datafile.write("\n")
        counter += 1
    datafile.close()

def graphDataK(numQubits,trial):
    '''
    Graphs a plot of fidelity vs runtime

    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''

    f = open("Data/kValues/Q{0}/Trial{1}.txt".format(numQubits,trial))
    lines = []
    line = f.readline()
    fidelities = []
    runtimes = []
    kvalues = []

    counter = 0
    while line != "":
        if line == "\n":
            plt.plot(runtimes,fidelities,"-o",label = kvalues[counter],markersize = 2)
            counter += 1
            fidelities = []
            runtimes = []
        elif line[0] == "E":
            line = f.readline()
            continue
        elif line[0] == "k":
            line = line.strip("\n")
            line = line.split(" ")
            kvalues.append(int(line[3]))
        else:
            line = line.strip("\n")
            line = line.split(" ")
            fidelities.append(float(line[1]))
            runtimes.append(float(line[2]))
        line = f.readline()

    plt.xlabel("Runtime (Seconds)")
    plt.ylabel("Fidelity")
    plt.title("Learning Curve for Various k Values for N = {0}".format(numQubits))
    plt.legend()
    plt.savefig("Data/kValues/Q{0}/Trial{1}".format(numQubits,trial),dpi = 200)
    plt.clf()
    f.close()

def tryIsing(epochs,b,lr,k,numSamples,mT,log_every,trial):
    '''
    Try running RBM on original tutorial data for 1D Ising Model.

    :param epochs: Total number of epochs to train.
    :type epochs: int
    :param b: Batch size.
    :type b: int
    :param lr: Learning rate.
    :type lr: float
    :param k: Number of contrastive divergence steps in training.
    :type k: int
    :param numSamples: Number of samples to use from sample file. Can use "All"
    :type numSamples: int
    :param mT: Maximum time elapsed during training.
    :type mT: int or float
    :param log_every: Update callbacks every this number of epochs.
    :type log_every: int
    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''
    results = []
    results.append(trainRBM(10,epochs,b,b,lr,k,numSamples,torch.optim.SGD,mT,log_every))

    datafile = open("Data/Ising/Trial{0}.txt".format(trial),"w")
    counter = 0
    for result in results:
        datafile.write("Epoch & Fidelity & Runtime" + " \n")
        for i in range(len(result["times"])):
            datafile.write(str(result["epochs"][i]) + " " +
                           str(round(result["fidelities"][i].item(),6)) + " " +
                           str(round(result["times"][i],6)) + "\n")
        datafile.write("\n")
        counter += 1
    datafile.close()

def tryIsingGraph(trial):
    '''
    Graphs a plot of fidelity vs runtime

    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''

    f = open("Data/Ising/Trial{0}.txt".format(trial))
    lines = []
    line = f.readline()
    fidelities = []
    runtimes = []

    while line != "":
        if line == "\n":
            plt.plot(runtimes,fidelities,"-o",markersize = 2)
            fidelities = []
            runtimes = []
        elif line[0] == "E":
            line = f.readline()
            continue
        else:
            line = line.strip("\n")
            line = line.split(" ")
            fidelities.append(float(line[1]))
            runtimes.append(float(line[2]))
        line = f.readline()

    plt.xlabel("Runtime (Seconds)")
    plt.ylabel("Fidelity")
    plt.title("Learning Curve for Ising Model")
    plt.savefig("Data/Ising/Trial{0}".format(trial),dpi = 200)
    plt.clf()
    f.close()

def tryThis(epochs,b,lr,k,numQubits,numSamples,mT,log_every,trial):
    '''
    Try running RBM on original tutorial data for 1D Ising Model.

    :param epochs: Total number of epochs to train.
    :type epochs: int
    :param b: Batch size.
    :type b: int
    :param lr: Learning rate.
    :type lr: float
    :param k: Number of contrastive divergence steps in training.
    :type k: int
    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param numSamples: Number of samples to use from sample file. Can use "All"
    :type numSamples: int
    :param mT: Maximum time elapsed during training.
    :type mT: int or float
    :param log_every: Update callbacks every this number of epochs.
    :type log_every: int
    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''
    results = trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.SGD,mT,log_every)
    datafile = open("Data/TryThis/Q{0}/Trial{1}.txt".format(numQubits,trial),"w")
    datafile.write("Epoch & Fidelity & Runtime" + " \n")
    for i in range(len(results["times"])):
        datafile.write(str(results["epochs"][i]) + " " +
                       str(round(results["fidelities"][i].item(),6)) + " " +
                       str(round(results["times"][i],6)) + "\n")
    datafile.write("\n")
    datafile.close()

def tryThisGraph(numQubits,trial):
    '''
    Graphs a plot of fidelity vs runtime

    :param trial: Trial number.
    :type trial: int

    :returns: None
    '''

    f = open("Data/TryThis/Q{0}/Trial{1}.txt".format(numQubits,trial))
    lines = []
    line = f.readline()
    fidelities = []
    runtimes = []

    while line != "":
        if line == "\n":
            plt.plot(runtimes,fidelities,"-o",markersize = 2)
            fidelities = []
            runtimes = []
        elif line[0] == "E":
            line = f.readline()
            continue
        else:
            line = line.strip("\n")
            line = line.split(" ")
            fidelities.append(float(line[1]))
            runtimes.append(float(line[2]))
        line = f.readline()

    plt.xlabel("Runtime (Seconds)")
    plt.ylabel("Fidelity")
    plt.title("Learning Curve for QST with N = {0}".format(numQubits))
    plt.savefig("Data/TryThis/Q{0}/Trial{1}".format(numQubits,trial),dpi = 200)
    plt.clf()
    f.close()
