import numpy as np
import matplotlib.pyplot as plt
import torch

from qucumber.nn_states import PositiveWavefunction
from qucumber.callbacks import MetricEvaluator
from qucumber.callbacks import Timer

import qucumber.utils.training_statistics as ts
import qucumber.utils.data as data


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
            dict["KLs"]
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
               "times": callbacks[1].epochTimes,
               "KLs":callbacks[0].KL}

    return results

def produceData(numQubits,epochs,b,lr,k,numSamples,opt,mT,log_every,**kwargs):
    '''
    Writes a datafile containing lists of fidelities and KLs for
    various times. The options are as follows.
    Option 1: List of batch sizes to try.
    Option 2: List of learning rates to try.
    Option 3: List of optimizers to try with their respective ideal LRs.
    Option 4: List of k values to try.
    Option 5: Standard run on one set of hyperparameters.

    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param epochs: Total number of epochs to train.
    :type epochs: int
    :param b: Batch size.
    :type b: int OR listof int
    :param lr: Learning rate.
    :type lr: float OR listof float
    :param k: Number of contrastive divergence steps in training.
    :type k: int OR listof int
    :param numSamples: Number of samples to use from sample file. Can use "All"
    :type numSamples: int
    :param optimizer: The constructor of a torch optimizer.
    :type optimizer: torch.optim.Optimizer OR listof torch.optim.Optimizer
    :param mT: Maximum time elapsed during training.
    :type mT: int or float
    :param log_every: Update callbacks every this number of epochs.
    :type log_every: int
    :param kwargs: Keyword arguments to pass to the optimizer

    :returns: None
    '''

    results = []
    # Test multiple batch sizes for fixed learning rate and optimizer
    if type(b) == list:
        for B in b:
            results.append(trainRBM(numQubits,epochs,B,B,lr,k,numSamples,optimizer,mT,log_every,**kwargs))
        files = os.listdir("Data/BatchSizes")
        trial = int(files[-1][5]) + 1
        datafile = open("Data/BatchSizes/Q{0}/Trial{1}.txt".format(numQubits,trial),"w")
        datafile.write("Batch Sizes: ")
        for B in b:
            datafile.write(str(B) + " ")
        datafile.write("\n")

    # Test multiple optimizers with their ideal learning rate
    elif type(opt) == list:
        for i in range(len(opt)):
            results.append(trainRBM(numQubits,epochs,b,b,lr[i],k,numSamples,optimizer[i],mT,log_every,**kwargs))
        files = os.listdir("Data/Optimizers")
        trial = int(files[-1][5]) + 1
        datafile = open("Data/Optimizers/Q{0}/Trial{1}.txt".format(numQubits,trial),"w")
        datafile.write("Optimizers: ")
        for OPT in opt:
            datafile.write(str(OPT)[10:len(OPT)] + " ")
        datafile.write("\n")
        datafile.write("Learning Rates: ")
        for LR in lr:
            datafile.write(str(LR) + " ")
        datafile.write("\n")

    # Test multiple learning rates for fixed optimizer
    elif type(lr) == list:
        for LR in lr:
            results.append(trainRBM(numQubits,epochs,b,b,LR,k,numSamples,optimizer,mT,log_every,**kwargs))
        files = os.listdir("Data/LearningRates")
        trial = int(files[-1][5]) + 1
        datafile = open("Data/LearningRates/Q{0}/Trial{1}.txt".format(numQubits,trial),"w")
        datafile.write("Learning Rates: ")
        for LR in lr:
            datafile.write(str(LR) + " ")
        datafile.write("\n")

    # Test multiple k values
    elif type(lr) == list:
        for K in k:
            results.append(trainRBM(numQubits,epochs,b,b,LR,K,numSamples,optimizer,mT,log_every,**kwargs))
        files = os.listdir("Data/GibbsSampling")
        trial = int(files[-1][5]) + 1
        datafile = open("Data/GibbsSampling/Q{0}/Trial{1}.txt".format(numQubits,trial),"w")
        datafile.write("k Values: ")
        for K in k:
            datafile.write(str(K) + " ")
        datafile.write("\n")

    # Else try run for single set of specified hyperparameters
    else:
        results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,optimizer,mT,log_every,**kwargs))
        files = os.listdir("Data/TryThis")
        trial = int(files[-1][5]) + 1
        datafile = open("Data/TryThis/Q{0}/Trial{1}.txt".format(numQubits,trial),"w")

    counter = 0
    for result in results:
        datafile.write("Epoch & Fidelity & Runtime" + " \n")
        for i in range(len(result["times"])):
            datafile.write(str(result["epochs"][i]) + " " +
                           str(round(result["fidelities"][i].item(),6)) + " " +
                           str(round(result["KLs"][i].item(),6)) + " " +
                           str(round(result["times"][i],6)) + "\n")
        datafile.write("\n")
        counter += 1
    datafile.close()

def graphData(folder,numQubits,trial,title):
    '''
    Graphs plots of fidelity vs RT and KL vs RT

    :param folder: Folder in Data containing file of interest.
    :type folder: str
    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int
    :param trial: Trial number.
    :type trial: int
    :param title: Title for graph.
    :type title: str

    :returns: None
    '''

    f = open("Data/{0}/Q{1}/Trial{2}.txt".format(folder,numQubits,trial))
    lines = []
    line = f.readline()
    hpValues = line.strip("\n").split(" ")
    if folder == "Optimizers":
        line = f.readline()
        LRs = line.strip("\n").split(" ")
    line = f.readline()
    fidelities = []
    runtimes = []
    KLs = []

    counter = 0
    while line != "":
        if line == "\n":
            counter += 1
            fidelities.append([])
            runtimes.append([])
            KLs.append([])
        elif line[0] == "E":
            line = f.readline()
            continue
        else:
            line = line.strip("\n")
            line = line.split(" ")
            fidelities[counter].append(float(line[1]))
            KLs[counter].append(float(line[2]))
            runtimes[counter].append(float(line[3]))
        line = f.readline()

    if folder == "Optimizers":
        label = hpValues[counter] + r"($\alpha$ = {0})".format(LRs[counter])
        plotname = "Data/{0}/Q{1}/Trial{2}".format(folder,numQubits,trial)
        plt.plot(runtimes,fidelities,"-o",label = label,markersize = 2)
        plt.xlabel("Runtime (Seconds)")
        plt.ylabel("Fidelity")
        plt.title(title)
        plt.legend()
        plt.savefig(plotname + "F",dpi = 200)
        plt.clf()
        plt.plot(runtimes,KLs,"-o",label = label,markersize = 2)
        plt.xlabel("Runtime (Seconds)")
        plt.ylabel("Fidelity")
        plt.title(title)
        plt.legend()
        plt.savefig(plotname + "K",dpi = 200)
        plt.clf()
    else:
        label = hpValues[counter]
        plotname = "Data/{0}/Q{1}/Trial{2}".format(folder,numQubits,trial)
        plt.plot(runtimes,fidelities,"-o",label = label,markersize = 2)
        plt.xlabel("Runtime (Seconds)")
        plt.ylabel("Fidelity")
        plt.title(title)
        plt.legend()
        plt.savefig(plotname + "F",dpi = 200)
        plt.clf()
        plt.plot(runtimes,KLs,"-o",label = label,markersize = 2)
        plt.xlabel("Runtime (Seconds)")
        plt.ylabel("Fidelity")
        plt.title(title)
        plt.legend()
        plt.savefig(plotname + "K",dpi = 200)
        plt.clf()

    f.close()
