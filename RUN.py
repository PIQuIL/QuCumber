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

########## STAGE 1: Test various batch sizes on regular SGD ##########

def produceDataB(epochs,k,numQubits,numSamples,mT,batchSizes,log_every):
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

    :returns: None
    '''

    results = []
    for b in batchSizes:
        results.append(trainRBM(numQubits,epochs,b,b,0.01,k,numSamples,torch.optim.SGD,mT,log_every))

    datafile = open("Data/BatchSizes/Q{0}/Epochs.txt".format(numQubits),"w")
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

def graphDataB(filename,numQubits):
    '''
    Graphs a plot of fidelity vs runtime

    :param filename: Name of file containing data
    :type filename: str
    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int

    :returns: None
    '''

    f = open(filename)
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
    plt.savefig(filename[0:len(filename) - 10] + "LC",dpi = 200)
    plt.clf()
    f.close()

'''
# Test N = 5
produceDataB(100000,1,5,5000,60,[2,4,8,16,32,64,128,256,512],10)
graphDataB("Data/BatchSizes/Trial1/Q5/Epochs.txt",5)
'''

######################################################################

#### STAGE 2: Test various learning rates on various optimizers ######

def produceData(epochs,b,k,numQubits,numSamples,lr,mT,log_every):
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
    :param lr: Learning rate.
    :type lr: float
    :param mT: Maximum time elapsed during training.
    :type mT: int or float
    :param batchSizes: List of batch sizes to try.
    :type batchSizes: listof int
    :param log_every: Update callbacks every this number of epochs.
    :type log_every: int

    :returns: None
    '''

    results = []
    results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.Adadelta,mT,log_every))
    results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.Adam,mT,log_every))
    results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.Adamax,mT,log_every))
    results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.SGD,mT,log_every))
    results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.SGD,mT,log_every,momentum=0.9))
    results.append(trainRBM(numQubits,epochs,b,b,lr,k,numSamples,torch.optim.SGD,mT,log_every,momentum=0.9,nesterov=True))

    datafile = open("Data/Optimizers/Q{0}/Epochs.txt".format(numQubits),"w")
    datafile.write("Learning rate is {0}\n".format(lr))
    datafile.write("\n")
    counter = 0
    for result in results:
        datafile.write("Optimizer is " + str(listOptimizers[counter]) + "\n")
        datafile.write("Epoch & Fidelity & Runtime" + " \n")
        for i in range(len(result["times"])):
            datafile.write(str(result["epochs"][i]) + " " +
                           str(round(result["fidelities"][i].item(),6)) + " " +
                           str(round(result["times"][i],6)) + "\n")
        datafile.write("\n")
        counter += 1
    datafile.close()

def graphData(filename,numQubits):
    '''
    Graphs a plot of fidelity vs runtime

    :param filename: Name of file containing data
    :type filename: str
    :param numQubits: Number of qubits in the quantum state.
    :type numQubits: int

    :returns: None
    '''

    f = open(filename)
    lines = []
    line = f.readline()
    line = line.strip("\n")
    line = line.split(" ")
    lr = line[3]
    line = f.readline()
    line = f.readline()
    fidelities = []
    runtimes = []

    counter = 0
    while line != "":
        if line == "\n":
            plt.plot(runtimes,fidelities,"-o",label = listOptimizers[counter],markersize = 2)
            counter += 1
            fidelities = []
            runtimes = []
        elif line[0] == "E" or line[0] == "O":
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
    plt.title("Learning Curve for Various Optimizers with " +
              r"$\alpha = {0}$".format(lr))
    plt.legend()
    plt.savefig(filename[0:len(filename) - 10] + "LC",dpi = 200)
    plt.clf()
    f.close()

'''
Nvalues = [10,15,20]
Bvalues = [128,256,512]

for N in Nvalues:
    for B in Bvalues:
        produceData(1000,B,B,1,N,20000)
        graphData("Data/Q{0}/Text/B{1}.txt".format(N,B),N)
'''
#produceData(10000,10,1,5,5000,0.01,10,1)
graphData("Data/Optimizers/LR0p01/Trial1/Q5/Epochs.txt",5)
graphData("Data/Optimizers/LR0p01/Trial1/Q10/Epochs.txt",10)
graphData("Data/Optimizers/LR0p01/Trial1/Q15/Epochs.txt",15)

######################################################################
