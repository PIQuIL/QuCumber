from qucumber.nn_states import PositiveWavefunction
import matplotlib.pyplot as plt
import numpy as np

def loaddata(datafile):
    data = open(datafile)
    line = data.readline()
    samples = []
    while line != "":
        samples.append(list(map(int,line.strip("\n").split(" ")[0:-1])))
        line = data.readline()

    samples = np.array(samples)
    return samples

def correlation(model,datafile):

    datasamples = loaddata(datafile)
    covData = np.corrcoef(datasamples,rowvar = False)
    plt.matshow(covData)
    plt.colorbar()
    plt.show()

    nn_state = PositiveWavefunction.autoload(model)
    new_samples = nn_state.sample(k = 100,num_samples = 10000)

    data = np.array(new_samples)
    cov = np.corrcoef(data,rowvar = False)
    plt.matshow(cov)
    plt.colorbar()
    plt.show()

    row = int(len(cov)/2)
    curve = cov[row,row:]
    curveData = covData[row,row:]
    plt.plot(curve,"bo",label = "RBM")
    plt.plot(curveData,"ro",label = "DMRG")
    plt.show()

    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(curve,"bo",label = "RBM")
    ax.plot(curve,"ro",label = "DMRG")
    plt.show()

model = "Data/TFIM1D/NhStudy/Q50/16/Nh25/model.pt"
datafile = "Samples/TFIM1D/50Q/Samples.txt"
correlation(model,datafile)
