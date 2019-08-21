from qucumber.nn_states import PositiveWavefunction
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("apsWD.mplstyle")

def loaddata(datafile):
    data = open(datafile)
    line = data.readline()
    samples = []
    while line != "":
        samples.append(list(map(int,line.strip("\n").split(" ")[0:-1])))
        line = data.readline()

    samples = np.array(samples)
    return samples

def correlation(model,datafile,rnnfile):

    datasamples = loaddata(datafile)
    covData = np.corrcoef(datasamples,rowvar = False)
    plt.matshow(covData)
    plt.colorbar()
    plt.savefig("Covariance1")
    plt.clf()
    plt.close()

    rnnCov = np.load(rnnfile)
    plt.matshow(rnnCov)
    plt.colorbar()
    plt.savefig("Covariance2")
    plt.clf()
    plt.close()

    nn_state = PositiveWavefunction.autoload(model)
    new_samples = nn_state.sample(k = 100,num_samples = 10000)

    data = np.array(new_samples)
    cov = np.corrcoef(data,rowvar = False)
    plt.matshow(cov)
    plt.colorbar()
    plt.savefig("Covariance3")
    plt.clf()
    plt.close()

    ylab = r"$\left \langle S_{N/2}^{z}S_{N/2+n}^{z} \right \rangle$"
    row = int(len(cov)/2)
    curve = cov[row,row:-10]
    curveData = covData[row,row:-10]
    rnnData = rnnCov[row,row:-10]
    plt.plot(curve,"o",label = "RBM")
    plt.plot(rnnData,"o",label = "RNN")
    plt.plot(curveData,"o",label = "DMRG")
    plt.xlabel("$n$")
    plt.ylabel(ylab)
    plt.legend()
    plt.savefig("TPCorrelator")
    plt.clf()
    plt.close()

    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(curve,"o",label = "RBM")
    ax.plot(rnnData,"o",label = "RNN")
    ax.plot(curveData,"o",label = "DMRG")
    plt.xlabel("$n$")
    plt.ylabel(ylab)
    plt.legend()
    plt.savefig("TPCorrelatorLog")
    plt.clf()
    plt.close()

model = "Data/TFIM1D/NhStudy/Q100/66/Nh50/model.pt"
datafile = "Samples/TFIM1D/100Q/Samples.txt"
correlation(model,datafile,"cov.npy")
