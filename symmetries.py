import matplotlib as mpl
mpl.use('pdf')
from qucumber.nn_states import PositiveWavefunction
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("apsSymmetry.mplstyle")

def plotWeights(model):

    # Load model
    nn_state = PositiveWavefunction.autoload(model)

    # Store the parameter names and values to a dictionary
    params = {}
    for name, param in nn_state.named_parameters():
        if param.requires_grad:
            params[name] = param.data

    visibleBias = np.array(params["visible_bias"])
    hiddenBias = np.array(params["hidden_bias"])
    weights = np.array(params["weights"])

    rowSums = np.sum(weights,axis = 1)
    colSums = np.sum(weights,axis = 0)

    fig,axes = plt.subplots(1,2)
    label1 = "$\sum_{j}{W_{ij}}$"
    label2 = "$b_{i}$"
    label3 = "{0}/{1}".format(label1,label2)
    xpoints = list(range(len(colSums)))
    # ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    axes[0].plot(xpoints,colSums,label = label1)
    axes[0].plot(xpoints,visibleBias,label = label2)
    axes[0].plot(xpoints,colSums/visibleBias,"r--",label = label3)
    axes[0].axhline(0,color = "k")
    axes[0].set_ylim(-8,8)
    axes[0].legend()
    axes[0].set_xlabel("$i$")

    label1 = "$\sum_{i}{W_{ij}}$"
    label2 = "$c_{j}$"
    label3 = "{0}/{1}".format(label1,label2)
    xpoints = list(range(len(rowSums)))
    # ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    axes[1].plot(xpoints,rowSums,label = label1)
    axes[1].plot(xpoints,hiddenBias,label = label2)
    axes[1].plot(xpoints,rowSums/hiddenBias,"r--",label = label3)
    axes[1].axhline(0,color = "k")
    axes[1].set_ylim(-14,14)
    axes[1].legend()
    axes[1].set_xlabel("$j$")
    plt.savefig("Symmetries")
    plt.clf()
    plt.close()

    vbiases = []
    hbiases = []
    for i in range(len(weights)):
        vindex = np.argmax(abs(weights[i]))
        vbias = visibleBias[vindex]
        hbias = hiddenBias[i]
        vbiases.append(vbias)
        hbiases.append(hbias)
    vbiases = np.array(vbiases)
    hbiases = np.array(hbiases)

    label1 = "$b_{j}^{Strong}$"
    label2 = "$c_{i}$"
    label3 = "{0}/{1}".format(label1,label2)
    xpoints = list(range(len(vbiases)))
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    ax.plot(xpoints,vbiases,label = label1)
    ax.plot(xpoints,hbiases,label = label2)
    ax.plot(xpoints,vbiases/hbiases,"r--",label = label3)
    ax.axhline(0,color = "k")
    plt.legend()
    plt.xlabel("Row")
    plt.savefig("SymmetryVH")
    plt.clf()
    plt.close()

models = ["Data/TFIM1D/NhStudy/Q10/39/Nh5/model.pt",
          "Data/TFIM1D/NhStudy/Q20/22/Nh10/model.pt",
          "Data/TFIM1D/NhStudy/Q30/38/Nh14/model.pt",
          "Data/TFIM1D/NhStudy/Q40/12/Nh19/model.pt",
          "Data/TFIM1D/NhStudy/Q50/16/Nh25/model.pt",
          "Data/TFIM1D/NhStudy/Q60/47/Nh30/model.pt",
          "Data/TFIM1D/NhStudy/Q70/24/Nh35/model.pt",
          "Data/TFIM1D/NhStudy/Q80/29/Nh40/model.pt",
          "Data/TFIM1D/NhStudy/Q90/55/Nh45/model.pt",
          "Data/TFIM1D/NhStudy/Q100/66/Nh50/model.pt"]

plotWeights(models[3])
