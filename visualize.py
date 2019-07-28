import matplotlib as mpl
mpl.use('pdf')
from qucumber.nn_states import PositiveWavefunction
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid.inset_locator import InsetPosition

# Some parameters to make the plots look nice
params = {
    "text.usetex": True,
    "font.family": "serif",
    "legend.fontsize": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2,
    "lines.markeredgewidth": 0.8,
    "lines.markersize": 5,
    "patch.edgecolor": "black",
}
plt.rcParams.update(params)

def plotWeights(model,label,plot = False,threshold = 0):

    # Load model
    nn_state = PositiveWavefunction.autoload(model)

    # Store the parameter names and values to a dictionary
    params = {}
    for name, param in nn_state.named_parameters():
        if param.requires_grad:
            params[name] = param.data

    visibleBias = abs(np.array(params["visible_bias"]))
    hiddenBias = abs(np.array(params["hidden_bias"]))

    weights = params["weights"]
    weights = np.array(weights)

    if label == "h/J = 1":
        plt.figure()
        ax = plt.gca()
        im = ax.matshow(weights)

        # Create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size = "5%",pad = 0.05)
        plt.colorbar(im,cax = cax)
        plt.savefig("WeightMatrix",dpi = 300)
        plt.clf()
        plt.close()

        ax = plt.subplot(111)
        ax.set_xscale("log")
        ax.set_yscale("log")

    markers = ["o","s","D" ,"P" ,"X",">"]
    if label == "h/J = 1":
        marker = markers[0]
    elif label == "h/J = 2":
        marker = markers[1]
    elif label == "h/J = 5":
        marker = markers[2]
    elif label == "h/J = 8":
        marker = markers[3]
    elif label == "h/J = 10":
        marker = markers[4]
    sortedW = abs(weights.flatten())
    sortedW[::-1].sort()
    plt.plot(sortedW,"o",label = "${0}$".format(label),marker = marker)
    if label == "h/J = 10":
        plt.ylabel(r"$\left | W_{ij} \right |$")
        plt.legend()
        plt.savefig("WeightDecay",dpi = 300)

    if plot:
        fig, ax1 = plt.subplots()
        ax1.plot(sortedW,"bo")
        plt.ylabel(r"abs$\left ( W_{ij} \right )$")
        # Create a set of inset Axes: these should fill the bounding box
        # allocated to them. Then manually set the position and relative
        # size of the inset axes within ax1
        ax2 = plt.axes([0,0,1,1])
        ip = InsetPosition(ax1,[0.35,0.3,0.5,0.5])
        ax2.set_axes_locator(ip)
        axins = inset_axes(ax2,
                       width = "5%",
                       height = "100%",
                       loc = "lower left",
                       bbox_to_anchor = (1.05,0,1,1),
                       bbox_transform = ax2.transAxes,
                       borderpad = 0)
        img = ax2.matshow(weights)
        ax2.tick_params(axis = "x",           # changes apply to the x-axis
                        which = "both",       # affect major and minor ticks
                        bottom = False,       # ticks on bottom edge are off
                        top = True,           # ticks on top edge are off
                        labelbottom = False,  # labels on bottom edge are off
                        labeltop = True)      # labels on top edge are off
        fig.colorbar(img,cax = axins)
        plt.savefig("Histogram",dpi = 200)
        plt.clf()

    numLW = np.sum(sortedW > threshold)
    numLVB = np.sum(visibleBias > threshold)
    numLHB = np.sum(hiddenBias > threshold)

    return numLW

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

# Plot weight decay and histogram for paper
plotWeights("Data/TFIM1D/NhStudy/Q60/47/Nh30/model.pt",label = "h/J = 1")
plotWeights("Data/TFIM1D2p0/NhStudy/Q60/13/Nh29/model.pt",label = "h/J = 2")
plotWeights("Data/TFIM1D5p0/NhStudy/Q60/15/Nh25/model.pt",label = "h/J = 5")
plotWeights("Data/TFIM1D8p0/NhStudy/Q60/78/Nh14/model.pt",label = "h/J = 8")
plotWeights("Data/TFIM1D10p0/NhStudy/Q60/15/Nh6/model.pt",label = "h/J = 10")

def thresholdScaling():
    # Plot total number of parameters for multiple thresholds
    thresholds = list(range(1,5))
    nQs = list(range(10,101,10))

    colours = ["b","g","r","c","m"]
    for i in range(len(thresholds)):
        numLPs = []
        for j in range(len(models)):
            numLP = plotWeights(models[j],thresholds[i],plot = False)
            numLPs.append(numLP)

        slope,intercept = np.polyfit(nQs,numLPs,1)
        lineValues = [slope * k + intercept for k in nQs]

        label = r"$\tau = {0}$".format(thresholds[i])
        plt.plot(nQs,numLPs,"o",label = label,color = colours[i])
        plt.plot(nQs,lineValues,color = colours[i])

    plt.legend()
    plt.xlabel("Number of Qubits")
    plt.ylabel("Number of Parameters")
    plt.title(r"Number of Parameters for Various Thresholds $\tau$")
    plt.show()
