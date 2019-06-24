from qucumber.nn_states import PositiveWavefunction
import matplotlib.pyplot as plt
import numpy as np

def plotWeights(model):

    # Load model
    nn_state = PositiveWavefunction.autoload(model)

    # Store the parameter names and values to a dictionary
    params = {}
    for name, param in nn_state.named_parameters():
        if param.requires_grad:
            params[name] = param.data

    weights = params["weights"]
    weights = np.array(weights)
    plt.matshow(weights)
    plt.colorbar()
    plt.show()

    sortedW = abs(weights.flatten())
    sortedW[::-1].sort()
    plt.plot(sortedW)
    plt.title("Sorted Abs(Weights)")
    plt.show()

plotWeights("Data/NhStudy/Q60/47/Nh30/model.pt")
