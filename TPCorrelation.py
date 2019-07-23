from qucumber.nn_states import PositiveWavefunction
import matplotlib.pyplot as plt
import numpy as np

def correlation(model):
    nn_state = PositiveWavefunction.autoload(model)
    new_samples = nn_state.sample(k = 100,num_samples = 10000)

    data = np.array(new_samples)
    cov = np.corrcoef(data,rowvar = False)
    plt.matshow(cov)
    plt.colorbar()
    plt.show()

    row = int(len(cov)/2)
    curve = cov[row,row:]
    plt.plot(curve,"bo")
    plt.show()

    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(curve,"bo")
    plt.show()

model = "Data/TFIM1D/NhStudy/Q50/16/Nh25/model.pt"
correlation(model)
