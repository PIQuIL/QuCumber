import HyperParams as HP
import torch

batchSizes = [16,32]
lrs = [0.001,0.01]
kValues = [2,4]
opts = [torch.optim.SGD,torch.optim.Adam]

def RUN(test):
    if test == "BatchSizes":
        HP.produceData(5,100000,batchSizes,0.01,1,20000,torch.optim.SGD,15,1)
        HP.graphData("BatchSizes",5,1,"Learning Curve")
    elif test == "GibbsSampling":
        HP.produceData(5,100000,16,0.01,kValues,20000,torch.optim.SGD,15,1)
        HP.graphData("GibbsSampling",5,1,"Learning Curve")
    elif test == "LearningRates":
        HP.produceData(5,100000,16,lrs,4,20000,torch.optim.SGD,15,1)
        HP.graphData("LearningRates",5,1,"Learning Curve")
    elif test == "Optimizers":
        HP.produceData(5,100000,16,lrs,4,20000,opts,15,1)
        HP.graphData("Optimizers",5,1,"Learning Curve")
    elif test == "TryThis":
        HP.produceData(5,100000,16,0.01,1,20000,torch.optim.SGD,15,1)
        HP.graphData("TryThis",5,1,"Learning Curve")
