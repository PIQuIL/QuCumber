import HyperParams as HP

batchSizes = [2,4,8,16,32,64,128,256,512]
lrs = [0.0001,0.001,0.01,0.1,1]
kValues = [1,2,4,8,16]

def RUN(test):
    if test == "BatchSizes":
        HP.produceData(10,100000,batchSizes,0.01,1,20000,torch.optim.SGD,30,1)
