import HyperParams as HP

batchSizes = [2,4,8,16,32,64,128,256,512]
lrs = [0.0001,0.001,0.01,0.1,1]

def RUN(study,numQubits,trial,opt = "SGD"):
    if study == "BatchSizes":
        if numQubits == 10:
            HP.produceDataB(100000,1,numQubits,60000,300,batchSizes,1,trial)
        elif numQubits == 15:
            HP.produceDataB(100000,1,numQubits,70000,300,batchSizes,1,trial)
        elif numQubits == 20:
            HP.produceDataB(100000,1,numQubits,80000,300,batchSizes,1,trial)
        HP.graphDataB(numQubits,trial)
    if study == "LearningRates":
        if numQubits == 10:
            HP.produceData(100000,4,1,numQubits,60000,lrs,opt,300,1,trial)
        if numQubits == 15:
            HP.produceData(100000,4,1,numQubits,70000,lrs,opt,300,1,trial)
        if numQubits == 20:
            HP.produceData(100000,4,1,numQubits,80000,lrs,opt,300,1,trial)
        HP.graphData(numQubits,opt,trial)

RUN("LearningRates",10,1)
HP.graphLR(10,1)
