import HyperParams as HP

batchSizes = [2,4,8,16,32,64,128,256,512]

def RUN(study,numQubits,trial):
    if study == "BatchSizes":
        if numQubits == 10:
            HP.produceDataB(100000,1,numQubits,60000,300,batchSizes,1,trial)
        elif numQubits == 15:
            HP.produceDataB(100000,1,numQubits,70000,300,batchSizes,1,trial)
        elif numQubits == 20:
            HP.produceDataB(100000,1,numQubits,80000,300,batchSizes,1,trial)
        HP.graphDataB(numQubits,trial)
