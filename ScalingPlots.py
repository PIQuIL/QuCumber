import matplotlib.pyplot as plt
import numpy as np

numQubits = [10,20,30,40,50,60,70,80,90,100]
nh = [5,10,16,22,28,33,39,45,50,57]

slope, intercept = np.polyfit(numQubits,nh,1)
lineValues = [slope * i + intercept for i in numQubits]

plt.plot(numQubits,nh,"o")
plt.plot(numQubits,lineValues,"b")
plt.title(r"$Min N_{h}$ Required for ROE $\leq 2 \cdot 10^{-3}$ with 99% Confidence")
plt.xlabel("Number of Qubits")
plt.ylabel("$N_{h}$")
plt.savefig("NhPlot",dpi = 200)
plt.clf()

numQubits = [10,20,30,40,50,60,70,80,90,100]
numSamples = [7500,15000,20000,27500,35000,42500,47500,57500,65000,72500]

slope, intercept = np.polyfit(numQubits,numSamples,1)
lineValues = [slope * i + intercept for i in numQubits]

lowerErrors = []
upperErrors = []
for i in range(len(numQubits)):
    lowerErrors.append(2500)
    upperErrors.append(0)

plt.errorbar(numQubits,numSamples,yerr = [lowerErrors,upperErrors],fmt = "o")
plt.plot(numQubits,lineValues,"b")
plt.title(r"Min M Required for ROE $\leq 2 \cdot 10^{-3}$ with 99% Confidence")
plt.xlabel("Number of Qubits")
plt.ylabel("Number of Samples")
plt.savefig("MPlot",dpi = 200)
