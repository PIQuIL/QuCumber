import matplotlib.pyplot as plt
import numpy as np

numQubits = [10,15,20,25,30,40,50,60,70,80,90,100,125]
nh = [5,7,10,14,16,22,28,33,39,45,50,57,71]

slope, intercept = np.polyfit(numQubits,nh,1)
lineValues = [slope * i + intercept for i in numQubits]

plt.plot(numQubits,nh,"o")
plt.plot(numQubits,lineValues,"b")
plt.title(r"$N_{h}$ Required for ROE $\leq 2 \cdot 10^{-3}$ with 99% Confidence")
plt.xlabel("Number of Qubits")
plt.ylabel("$N_{h}$")
plt.show()
