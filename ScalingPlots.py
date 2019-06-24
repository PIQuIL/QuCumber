import matplotlib.pyplot as plt
import numpy as np
import os
import Energy as energy

def readROEs(resultsfile,nQ):
    '''
    Return list of roe upper bounds from results file.

    :param resultsfile: Name of results file.
    :type resultsfile: str
    :param nQ: Number of qubits.
    :type nQ: int

    :return: List of roe upper bounds.
    :rtype: listof float
    '''

    # Open datafile and skip first 8 lines
    results = open(resultsfile)
    for i in range(8):
        line = results.readline()

    # Store roe upper bounds
    roes = []
    while line != "":
        if nQ == 10:
            roe = float(line.split(" ")[15])
        else:
            roe = float(line.split(" ")[11])
        roes.append(roe)
        line = results.readline()
    results.close()

    return roes

def plotScaling(listQ,study,tol,pat,reqs,fit = False):
    '''
    Plot scaling of number of hidden units or number of samples
    versus system size for various thresholds on the ROE upper bound.

    :param listQ: List of system sizes to check.
    :type listQ: listof int
    :param study: Type of study.
    :type study: anyof "Nh" "M"
    :param tol: Tolerance.
    :type tol: float
    :param pat: Patience.
    :type pat: int
    :params reqs: List of thresholds to consider.
    :type req: listof float

    :returns: None
    '''

    for i in range(len(reqs)):

        vals = []
        counter = 0
        passed = True
        for nQ in listQ:
            vals.append([])
            Nfolder = "Data/{0}Study/Q{1}".format(study,nQ)
            seeds = [name for name in os.listdir(Nfolder)]
            for seed in seeds:
                folder = "Data/{0}Study/Q{1}/{2}".format(study,nQ,seed)
                trials = [name for name in os.listdir(folder)]
                for trial in sorted(trials):
                    rp = folder + "/" + trial + "/" + "Results.txt"
                    roes = readROEs(rp,nQ)
                    result = energy.earlyStopping(roes,tol,pat,reqs[i])
                    if result != False:
                        if result[-2] == "!":
                            vals[counter].append(int(trial[len(study):]))
                            passed = True
                            break
                if not passed:
                    vals[counter].append(100000)
                passed = False
            counter += 1

        vals = np.array(vals)
        valsMin = np.min(vals,axis = 1)
        colours = ["b","g","r","c","m"]
        if fit:
            slope,intercept = np.polyfit(listQ,valsMin,1)
            lineValues = [slope * k + intercept for k in listQ]
            plt.plot(listQ,valsMin,"o",label = reqs[i],color = colours[i])
            plt.plot(listQ,lineValues,color = colours[i])
        else:
            plt.plot(listQ,valsMin,"-o",label = req)

    plt.xlabel("Number of Qubits")
    if study == "Nh":
        plt.ylabel("Number of Hidden Units")
        title = r"Min $N_{h}$ for various ROE Bounds"
        title += " with 99% CI (Across {0} Trials)".format(len(seeds))
    elif study == "M":
        plt.ylabel("Number of Samples")
        title = r"Min $M$ for various ROE Bounds"
        title += " with 99% CI (Across {0} Trials)".format(len(seeds))
    plt.title(title)
    plt.legend()
    plt.show()

plotScaling(listQ = list(range(10,81,10)),
            study = "Nh",
            tol = 0.0005,
            pat = 50,
            reqs = [0.002,0.003,0.004,0.005],
            fit = True)

# numQubits = [10,20,30,40,50,60,70,80,90,100]
# nh = [5,10,16,22,28,33,39,45,50,57]
#
# slope, intercept = np.polyfit(numQubits,nh,1)
# lineValues = [slope * i + intercept for i in numQubits]
#
# plt.plot(numQubits,nh,"o")
# plt.plot(numQubits,lineValues,"b")
# plt.title(r"Min $N_{h}$ Required for ROE $\leq 2 \cdot 10^{-3}$ with 99% Confidence")
# plt.xlabel("Number of Qubits")
# plt.ylabel("$N_{h}$")
# plt.savefig("NhPlot",dpi = 200)
# plt.clf()
#
# numQubits = [10,20,30,40,50,60,70,80,90,100]
# numSamples = [7500,15000,20000,27500,35000,42500,47500,57500,65000,72500]
#
# slope, intercept = np.polyfit(numQubits,numSamples,1)
# lineValues = [slope * i + intercept for i in numQubits]
#
# lowerErrors = []
# upperErrors = []
# for i in range(len(numQubits)):
#     lowerErrors.append(2500)
#     upperErrors.append(0)
#
# plt.errorbar(numQubits,numSamples,yerr = [lowerErrors,upperErrors],fmt = "o")
# plt.plot(numQubits,lineValues,"b")
# plt.title(r"Min $M$ Required for ROE $\leq 2 \cdot 10^{-3}$ with 99% Confidence")
# plt.xlabel("Number of Qubits")
# plt.ylabel("Number of Samples")
# plt.savefig("MPlot",dpi = 200)
