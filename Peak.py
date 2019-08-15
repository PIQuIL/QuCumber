import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import os
import Energy as energy

plt.style.use("apsPeak.mplstyle")

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

def plotPeak(listQ,models,tol,pat,req,labels,ratios,ax = "None"):
    '''
    Plot number of required hidden units for various h/Js for
    fixed number of qubits.

    :param listQ: List of system sizes to check.
    :type listQ: listof int
    :param tol: Tolerance.
    :type tol: float
    :param pat: Patience.
    :type pat: int
    :params reqs: List of thresholds to consider.
    :type req: listof float

    :returns: None
    '''

    for m in range(len(models)):

        vals = []
        counter = 0
        passed = False
        for nQ in listQ:
            vals.append([])
            Nfolder = "Data/{0}/{1}Study/Q{2}".format(models[m],"Nh",nQ)
            seeds = [name for name in os.listdir(Nfolder)]
            for seed in seeds:
                folder = Nfolder + "/{0}".format(seed)
                trials = [name for name in os.listdir(folder)]
                trialsAlphaNum = {}
                for j in range(len(trials)):
                    if len(trials[j]) == 3:
                        trialsAlphaNum["Nh0" + trials[j][-1]] = j
                    else:
                        trialsAlphaNum[trials[j]] = j

                for trial in sorted(trialsAlphaNum):
                    tfile = trials[trialsAlphaNum[trial]]
                    rp = folder + "/" + tfile + "/" + "Results.txt"
                    roes = readROEs(rp,nQ)
                    result = energy.earlyStopping(roes,tol,pat,req)
                    if result != False:
                        if result[-2] == "!":
                            vals[counter].append(int(trial[len("Nh"):]))
                            passed = True
                            break

                if not passed:
                    vals[counter].append(100000)
                passed = False
            counter += 1

        vals = np.array(vals)
        valsM = np.min(vals,axis = 1)
        if ax == "None":
            plt.plot(ratios[m],valsM[0],
                     color = "C1",
                     marker = "o",
                     label = labels[m])
        else:
            ax.plot(ratios[m],valsM[0],
                    color = "C1",
                    marker = "o",
                    label = labels[m])

    if ax == "None":
        plt.xlabel("$h/J$")
        plt.ylabel("$N_{h}$")
        plt.savefig("Peak")
    else:
        ax.set_xlabel("$h/J$",fontsize = 6)
        ax.set_ylabel("$N_{h}$",fontsize = 6)

def plot():
    plotPeak(listQ = [50],
             models = ["TFIM1D","TFIM1D0p95","TFIM1D0p9",
                       "TFIM1D0p8","TFIM1D0p75","TFIM1D0p7","TFIM1D0p65",
                       "TFIM1D0p6"],
            tol = 0.0005,
            pat = 50,
            req = 0.002,
            labels = ["$h/J = 1$","$h/J = 0.95$","$h/J = 0.9$",
                      "$h/J = 8$","$h/J = 0.75$","$h/J = 0.7$",
                      "$h/J = 0.65$","$h/J = 0.6$"],
            ratios = [1,0.95,0.9,0.8,0.75,0.7,0.65,0.6])
