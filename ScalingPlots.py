import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import os
import Energy as energy
import Peak as peak

# # Some parameters to make the plots look nice
# params = {
#     "text.usetex": True,
#     "font.family": "serif",
#     "legend.fontsize": 14,
#     "axes.labelsize": 16,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
#     "lines.linewidth": 2,
#     "lines.markeredgewidth": 0.8,
#     "lines.markersize": 5,
#     "patch.edgecolor": "black",
# }
# plt.rcParams.update(params)

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

def plotScaling(listQ,models,study,tol,pat,reqs,labels,ratios,fit = False,ratio = 0,c = 0,ax = None):
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

    slopes = []
    for i in range(len(reqs)):
        for m in range(len(models)):

            vals = []
            counter = 0
            passed = False
            for nQ in listQ:
                vals.append([])
                if ratio == 0:
                    Nfolder = "Data/{0}/{1}Study/Q{2}".format(models[m],study,nQ)
                else:
                    Nfolder = "Data/{0}/{1}Study{2}/Q{3}".format(models[m],study,ratio,nQ)
                seeds = [name for name in os.listdir(Nfolder)]
                for seed in seeds:
                    folder = Nfolder + "/{0}".format(seed)
                    trials = [name for name in os.listdir(folder)]
                    trialsAlphaNum = {}
                    for j in range(len(trials)):
                        if study == "Nh" and len(trials[j]) == 3:
                            trialsAlphaNum["Nh0" + trials[j][-1]] = j
                        elif study == "M" and len(trials[j]) == 5:
                            trialsAlphaNum["M0" + trials[j][1:]] = j
                        else:
                            trialsAlphaNum[trials[j]] = j

                    for trial in sorted(trialsAlphaNum):
                        tfile = trials[trialsAlphaNum[trial]]
                        rp = folder + "/" + tfile + "/" + "Results.txt"
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
            if study == "Nh":
                valsM = np.min(vals,axis = 1)
            elif study == "M":
                valsM = np.average(vals,axis = 1)
                valsErr = np.std(vals,axis = 1)
            markers = ["o","s","D" ,"P" ,"X",">"]
            if study == "M":
                lbs = []
                ubs = []
                # for qubits in listQ:
                #     lbs.append(2500)
                #     ubs.append(0)
                # yerr = [lbs,ubs]
                a,b = np.polyfit(listQ,valsM,1)
                lineValues = [a * k + b for k in listQ]
                plt.plot(listQ,lineValues,color = "C{0}".format(c))
                plt.errorbar(listQ,valsM,
                             yerr = valsErr,
                             fmt = "o",
                             capsize = 2,
                             marker = markers[c],
                             color = "C{0}".format(c),
                             label = labels[0])
            elif fit:
                slope,intercept = np.polyfit(listQ,valsM,1)
                slopes.append(slope)
                lineValues = [slope * k + intercept for k in listQ]
                if ax == None:
                    plt.plot(listQ,lineValues,color = "C{0}".format(m))
                    plt.plot(listQ,valsM,"o",
                             label = labels[m],
                             marker = markers[m],
                             color = "C{0}".format(m))
                else:
                    ax.plot(listQ,lineValues,color = "C{0}".format(m))
                    ax.plot(listQ,valsM,"o",
                            label = labels[m],
                            marker = markers[m],
                            color = "C{0}".format(m))
            else:
                plt.plot(listQ,valsM,"-o",label = req)

    if study == "Nh" and ax == None:
        plt.xlabel("$N$")
        plt.ylabel("$N_{h}$")
        title = r"Min $N_{h}$ for various ROE Bounds"
        title += " with 99% CI (Across {0} Trials)".format(len(seeds))
    elif study == "M":
        title = r"Min $M$ for various ROE Bounds"
        title += " with 99% CI (Across {0} Trials)".format(len(seeds))

    if study == "Nh" and ax == None:
        plt.legend()
        plt.savefig("Scaling",dpi = 300)
        plt.clf()
    else:
        ax.legend()

    # Plot slope vs h/J
    # if study == "Nh":
    #     for h in range(len(ratios)):
    #         plt.plot(ratios[h],slopes[h],"bo")
    #
    #     plt.xlabel("h/J")
    #     plt.ylabel("Slope")
    #     plt.savefig("SlopeScaling",dpi = 300)

def illustrateScaling():

    fig, ax1 = plt.subplots()

    # These are in unitless percentages of the figure size.
    # 00 is bottom left
    left, bottom, width, height = [0.55, 0.55, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])

    peak.plotPeak(listQ = [50],
                  models = ["TFIM1D","TFIM1D0p95","TFIM1D0p9",
                            "TFIM1D0p8","TFIM1D0p75","TFIM1D0p7","TFIM1D0p65",
                            "TFIM1D0p6"],
                  tol = 0.0005,
                  pat = 50,
                  req = 0.002,
                  labels = ["$h/J = 1$","$h/J = 0.95$","$h/J = 0.9$",
                            "$h/J = 8$","$h/J = 0.75$","$h/J = 0.7$",
                            "$h/J = 0.65$","$h/J = 0.6$"],
                  ratios = [1,0.95,0.9,0.8,0.75,0.7,0.65,0.6],
                  ax = ax2)

    nh = list(range(5,11))
    roes = [0.02965658,0.02172137,0.01319593,0.00765384,0.00602224,0.00195432]

    ax1.plot(nh,roes,"bo")
    ax1.axhline(0.002,linestyle = "--",color = "r")
    ax1.set_xlabel(r"$N_{h}$")
    ax1.set_ylabel(r"$\epsilon$")
    plt.savefig("ScalingProcedure",dpi = 300)
    plt.clf()

# # Three subplots sharing both x/y axes
# f, (ax1, ax2) = plt.subplots(2,sharex = True,sharey = True)
#
# plotScaling(listQ = list(range(10,91,10)),
#             models = ["TFIM1D","TFIM1D2p0","TFIM1D5p0",
#                       "TFIM1D8p0","TFIM1D10p0","TFIM1D12p0"],
#             study = "Nh",
#             tol = 0.0005,
#             pat = 50,
#             reqs = [0.002],
#             labels = ["$h/J = 1$","$h/J = 2$","$h/J = 5$",
#                       "$h/J = 8$","$h/J = 10$","$h/J = 12$"],
#             ratios = [1,2,5,8,10,12],
#             fit = True,
#             ax = ax1)
#
# plotScaling(listQ = list(range(10,91,10)),
#             models = ["TFIM1D","TFIM1D0p7","TFIM1D0p6","TFIM1D0p2"],
#             study = "Nh",
#             tol = 0.0005,
#             pat = 50,
#             reqs = [0.002],
#             labels = ["$h/J = 1$","$h/J = 0.7$","$h/J = 0.6$","$h/J = 0.2$"],
#             ratios = [1,0.7,0.6,0.2],
#             fit = True,
#             ax = ax2)
#
# f.text(0.5, 0.04, '$N$', ha='center')
# f.text(0.04, 0.5, '$N_{h}$', va='center', rotation='vertical')
# f.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# plt.savefig("Scaling")
# plt.clf()
# plt.close()

# alphas = [["0p5","0p6","0p7","0p8"],[0.5,0.6,0.7,0.8]]
# for i in range(len(alphas[0])):
#     plotScaling(listQ = list(range(10,101,10)),
#                 models = ["TFIM1D"],
#                 study = "M",
#                 tol = 0.0005,
#                 pat = 50,
#                 reqs = [0.002],
#                 ratios = [1],
#                 labels = [r"$\alpha = {0}$".format(alphas[1][i])],
#                 fit = True,
#                 ratio = alphas[0][i],
#                 c = i)
#
# plt.xlabel("$N$")
# plt.ylabel("$M$")
# plt.legend()
# plt.subplots_adjust(left = 0.17)
# plt.savefig("Scaling",dpi = 200)
# plt.clf()

illustrateScaling()
