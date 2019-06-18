import sys
import itertools
import Energy as energy

TASK_ID = int(sys.argv[1]) # 0 refers to the script name

Ns = list(range(10, 101, 10))
seeds = [777,888,999]

# Number of jobs is len(tasks)
tasks = list(itertools.product(Ns, seeds))
parameter = tasks[TASK_ID]
N = parameter[0]
seed = parameter[1]

if N == 10:
    sf = True
else:
    sf = False

energy.trainEnergy(numQubits = N,
                   nh = 10,
                   numSamples1 = 100000,
                   numSamples2 = 5000,
                   burn_in = 500,
                   steps = 100,
                   mT = 10000,
                   storeFidelities = sf,
                   model = "TFIM1D",
                   earlyStoppingParams = [0.0005,10,0.0005],
                   seeds = [seed],
                   study = "Nh")
