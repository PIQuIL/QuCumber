import torch
import qucumber
from qucumber.positive_wavefunction import PositiveWavefunction
from qucumber.quantum_reconstruction import QuantumReconstruction
import pickle


def generate_visible_space(num_visible, device="cpu"):
    """Generates all possible visible states.

    :returns: A tensor of all possible spin configurations.
    :rtype: torch.Tensor
    """
    space = torch.zeros((1 << num_visible, num_visible),
                        device=device, dtype=torch.double)
    for i in range(1 << num_visible):
        d = i
        for j in range(num_visible):
            d, r = divmod(d, 2)
            space[i, num_visible - j - 1] = int(r)

    return space


def partition(nn_state, visible_space):
    """The natural logarithm of the partition function of the RBM.

    :param visible_space: A rank 2 tensor of the entire visible space.
    :type visible_space: torch.Tensor

    :returns: The natural log of the partition function.
    :rtype: torch.Tensor
    """
    return torch.tensor(
        nn_state.rbm_am.compute_partition_function(visible_space),
        dtype=torch.double, device=nn_state.device
    )


def probability(nn_state, v, Z):
    """Evaluates the probability of the given vector(s) of visible
    units; NOT RECOMMENDED FOR RBMS WITH A LARGE # OF VISIBLE UNITS

    :param v: The visible states.
    :type v: torch.Tensor
    :param Z: The partition function.
    :type Z: float

    :returns: The probability of the given vector(s) of visible units.
    :rtype: torch.Tensor
    """
    return nn_state.psi(v)[0]**2 / Z


def compute_numerical_kl(nn_state, target_psi, vis, Z):
    KL = 0.0
    for i in range(len(vis)):
        KL += ((target_psi[i, 0])**2)*((target_psi[i, 0])**2).log()
        KL -= (((target_psi[i, 0])**2)
               * (probability(nn_state, vis[i], Z)).log().item())
    return KL


def compute_numerical_NLL(nn_state, data, Z):
    NLL = 0
    batch_size = len(data)

    for i in range(batch_size):
        NLL -= (probability(nn_state, data[i], Z).log().item()
                / float(batch_size))

    return NLL


def algorithmic_gradKL(nn_state, target_psi, vis):
    Z = partition(nn_state, vis)
    grad_KL = torch.zeros(nn_state.rbm_am.num_pars,
                          dtype=torch.double, device=nn_state.device)
    for i in range(len(vis)):
        grad_KL += ((target_psi[i, 0])**2)*nn_state.gradient(vis[i])
        grad_KL -= probability(nn_state, vis[i], Z)*nn_state.gradient(vis[i])
    return grad_KL


def algorithmic_gradNLL(qr, data, k):
    # qr.nn_state.set_visible_layer(data)
    return qr.compute_batch_gradients(k, data, data)


def numeric_gradKL(nn_state, target_psi, param, vis, eps):
    num_gradKL = []
    for i in range(len(param)):
        param[i] += eps

        Z = partition(nn_state, vis)
        KL_p = compute_numerical_kl(nn_state, target_psi, vis, Z)

        param[i] -= 2*eps

        Z = partition(nn_state, vis)
        KL_m = compute_numerical_kl(nn_state, target_psi, vis, Z)

        param[i] += eps

        num_gradKL.append((KL_p - KL_m) / (2*eps))

    return num_gradKL


def numeric_gradNLL(nn_state, param, data, vis, eps):
    num_gradNLL = []
    for i in range(len(param)):
        param[i] += eps

        Z = partition(nn_state, vis)
        NLL_p = compute_numerical_NLL(nn_state, data, Z)

        param[i] -= 2*eps

        Z = partition(nn_state, vis)
        NLL_m = compute_numerical_NLL(nn_state, data, Z)

        param[i] += eps

        num_gradNLL.append((NLL_p - NLL_m) / (2*eps))
    return num_gradNLL


def run(qr, target_psi, data, vis, eps, k):
    nn_state = qr.nn_state
    data = data.to(device=nn_state.device)
    vis = vis.to(device=nn_state.device)
    target_psi = target_psi.to(device=nn_state.device)

    alg_grad_KL = algorithmic_gradKL(nn_state, target_psi, vis)
    alg_grad_NLL = algorithmic_gradNLL(qr, data, k)
    num_grad_KL = numeric_gradKL(nn_state, target_psi,
                                 nn_state.rbm_am.weights.view(-1), vis, eps)
    num_grad_NLL = numeric_gradNLL(nn_state, nn_state.rbm_am.weights.view(-1),
                                   data, vis, eps)
    counter = 0
    print("\nTesting weights...")
    print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
    for i in range(len(nn_state.rbm_am.weights.view(-1))):
        print("{: 10.8f}\t{: 10.8f}\t\t"
              .format(num_grad_KL[i], alg_grad_KL[counter].item()),
              end="", flush=True)
        print("{: 10.8f}\t{: 10.8f}\t\t"
              .format(num_grad_NLL[i], alg_grad_NLL[0][i].item()))
        counter += 1

    num_grad_KL = numeric_gradKL(nn_state, target_psi,
                                 nn_state.rbm_am.visible_bias, vis, eps)
    num_grad_NLL = numeric_gradNLL(nn_state, nn_state.rbm_am.visible_bias,
                                   data, vis, eps)
    print("\nTesting visible bias...")
    print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
    for i in range(len(nn_state.rbm_am.visible_bias)):
        print("{: 10.8f}\t{: 10.8f}\t\t"
              .format(num_grad_KL[i], alg_grad_KL[counter].item()),
              end="", flush=True)
        print("{: 10.8f}\t{: 10.8f}\t\t"
              .format(num_grad_NLL[i], alg_grad_NLL[0][counter].item()))
        counter += 1

    num_grad_KL = numeric_gradKL(nn_state, target_psi,
                                 nn_state.rbm_am.hidden_bias, vis, eps)
    num_grad_NLL = numeric_gradNLL(nn_state, nn_state.rbm_am.hidden_bias,
                                   data, vis, eps)

    print("\nTesting hidden bias...")
    print("Numerical KL\tAlg KL\t\t\tNumerical NLL\tAlg NLL")
    for i in range(len(nn_state.rbm_am.hidden_bias)):
        print("{: 10.8f}\t{: 10.8f}\t\t"
              .format(num_grad_KL[i], alg_grad_KL[counter].item()),
              end="", flush=True)
        print("{: 10.8f}\t{: 10.8f}\t\t"
              .format(num_grad_NLL[i], alg_grad_NLL[0][counter].item()))
        counter += 1


if __name__ == '__main__':
    k = 10
    num_chains = 100
    seed = 1234
    with open('test_data.pkl', 'rb') as fin:
        test_data = pickle.load(fin)

    qucumber.set_random_seed(seed, cpu=True, gpu=True, quiet=True)
    train_samples = torch.tensor(test_data['tfim1d']['train_samples'],
                                 dtype=torch.double)
    target_psi = torch.tensor(test_data['tfim1d']['target_psi'],
                              dtype=torch.double)
    nh = train_samples.shape[-1]
    eps = 1.e-6

    nn_state = PositiveWavefunction(num_visible=train_samples.shape[-1],
                                    num_hidden=nh, gpu=True)
    qr = QuantumReconstruction(nn_state)
    vis = generate_visible_space(train_samples.shape[-1])
    run(qr, target_psi, train_samples, vis, eps, k)
