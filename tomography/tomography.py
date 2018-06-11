from rbm import RBM
import torch


class Tomography:
    def __init__(self, data, num_hidden=None, gpu=True, seed=1234):
        self.data = data
        self.rbm = RBM(num_visible=self.data.shape[-1],
                       num_hidden=(self.data.shape[-1]
                                   if num_hidden is None
                                   else num_hidden),
                       gpu=gpu,
                       seed=seed)

    def train(self, epochs, batch_size, **kwargs):
        return self.rbm.train(self.data, epochs, batch_size, **kwargs)

    def overlap(self, target):
        target_psi = torch.tensor(target).to(device=self.rbm.device)
        visible_space = self.rbm.generate_visible_space()
        probs = self.rbm.unnormalized_probability(visible_space)
        Z = self.rbm.partition(visible_space)
        return torch.dot(target_psi, (probs/Z).sqrt())

    @staticmethod
    def _flip_spin(i, s):
        s[:, i] *= -1.0

    @staticmethod
    def _to_pm1(samples):
        return samples.mul(2.).sub(1.)

    @staticmethod
    def _to_01(samples):
        return samples.add(1.).div(2.)

    def energy_1d_tfim(self, h, k, num_samples):
        r"""Compute the energy per spin site for a 1D TFIM chain

        $$\langle E \rangle =
            - \frac{1}{M} \sum_\sigma \sum_i \sigma_i \sigma_{i+1}
            - \frac{h}{M} \sum_\sigma \frac{1}{\sqrt{p(\sigma)}}
                          \sum_i \sqrt{p(\sigma_{-i})} $$
        """
        samples = self._to_pm1(self.rbm.sample(k, num_samples))
        psis = self.rbm.unnormalized_probability(self._to_01(samples)) \
                       .sqrt()

        interaction_term = ((samples[:, :-1] * samples[:, 1:])
                            .sum(1)   # sum over spin sites
                            .mean()   # average the results
                            .item())  # retrieve the numerical value

        flipped_psis = torch.zeros_like(psis)

        for i in range(samples.shape[-1]):  # sum over spin sites
            self._flip_spin(i, samples)  # flip the spin at site i
            flipped_psis += self.rbm.unnormalized_probability(
                                        self._to_01(samples)
                                    ).sqrt()
            self._flip_spin(i, samples)  # flip it back

        transverse_field_term = (flipped_psis
                                 .div(psis)
                                 .mean()   # average over the samples
                                 .item())  # retrieve the numerical value

        return -((interaction_term + (h * transverse_field_term))
                 / samples.shape[-1])
