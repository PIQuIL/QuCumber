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

    def train(self, **kwargs):
        return self.rbm.train(**kwargs)

    def overlap(self, target):
        target_psi = torch.tensor(target).to(device=self.rbm.device)
        visible_space = self.rbm.generate_visible_space()
        probs = self.rbm.unnormalized_probability(visible_space)
        Z = self.rbm.partition(visible_space)
        return torch.dot(target_psi, (probs/Z).sqrt())

    @staticmethod
    def _flip_spin(i, spin_config):
        s = spin_config.clone()
        s[:, i] *= -1.0
        return s

    @staticmethod
    def _to_pm1(samples):
        return (2.0 * samples) - 1.0

    def energy_1d_tfim(self, h, k, num_samples):
        samples = self._to_pm1(self.rbm.sample(k, num_samples))
        probs = self.rbm.unnormalized_probability(samples)

        interaction_term = ((samples[:, 1:] * samples[:, :-1])
                            .sum(1).mean().item())

        flipped_probs = torch.zeros_like(probs)

        for i in range(samples.shape[-1]):
            flipped_probs += self.rbm.unnormalized_probability(
                self._flip_spin(i, samples))

        flipped_probs /= num_samples

        transverse_field_term = ((flipped_probs / probs)
                                 .sqrt().sum().item())

        return -(interaction_term + (h * transverse_field_term))
