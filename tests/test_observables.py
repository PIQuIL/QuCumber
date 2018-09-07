import unittest
import qucumber.observables as observables
from qucumber.nn_states import ComplexWavefunction

class TestPauli(unittest.TestCase):
    def test_apply(self):
        test_psi = ComplexWavefunction(2, num_hidden=3)
        test_sample = test_psi.sample(100, num_samples=1000)
        X = observables.SigmaX()
        X.apply(test_psi, test_sample)


if __name__ == '__main__':
    unittest.main()
