Theory
======================

Here we will explain the theory behind the code.

compute_batch_gradient
----------------------

- The gradients for the amplitude and the phase are initialized as
  zero tensors.
- For each element in the batch the number of non-trivial unitaries is counted
- If all the unitaries are trivial, the gradient for the amplitude reads:

.. math::
    \frac{1}{M}\nabla_{\lambda} \log p_{\lambda}(\hat{\bm\sigma})


Calculating the gradients
-------------------------

If the number of non-trivial unitaries :math:`N_U>0` is not equal zero we
construct the following state for all possible :math:`S_j`
(for simplicity we assume there is only one unitary):

.. math::
        \ket{\bm{\sigma}} = \ket{\sigma_1^{b=z} \dots S_j^{b=z} \sigma_{j+1}^{b=z}}
