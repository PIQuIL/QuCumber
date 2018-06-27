Theory
======================

Here we will explain the theory behind the code.

compute_batch_gradient
----------------------

- The gradients for the amplitude and the phase are initialized as zero tensors.
- For each element in the batch the number of non-trivial unitaries is counted
- If all the unitaries are trivial, the gradient for the amplitude reads:

.. math::
    \nabla_{\lambda} \log p_{\lambda}(\hat{}\sigma ) / M


Calculating the gradients
-------------------------

If the number of non-trivial unitaries :math:`N_U>0` is not equal zero we construct the following state for all possible :math: `S_j`: (For simplicity we assume there is only one unitary.)

.. math::
        | \bm{\sigma} \rangle = | \sigma_1^{b=z} \dots S_j^{b = z} \sigma_{j+1}^{b=z} \rangle
        


Special triangles
-----------------

There are two special kinds of triangle
for which :mod:`trianglelib` offers special support.

*Equilateral triangle*
    All three sides are of equal length.

*Isosceles triangle*
    Has at least two sides that are of equal length.

These are supported both by simple methods
that are available in the :mod:`trianglelib.utils` module,
and also by a pair of methods of the main
:class:`~trianglelib.shape.Triangle` class itself.

.. _triangle-dimensions:

Example Code
-------------------

example Code

   >>> from trianglelib.shape import Triangle
   >>> t1 = Triangle(3, 4, 5)
   >>> t2 = Triangle(4, 5, 3)
   >>> t3 = Triangle(3, 4, 6)
   >>> print t1 == t2
   True
   >>> print t1 == t3
   False
   >>> print t1.area()
   6.0
   >>> print t1.scale(2.0).area()
   24.0
   >>> blabla

Math example
---------------

math example

.. math::

   a + b > c
   \int dx \partial q
   

While the documentation
for each function in the :mod:`~trianglelib.utils` module
simply specifies a return value for cases that are not real triangles,
the :class:`~trianglelib.shape.Triangle` class is more strict
and raises an exception if your sides lengths are not appropriate:

    >>> from trianglelib.shape import Triangle
    >>> Triangle(1, 1, 3)
    Traceback (most recent call last):
      ...
    ValueError: one side is too long to make a triangle

If you are not sanitizing your user input
to verify that the three side lengths they are giving you are safe,
then be prepared to trap this exception
and report the error to your user.
