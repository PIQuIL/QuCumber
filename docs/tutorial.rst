========================
Installation
========================

QuCumber only supports Python 3, not Python 2. If you are using Python 2, please update! You will also want to install the following packages if you have not already.

#. Pytorch (https://pytorch.org/)
#. tqdm (https://github.com/tqdm/tqdm)

-------
Windows
-------

Navigate to the directory (through command prompt) where pip.exe is installed (usually C:\\Python\\Scripts\\pip.exe) and type::
    
    pip.exe install qucumber

-------------
Linux / MacOS
-------------

Open up a terminal, then type::

    pip install qucumber

------------
Operation
------------

To make sure everything is installed correctly, open a python shell / document and type::

    from qucumber import *

To begin training an RBM on your data, you must initialize a BinomialRBM object with the number of visible units (i.e. sites). For example, given the number of sites, *num_visible*, ::
    
    rbm = BinomialRBM(num_visible)

Then, one can now train with ::

    rbm.fit(your_datafile)

Please ensure that your input data is a numpy array or torch tensor. Additional arguments to *fit* can be found here :meth:`qucumber.rbm.BinomialRBM.fit`.

Once your RBM has been trained, you can generate more data by doing the following.::

    rbm.load('saved_params.pkl')
    num_samples = 1000
    new_data = (rbm.sample(num_samples)).data.cpu().numpy()
    np.savetxt('generated_samples.txt', new_data, fmt='%d')

An additional argument to the *sample* function is the number of Gibbs steps, *k*. The default value of *k* is 10. See :meth:`qucumber.rbm.BinomialRBM.sample` for more information. 
