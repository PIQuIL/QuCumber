========================
Installation
========================

QuCumber only supports Python 3, not Python 2. If you are using Python 2, please update! You will also want to install the following packages if you have not already.

#. Pytorch (https://pytorch.org/)
#. Click (http://click.pocoo.org/5/) - used for command line input
#. tqdm (https://github.com/tqdm/tqdm)

Once you have installed these python packages, click LINK TO ZIP FILE TO DOWNLOAD and download FILENAME.

-------
Windows
-------

`Lol windows <https://stackoverflow.com/questions/4750806/how-do-i-install-pip-on-windows>`

-------------
Linux / MacOS
-------------

Open up a terminal, then type::

    pip install FILENAME

=========
Operation
=========

---------------------------
Positive-real wavefunctions
---------------------------

^^^^^^^^
Training 
^^^^^^^^

QuCumber's positive-real wavefunction training runs given command line arguments. For a list of those command line arguments, type::

    python run_rbm.py train_real --help

This will list all of the adjustable parameters to train on your data. So, for example, if I have a datafile called pickle.txt in my current directory,::

    python run_rbm.py train_real --train-path pickle.txt -n 10 -e 1000 -b 10 -k 1 -lr 0.01 --seed 1234 

We recommend that you leave the number of hidden units (-n), the batch size (-b) and the learning rate (-lr) as their default values if you are not aware of how those parameters affect the training of an RBM. 

^^^^^^^^^^^^^^^^^^^
Generating new data
^^^^^^^^^^^^^^^^^^^

Once the positive-real RBM has been trained, you can generate new data. For a list of commands for generating new data, type::

    python run_rbm.py generate --help

This will list all of the adjustable parameters to generate new data. For example, if I wanted to generate 10000 new data points using 1000 Gibbs steps,::
    
    python run_rbm.py --num-samples 10000 -k 1000

This command will save your newly generated samples to a text file.

--------------------------------------------------
Operation for training complex wavefunctions
--------------------------------------------------

QuCumber's complex wavefunction training is currently in development, although it's available to use. However, we recommend against using this feature at the moment, as it might be unstable.
