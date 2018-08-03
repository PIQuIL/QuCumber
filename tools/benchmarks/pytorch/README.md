# PyTorch RBM code

Run `RBM_cleaned_runner.py`. Helper file is automatically loaded.

The files `target_psi.txt` and `trainin_data.txt` have to be in the same folder as the runner and helper file

To use GPU support set `gpu = True`

The 'Dummy Training Set' is for a simple first test of the RBM and the overlapp function. It does not work with GPU.To use it set `dummy_training = True`.

## The Code has been tested with the following specs:
### GPU and CPU tested on

python 2.7.15

torch 0.4.0

numpy 1.14.2

### CPU tested on

python 3.6.4

torch 0.3.1.post2

numpy 1.13.3
