========================
Installation
========================

QuCumber only supports Python 3 (specifically, 3.6 and up), not Python 2.
If you are using Python 2, please update! You may also want to install PyTorch
(https://pytorch.org/), if you have not already.

If you're running a reasonably up-to-date Linux or macOS system, PyTorch should
be installed automatically when you install QuCumber with `pip`.

-------
Windows
-------

Windows 10 is recommended. PyTorch is required (following
https://pytorch.org/get-started/locally/). One way for getting PyTorch is having
Anaconda installed first and using the 64-bit graphical installer
(https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe).

**Before** you install Anaconda, make sure to have a LaTeX distribution installed,
for example MiKTeX
(https://miktex.org/download/ctan/systems/win32/miktex/setup/windows-x64/basic-miktex-2.9.7417-x64.exe),
as Matplotlib libraries require LaTeX for nice visualization in PyTorch

After the Anaconda installation, follow specific instructions on
https://pytorch.org/get-started/locally/ to get the correct
installation command for PyTorch, which is CUDA/Python version dependent. For
example, if your system does not have a GPU card, you will need the CPU version::

    conda install pytorch torchvision cpuonly -c pytorch

 To install QuCumber on Anaconda, start the Anaconda prompt,
 or navigate to the directory (through command prompt) where `pip.exe`
 is installed (usually :code:`C:\Python\Scripts\pip.exe`) and then type::

     pip.exe install qucumber


-------------
Linux / macOS
-------------

Open up a terminal, then type::

    pip install qucumber


-------
GitHub
-------

Navigate to the qucumber page on GitHub (https://github.com/PIQuIL/QuCumber)
and clone the repository by typing::

    git clone https://github.com/PIQuIL/QuCumber.git

Navigate to the main directory and type::

    python setup.py install
