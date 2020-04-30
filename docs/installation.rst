========================
Installation
========================

QuCumber only supports Python 3, not Python 2. If you are using Python 2,
please update! You may also want to install PyTorch (https://pytorch.org/),
if you have not already.

If you're running a reasonably up-to-date Linux or macOS system, PyTorch should
be installed automatically when you install QuCumber with `pip`.

-------
GitHub
-------

Navigate to the qucumber page on GitHub (https://github.com/PIQuIL/QuCumber)
and clone the repository by typing::

    git clone https://github.com/PIQuIL/QuCumber.git

Navigate to the main directory and type::

    python setup.py install

-------
Windows
-------

Pytorch is required. Anaconda install (following https://pytorch.org/get-started/locally/) 
and using 64-bit graphical installer works well
(https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe). 
Before you install Anaconda, make sure to have Latex installed, e.g. MikTex (https://miktex.org/download/ctan/systems/win32/miktex/setup/windows-x64/basic-miktex-2.9.7417-x64.exe),
as Matplotlib libraries require Latex for nice visualization in Pytorch

Follow the instructions on Navigate to the directory (through command prompt) where `pip.exe` is installed
(usually :code:`C:\Python\Scripts\pip.exe`) and type::

    pip.exe install qucumber

-------------
Linux / macOS
-------------

Open up a terminal, then type::

    pip install qucumber
