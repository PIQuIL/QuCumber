import setuptools
import qucumber

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='qucumber',
      version=qucumber.__version__,
      description='Neural network quantum state tomography.',
      long_description=log_description,
      classifiers=[
        'License :: Apache License 2.0',
        'Programming language :: Python :: 3',
        'Operating System :: OS Independent'
      ],
      install_requires=['torch','tqdm'],
      include_package_data=True,
      url='http://github.com/MelkoCollective/QuCumber',
      author='PIQuIL',
      license='Apache License 2.0',
      packages=setuptools.find_packages()
)
