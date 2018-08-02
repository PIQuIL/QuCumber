import setuptools
import qucumber

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='qucumber',
    version=qucumber.__version__,
    description='Neural network quantum state tomography.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    install_requires=[
        'torch>=0.4.1,<=0.5',
        'tqdm>=4.23',
        'numpy>=1.14'
    ],
    include_package_data=True,
    url='http://github.com/PIQuIL/QuCumber',
    author='PIQuIL',
    author_email='piquildbeets@gmail.com',
    license='Apache License 2.0',
    packages=setuptools.find_packages(),
    zip_safe=False
)
