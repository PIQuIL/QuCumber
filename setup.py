import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

init_file = {}
with open("./qucumber/__version__.py", "r") as f:
    exec(f.read(), init_file)

with open("./requirements.txt", "r") as f:
    install_requires = [req.strip() for req in f.readlines()]

setuptools.setup(
    name="qucumber",
    version=init_file["__version__"],
    description="Neural Network Quantum State Tomography.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    include_package_data=True,
    url="http://github.com/PIQuIL/QuCumber",
    author="PIQuIL",
    author_email="piquildbeets@gmail.com",
    license="Apache License 2.0",
    packages=setuptools.find_packages(),
    zip_safe=False,
)
