import setuptools

# Get the version from wwgvr/__version__.py
fd = {}
with open("./iww_gvr/__version__.py", "r") as f:
    exec(f.read(), fd)
    version = fd["__version__"]
# TODO dvp: why do not just read the file and parse the line with "__version__"? Or even better switch to pyproject.toml


setuptools.setup(
    name="iww_gvr",
    version=version,
    author="Marco Fabbri, Alvaro Cubi",
    author_email="marco.fabbri@f4e.europa.eu",
    description="Weight window Manipulator & Global Variance Reduction Tool",
    packages=setuptools.find_packages(),
    include_package_data=True,  # TODO dvp: is to be False actually
    install_requires=[
        "numpy >= 1.14.3",
        "pyevtk>=1.1.1",
        "matplotlib==3.3.3",
        "vtk>=8.1.2",
        "scipy>=1.1.0",
        "tqdm>=4.35.0",
        "pillow",
        "PyQt5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["iww_gvr = iww_gvr.main:main"]},
)
