import setuptools
import sys

# noinspection PyPep8Naming
from setuptools.command.test import test as TestCommand


# See recommendations in https://docs.pytest.org/en/latest/goodpractices.html
class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex

        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


def get_version() -> str:
    fd = {}
    with open("./iww_gvr/__version__.py", "r") as f:
        exec(f.read(), fd)
        return fd["__version__"]


version = get_version()


setuptools.setup(
    name="iww_gvr",
    version=version,
    author="Marco Fabbri, Alvaro Cubi",
    author_email="marco.fabbri@f4e.europa.eu",
    description="Weight window Manipulator & Global Variance Reduction Tool",
    url="git@github.com:Radiation-Transport/iWW-GVR.git",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy >= 1.14.3",
        "pyevtk>=1.1.1",
        "matplotlib>=2.2.2",
        "vtk>=8.1.2",
        "scipy>=1.1.0",
        "tqdm>=4.35.0",
        "pillow",
    ],
    tests_require=["pytest", "pytest-cov>=2.3.1"],
    cmdclass={"test": PyTest},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["iww_gvr = iww_gvr.main:main"]},
)
