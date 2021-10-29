# Always prefer setuptools over distutils
from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='iww_gvr',
    version='2.0.0',
    description='Tool to manipulate weight-windows and generate GVRs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='EUL',
    url='https://github.com/Radiation-Transport/iWW-GVR',
    author='Alvaro Cubi',
    keywords='MCNP, radiation, weight-window, gvr',

    packages=['tests', 'iww_gvr'],  # Required
    python_requires='>=3.7',

    install_requires=['numpy',
                      'vtk',
                      'pyevtk',
                      'tqdm',
                      'pyvista'], 
    extras_require={
        'test': ['unittest'],
    },

    entry_points={
        'console_scripts': [
            'iww_gvr = iww_gvr.main:main',
        ],
    },
)
