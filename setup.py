import setuptools
# from glob import glob

# Get the version from wwgvr/__init__.py
fd = {}
with open('./iww_gvr/__version__.py', 'r') as f:
    exec(f.read(), fd)
    version = fd['__version__']

setuptools.setup(
    name="iww_gvr",
    version=version, # __version__
    author="Marco Fabbri, Alvaro Cubi",
    author_email="marco.fabbri@f4e.europa.eu",
    description="Weight window Manipulator & Global Variance Reduction Tool",
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
#    data_files=[glob('doc/*'),'testing'
#    ],
    include_package_data=True,
    install_requires=[
          "numpy >= 1.14.3","pyevtk>=1.1.1","matplotlib>=2.2.2","vtk>=8.1.2","scipy>=1.1.0","tqdm>=4.35.0","pillow",
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={'console_scripts': ['iww_gvr = iww_gvr.main:main']},
)