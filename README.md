# iWW-GVR: A tool to manipulate MCNP weight window (WW) and to generate Global Variance Reduction (GVR) parameters

The tool is a python 3.6 based script able to generate global variance reductions weight window (WW) using any mesh format in meshtally by D1SUNED, MCNP5 or MCNP6. Mesh format includes usual MCNP column or matrix format and also specific D1SUNED format including cell or isotope contribution binning and source mesh importance format. Only Cartesian can be read. The tool incorporates also simple functions to operate with weight windows (e.g., analyse, add and remove WW set, write, plot).


BUILDING A NEW VERSION:
1. Install proper tools, from command line execute:

	> python -m pip install --user --upgrade setuptools wheel

2. Go to iww_gvr/iww_gvr: add new scripts and modify main.py accordingly if necessary 
3. Go to iww_gvr parent folder
4. Open 'setup.py'.
5. Modify the version number in the variable 'version' and save.
6. Go to the same folder where setup.py is located and from command line execute:
	
	> python setup.py sdist bdist_wheel
	
7. A new version ready for installation has been built and stored in the "dist" folder.
	
INSTALLATION/UPDATING:
Enter in iww_gvr/dist folder:
	
	> pip install iww_gvr-<version>.tar.gz --user
	
The correspondent version of iww_gvr will be installed and the libraries copied into Pyhton36/site-packages the folder.
Please be aware that the content of the iww_gvr/iww_gvr is neither updated nor modified by the installation.
	
GENERAL EXECUTION:

	> python -m iww_gvr