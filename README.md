![GitHub last commit](https://img.shields.io/github/last-commit/Radiation-Transport/iWW-GVR)
![GitHub issues](https://img.shields.io/github/issues/Radiation-Transport/iWW-GVR)
![GitHub closed issues](https://img.shields.io/github/issues-closed-raw/Radiation-Transport/iWW-GVR)
![GitHub top language](https://img.shields.io/github/languages/top/Radiation-Transport/iWW-GVR)
![](https://img.shields.io/badge/license-EU%20PL-blue)

# iWW-GVR: A tool to manipulate MCNP weight window (WW) and to generate Global Variance Reduction (GVR) parameters

The tool is a python 3.6 based script able to generate global variance reductions weight window (WW) using any mesh format in meshtally by D1SUNED, MCNP5 or MCNP6. Mesh format includes usual MCNP column or matrix format and also specific D1SUNED format including cell or isotope contribution binning and source mesh importance format. Only Cartesian can be read. The tool incorporates also simple functions to operate with weight windows (e.g., analyse, add and remove WW set, write, plot).


## BUILDING A NEW VERSION:
1. Install proper tools, from command line execute:

	> python -m pip install --user --upgrade setuptools wheel

2. Go to iww_gvr/iww_gvr: add new scripts and modify main.py accordingly if necessary 
3. Go to iww_gvr parent folder
4. Open '__version__.py'.
5. Modify the version number in the variable 'version' and save.
6. Go to iww_gvr parent folder
7. Go to the same folder where setup.py is located and from command line execute:

    > python setup.py sdist bdist_wheel clean --all install clean --all
	

	
## INSTALLATION/UPDATING:
Enter in iww_gvr/dist folder:
	
	> pip install iww_gvr-<version>.tar.gz --user
	
The correspondent version of iww_gvr will be installed and the libraries copied into Pyhton36/site-packages the folder.
Please be aware that the content of the iww_gvr/iww_gvr is neither updated nor modified by the installation.
	
## GENERAL EXECUTION:

	> python -m iww_gvr
    
    
## LICENSE
Copyright 2019 F4E | European Joint Undertaking for ITER and the Development of Fusion Energy (‘Fusion for Energy’). Licensed under the EUPL, Version 1.2 or - as soon they will be approved by the European Commission - subsequent versions of the EUPL (the “Licence”). You may not use this work except in compliance with the Licence. You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl.html   
Unless required by applicable law or agreed to in writing, software distributed under the Licence is distributed on an “AS IS” basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the Licence permissions and limitations under the Licence.

   
