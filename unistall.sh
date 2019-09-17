#!/bin/bash 

#
#
# Installs a link to current directory. 
# Thus, changes in any py files do not require
# reinstallation. 
#
# However, the numjuggler script is generated
# automatically and contains current version numger.
# If this number is chaged in setup.py, the
# package should be reinstalled, in order
# to update the numjuggler script.


# Uninstall previous version
pip uninstall iww-gvr;

