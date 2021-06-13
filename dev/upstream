#!/bin/bash
#
#  Set up branch upstream/master pointing to master on original iWW-GVR github project.
#

# URL of original github project
url=git@github.com:Radiation-Transport/iWW-GVR.git

# branch to merge from upstream
mbr=${1:-devel}

set -e

git remote | grep upstream > /dev/null || git remote add upstream $url
git fetch upstream master
git checkout master
git branch --list | grep $mbr > /dev/null && git checkout $mbr || git checkout -b $mbr
git merge upstream/master

