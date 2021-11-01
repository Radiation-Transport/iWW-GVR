@echo off
::
:: Jupyter kernel setup
::
::
:: dvp, Nov 2021
::

set app=%1

if "%app%"=="" (
    set app=gvr
)

echo Creating jupyter kernel for conda environment %app%

:: Fix pywin32 version for tornado for python3.9
:: tornado (in jupyter) doesn't work with newer version of pywin, check this on jupyter dependencies updates
:: TODO dvp: check on dependencies updates
:: The following sets version 228 on python3.9 (after pip or poetry it was 300)
:: call conda install pywin32 -y

call conda install jupyterlab -y

:: Create jupyter kernel pointing to the conda environment
call python -m ipykernel install --user --name %app%
if errorlevel 1 (
    echo ERROR: something wrong with installing Jupyter kernel for %app% environment
    set errorlevel=1
) else (
    echo To use %app% environment in jupyter
    echo   - Run 'jupyter lab'
    echo   - Open or create notebook
    echo   - Select kernel %app%
    echo   - check if 'import iww_gvr' in the notebook works
)
