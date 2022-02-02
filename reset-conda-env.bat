@echo off
::
:: Prepare conda environment for app development on Windows.
::
:: dvp, Nov 2021
::

set app_version=1.8

if "%1"=="--help" (
    echo.
    echo Usage:
    echo.
::    echo set-conda-env conda_env install_tool python_version
    echo set-conda-env conda_env python_version
    echo.
    echo All the parameters are optional.
    echo.
    echo Defaults:
    echo   conda_env=gvr%app_version%
::    echo   install_tool=pip   another valid value: poetry
    echo   python_version=3.9
    echo.
    goto END
)

set app=%1
shift
if "%app%"=="" set app=gvr%app_version%


# set install_tool=%1
# shift
set install_tool=

if "%install_tool%"=="" set install_tool=poetry

if "%install_tool%"=="poetry" (
    call poetry --version > NUL
    if errorlevel 1 (
        echo ERROR\: Poetry is not available
        echo        See poetry install instructions: https://python-poetry.org
        goto END
    )
) else (
    if "%install_tool%" NEQ "pip" (
        echo ERROR\: unknown install tool %install_tool%. Should be either `pip` or `poetry`
        goto END
    )
)


set python_version=%1
shift
if "%python_version%"=="" (
    set python_version=3.8
)


echo Installing conda environment %app% with %install_tool%

call conda deactivate
call conda activate
call conda env remove -n %app% -q -y
call conda create -n %app% python=%python_version% -q -y
call conda activate %app%

if "%install_tool%"=="pip" (
    pip install .
    pip install -r requirements-dev.txt
) else (
    ::   this creates egg-link in the environment to current directory (development install)
    call poetry install
)
if errorlevel 1  (
    echo "ERROR: failed to run install with %install_tool%"
    goto END
)

:: app --version
:: if errorlevel 1  (
::     echo "ERROR: failed to install app"
::     goto END
:: )
:: echo.

echo SUCCESS: app has been installed
echo.


pytest -m "not slow"
if errorlevel 1 (
    echo ERROR: failed to run tests
    goto END
)
echo.
echo SUCCESS: pytest is passed OK
echo.



:: if "%install_tool%"=="poetry" (
::     :: verify nox
::     nox --list
::     :: safety first - run this on every dependency addition or update
::     :: test often - who doesn't?
::     nox -s safety -s tests -p %python_version% -- -m "not slow" --cov
::     call poetry build
::     if errorlevel 1 (
::         echo ERROR: failed to run poetry build
::         goto END
::     )
:: ) else (
::     pip install .
::     if errorlevel 1 (
::         echo ERROR: failed to collect dependencies with pip
::         goto END
::     )
:: )

call create-jk %app%
if errorlevel 1 (
    goto END
)

echo.
echo SUCCESS!
echo --------
echo.
:: echo Usage:
:: echo.
:: %app% --help
:: echo.
echo Conda environment %app% is all clear.
echo Set your IDE to use %CONDA_PREFIX%\python.exe
echo.
echo Enjoy!
echo.


:END
