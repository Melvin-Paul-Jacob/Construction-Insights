@echo off

echo Creating virtual environment...
py -3.10 -m venv ppedet

echo Activating virtual environment...
call .\ppedet\Scripts\activate

echo Upgrading pip...
py -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Setup complete.
echo Activate the environment with:
echo .\ppedet\Scripts\activate 

pause