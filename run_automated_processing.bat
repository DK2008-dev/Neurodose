@echo off
echo ================================================================================
echo AUTOMATED FULL DATASET PROCESSING
echo ================================================================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Checking Python environment...
python -c "import sys; print(f'Python: {sys.executable}')"
python -c "import mne; print(f'MNE version: {mne.__version__}')" 2>nul || (
    echo ERROR: MNE not found. Installing requirements...
    pip install -r requirements.txt
)

echo.
echo Starting automated processing...
python automated_full_processing.py

echo.
echo Processing completed. Press any key to exit...
pause >nul
