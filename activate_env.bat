@echo off
echo Activating Python venv...

REM Kích hoạt venv bằng đường dẫn tuyệt đối (ổn định 100%)
call "%~dp0venv310\Scripts\activate.bat"

echo Setting nnU-Net environment variables...

set "nnUNet_raw=%~dp0nnUNET_raw"
set "nnUNet_preprocessed=%~dp0nnUNET_preprocessed"
set "nnUNet_results=%~dp0nnUNET_results"

echo Environment is ready!
echo.
echo nnUNet_raw is SET TO: %nnUNet_raw%
echo nnUNet_preprocessed is SET TO: %nnUNet_preprocessed%
echo nnUNet_results is SET TO: %nnUNet_results%

cmd

