@echo off
echo ====================================================
echo STEP 1: Generating Master Plan and Image References
echo ====================================================
python orchestrator_v2.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Image generation failed.
    exit /b %ERRORLEVEL%
)

echo.
echo ====================================================
echo STEP 2: Loading LTX-Video and Generating Scenes
echo ====================================================
python generate_videos.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Video generation failed.
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Entire pipeline completed successfully!
pause
