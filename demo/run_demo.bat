@echo off
REM Check if argument is provided
IF "%~1"=="" (
    echo Usage: run_demo.bat video_name
    exit /b 1
)

SET name=%~1

REM Convert MP4 to WAV with mono audio and 16kHz sample rate
ffmpeg -i "demo\%name%.mp4" -ac 1 -ar 16000 "demo\%name%.wav"

REM Run the Python script
python demoLoCoNet_landmark.py --videoName %name%

REM Play the resulting video
ffplay -loop 0 -vf "setpts=1.0*PTS" "demo\%name%\pyavi\video_out.avi"