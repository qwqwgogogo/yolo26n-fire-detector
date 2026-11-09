@echo off
chcp 65001 >nul
set PYTHONUTF8=1
"C:\Users\qwqw\.workbuddy\binaries\python\envs\default\Scripts\python.exe" "E:\火情监测\火情监测\distill_pipeline\fire_launcher.py" >> "E:\火情监测\火情监测\distill_pipeline\loop_console.log" 2>&1
