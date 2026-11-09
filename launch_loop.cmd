@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set FIRE_RUNS_DIR=D:\fire_opt\runs
set TMP=D:\fire_opt\tmp
set TEMP=D:\fire_opt\tmp
"C:\Users\qwqw\.workbuddy\binaries\python\envs\default\Scripts\python.exe" "E:\火情监测\火情监测\distill_pipeline\fire_autoopt.py" >> "E:\火情监测\火情监测\distill_pipeline\loop_console.log" 2>&1
