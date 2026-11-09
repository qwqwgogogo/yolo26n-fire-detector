@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set FIRE_RUNS_DIR=D:\fire_opt\runs
set TMP=D:\fire_opt\tmp
set TEMP=D:\fire_opt\tmp
if not exist "D:\fire_opt\tmp" mkdir "D:\fire_opt\tmp"
if not exist "D:\fire_opt\runs" mkdir "D:\fire_opt\runs"
"C:\Users\qwqw\.workbuddy\binaries\python\envs\default\Scripts\python.exe" "E:\火情监测\火情监测\distill_pipeline\exp_train_eval.py" "E:\火情监测\火情监测\distill_pipeline\config_expE.json" >> "E:\火情监测\火情监测\distill_pipeline\loop_expE.log" 2>&1
"C:\Users\qwqw\.workbuddy\binaries\python\envs\default\Scripts\python.exe" "E:\火情监测\火情监测\distill_pipeline\fire_launcher.py" >> "E:\火情监测\火情监测\distill_pipeline\loop_console.log" 2>&1
