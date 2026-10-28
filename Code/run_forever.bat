@echo off
:loop
echo [%date% %time%] Starting script... >> run_log.txt
python NHL_BETTING_SIMULATION_PREGAME.py
echo [%date% %time%] Script exited. Restarting in 10 seconds... >> run_log.txt
timeout /t 10
goto loop
