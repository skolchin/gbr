@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
set /A i=80
for %%f in ("IMG-20*") do (
echo %%f "->" go_board_!i!.png
ren %%f go_board_!i!.png
set /A i=i+1
)
