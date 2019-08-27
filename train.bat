@echo off
::--iter 20000
set PYTHONPATH=%PYTHONPATH%;.
where wtee.exe
if %ERRORLEVEL% EQU 0 (
%FASTER_RCNN_HOME%\tools\train_net.py --solver .\models\solver.prototxt --imdb gbr_train --iter 20000 --cfg .\models\gbr_rcnn.yml 2>&1 | wtee out\logs\train.log
) else (
%FASTER_RCNN_HOME%\tools\train_net.py --solver .\models\solver.prototxt --imdb gbr_train --iter 20000 --cfg .\models\gbr_rcnn.yml
)
