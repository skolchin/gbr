@echo off
set PYTHONPATH=%PYTHONPATH%;.
if "%1%" == "no_log" (
%FASTER_RCNN_HOME%\tools\train_net.py --solver .\models\solver.prototxt --imdb gbr_train --cfg .\models\gbr_rcnn.yml
) else (
%FASTER_RCNN_HOME%\tools\train_net.py --solver .\models\solver.prototxt --imdb gbr_train --cfg .\models\gbr_rcnn.yml 1> .\logs\train.log 2>&1
)
