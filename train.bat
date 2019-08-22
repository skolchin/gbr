@echo off
set PYTHONPATH=%PYTHONPATH%;.
if "%1%" == "log" (
%FASTER_RCNN_HOME%\tools\train_net.py --solver .\models\solver.prototxt --imdb gbr_train --iter 10000 --cfg .\models\gbr_rcnn.yml 1> .\logs\train.log 2>&1
) else (
%FASTER_RCNN_HOME%\tools\train_net.py --solver .\models\solver.prototxt --imdb gbr_train --iter 10000 --cfg .\models\gbr_rcnn.yml
)
