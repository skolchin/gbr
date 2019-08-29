@echo off
set PYTHONPATH=%PYTHONPATH%;.
start /B %CAFFE_ROOT%\tools\extra\plot_training_log.py 4 .\out\logs\stat_4.png .\out\logs\train.log
start /B %CAFFE_ROOT%\tools\extra\plot_training_log.py 5 .\out\logs\stat_5.png .\out\logs\train.log
start /B %CAFFE_ROOT%\tools\extra\plot_training_log.py 6 .\out\logs\stat_6.png .\out\logs\train.log
start /B %CAFFE_ROOT%\tools\extra\plot_training_log.py 7 .\out\logs\stat_7.png .\out\logs\train.log
