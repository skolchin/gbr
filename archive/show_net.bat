@echo off
if "%1" == "" (
set net=train
) else (
set net=%1
)
set PATH=%PATH%;C:\Program Files (x86)\Graphviz2.38\bin
%CAFFE_ROOT%\python\draw_net.py .\models\%net%.prototxt %net%.png --rankdir TB
%net%.png
