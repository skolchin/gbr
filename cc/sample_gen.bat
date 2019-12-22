rmdir s/ /S/Q
mkdir s/
perl createsamples.pl p/positives.txt n/negatives.txt s/ 2000  "C:\Users\kol\Documents\opencv\opencv-3.4.8\build\bin\Release\opencv_createsamples -bgcolor 120 -bgthresh 20 -maxxangle 1.1 -maxyangle 1.1 maxzangle 0.5 -h 40 -w 40 -maxidev 40"
python mergevec.py -v s/p -o samples.vec
