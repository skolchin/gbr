rmdir s/ /S/Q
mkdir s/
perl createsamples.pl p/positives.txt n/negatives.txt s/ 7000  "C:\Users\kol\Documents\opencv\opencv-3.4.8\build\bin\Release\opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1 -maxyangle 1.1 maxzangle 0.5 -h 20 -w 20 -maxidev 40"
python mergevec.py -v s/p -o samples.vec
