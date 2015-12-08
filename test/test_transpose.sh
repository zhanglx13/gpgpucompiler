cd ..
rm -rf test/transpose/output/*
java -cp ./lib/antlr-2.7.5.jar:./lib/cetus.jar:./lib/gcompiler.jar ece.ncsu.edu.gpucompiler.cuda.KernelDriver -merge0=-1:4 -merge1=-1:8 -partition -output=test/transpose/output test/transpose/transpose.c
