cd ..
rm -rf test/conv/output/*
java -cp ./lib/antlr-2.7.5.jar:./lib/cetus.jar:./lib/gcompiler.jar ece.ncsu.edu.gpucompiler.cuda.KernelDriver -iterator -output=test/conv/output test/conv/conv.c
