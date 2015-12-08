cd ..
rm -rf test/conv_16_16/output/*
java -cp ./lib/antlr-2.7.5.jar:./lib/cetus.jar:./lib/gcompiler.jar ece.ncsu.edu.gpucompiler.cuda.KernelDriver -iterator -output=test/conv_16_16/output test/conv_16_16/conv.c

