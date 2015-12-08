cd ..
rm -rf test/vectormul/output/*
java -cp ./lib/antlr-2.7.5.jar:./lib/cetus.jar:./lib/gcompiler.jar ece.ncsu.edu.gpucompiler.cuda.KernelDriver -output=test/vectormul/output test/vectormul/vectormul.c
