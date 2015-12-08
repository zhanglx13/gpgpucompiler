cd ..
rm -rf test/reduction/output/*
java -cp ./lib/antlr-2.7.5.jar:./lib/cetus.jar:./lib/gcompiler.jar ece.ncsu.edu.gpucompiler.cuda.KernelDriver -raw -output=test/reduction/output test/reduction/reduction.c
