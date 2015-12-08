cd ..
rm -rf test/reduction_complex/output/*
java -cp ./lib/antlr-2.7.5.jar:./lib/cetus.jar:./lib/gcompiler.jar ece.ncsu.edu.gpucompiler.cuda.KernelDriver -vectorization -output=test/reduction_complex/output -raw test/reduction_complex/reduction_complex.c
