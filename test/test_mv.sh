cd ..
rm -rf test/mv/output/*
java -cp ./lib/antlr-2.7.5.jar:./lib/cetus.jar:./lib/gcompiler.jar ece.ncsu.edu.gpucompiler.cuda.KernelDriver -output=test/mv/output -merge0=2:-1 -partition test/mv/mv.c
