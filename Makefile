default:
	gcc -Wall -O3 -o nn -I./include test/main.c core/nn.c core/node.c algs/backprop.c -lm
	gcc -Wall -O3 -o nntest -I./include test/test.c test/mnist.c core/nn.c core/node.c algs/backprop.c -lm

clean: 
	rm -f nn nntest