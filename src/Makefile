default:
	gcc -Wall -O3 -o nn main.c nn.c node.c -lm
	gcc -Wall -O3 -o nntest test.c mnist.c nn.c node.c -lm

clean: 
	rm -f nn