default:
	gcc -Wall -o nn main.c nn.c node.c -lm

clean: 
	rm -f nn