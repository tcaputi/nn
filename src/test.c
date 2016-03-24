#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include "mnist.h"
#include "nn.h"

#define NR_INPUTS MNIST_IMAGE_SIZE
#define HIDDEN_NPL 300
#define HIDDEN_LAYERS 1
#define NR_OUTPUTS 10

#define LEARNING_RATE 0.025
#define MOMENTUM 0.2

#define PRINT_GENERTION 1000

static double sigmoid(double value){
	return tanh(value);
}

static double sigmoid_derivative(double value){
	return 1.0 - value * value;
}

static void print_top3(struct nn_node *output_nodes, unsigned nr_outputs){
	int i;
	int fidx, sidx, tidx;
	double first = 0, second = 0, third = 0;
	
	for(i = 0; i < nr_outputs; i++){
		if(output_nodes[i].output > first){
			third = second;
			tidx = sidx;
			second = first;
			sidx = fidx;
			first = output_nodes[i].output;
			fidx = i;
		}else if(output_nodes[i].output > second){
			third = second;
			tidx = sidx;
			second = output_nodes[i].output;
			sidx = i;
		}else if(output_nodes[i].output > third){
			third = output_nodes[i].output;
			tidx = i;
		}
	}
	
	printf("TOP 3 CHOICES:\n");
	printf("\t%d : %f\n", fidx, first);
	printf("\t%d : %f\n", sidx, second);
	printf("\t%d : %f\n", tidx, third);
}

int main(int argc, char **argv){
	int ret, i, j;
	struct nn_array_network nn;
	struct mnist *mnist = NULL;
	double weight_limit = sqrt(((double)6) / ((double)(2 * (HIDDEN_NPL + 1))));
	double input[NR_INPUTS];
	double expected[NR_OUTPUTS];
	
	if(argc != 2){
		ret = EINVAL;
		fprintf(stderr, "path to MNIST data dir not provided\n");
		goto error;
	}
	
	printf("initializing array network\n");
	ret = nn_array_network_init(&nn, NR_INPUTS, NR_OUTPUTS, HIDDEN_NPL, HIDDEN_LAYERS, sigmoid, sigmoid_derivative, LEARNING_RATE, MOMENTUM, weight_limit, -weight_limit);
	if(ret) goto error;
	
	printf("loading mnist data\n");
	ret = load_all_mnist_data(argv[1], &mnist);
	if(ret) goto error;
	
	for(i = 0; i < mnist->train->nr_images; i++){
		for(j = 0; j < MNIST_IMAGE_SIZE; j++){
			input[j] = mnist->train->images[i].data[j] / 255.0;
		}
		
		for(j = 0; j < 10; j++){
			if(j == mnist->train->labels[i]) expected[j] = 1.0;
			else expected[j] = 0.0;
		}
		
		nn_array_network_process(&nn, input, expected, NN_MODE_TRAIN);
		
		if(i % PRINT_GENERTION == 0){
			printf("TRAINING: generation %d\n", i);
			printf("\texpected = %u\n", mnist->train->labels[i]);
			print_top3(nn.output_nodes, NR_OUTPUTS);
			printf("\terror (RMS) = %f\n", nn.error);
			printf("\n");
			mnist_image_print(&mnist->train->images[i]);
			printf("\n");
			printf("---------------------------------------\n");
		}
	}
	
	for(i = 0; i < mnist->test->nr_images; i++){
		for(j = 0; j < MNIST_IMAGE_SIZE; j++){
			input[j] = mnist->test->images[i].data[j] / 255.0;
		}
		
		for(j = 0; j < 10; j++){
			if(j == mnist->train->labels[i]) expected[j] = 1.0;
			else expected[j] = 0.0;
		}
		
		nn_array_network_process(&nn, input, expected, NN_MODE_TRAIN);
		
		if(i % PRINT_GENERTION == 0){
			printf("PREDICTION: generation %d\n", i);
			printf("\texpected = %u\n", mnist->train->labels[i]);
			print_top3(nn.output_nodes, NR_OUTPUTS);
			printf("\terror (RMS) = %f\n", nn.error);
			printf("\n");
			mnist_image_print(&mnist->train->images[i]);
			printf("\n");
			printf("---------------------------------------\n");
		}
	}
	
	nn_array_network_destroy(&nn);
	mnist_free(mnist);
	return 0;
	
error:
	fprintf(stderr, "failed to ruun mnist test\n");
	nn_array_network_destroy(&nn);
	if(mnist) mnist_free(mnist);
	return ret;
}