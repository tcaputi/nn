#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

#define NODES_PER_LAYER 1
#define LAYERS 1
#define LEARNING_RATE 0.1
#define MOMENTUM 0.1
#define PRINT_GENERTION 1000

#define TRAINING_GENERTIONS 500000
#define PREDICTION_TRIALS 10

static double rnd(void){
	return ((double)rand() / (double)(RAND_MAX));
}

static double sigmoid(double value){
	return 1 / (1 + pow(M_E, -value));
}

static double sigmoid_derivative(double value){
	return 1 / (1 + pow(M_E, -value)) * (1 - (1 / (1 + pow(M_E, -value))));
}

int main(int argc, char **argv){
	int ret, i;
	struct nn_array_network *nn = NULL;
	double weight_limit = sqrt(((double)6) / ((double)(2 * NODES_PER_LAYER)));
	double input[NODES_PER_LAYER];
	double expected, error;
	
	printf("allocating array network\n");
	ret = nn_array_network_alloc(NODES_PER_LAYER, LAYERS, sigmoid, sigmoid_derivative, LEARNING_RATE, MOMENTUM, weight_limit, -weight_limit, &nn);
	if(ret) goto error;
	
	for(i = 0; i < TRAINING_GENERTIONS; i++){
		input[0] = rnd();
		expected = input[0];
		
		nn_array_network_process(nn, input, expected, NN_MODE_TRAIN);
		error = (nn->out_node->output - expected) / expected;
		
		if(i % PRINT_GENERTION == 0){
			printf("TRAINING: generation %d\n", i);
			printf("\tinput = %f\n", input[0]);
			printf("\toutput = %f\n", nn->out_node->output);
			printf("\texpected = %f\n", expected);
			printf("\terror = %f\n", error);
			printf("---------------------------------------\n");
		}
	}
	
	for(i = 0; i < PREDICTION_TRIALS; i++){
		input[0] = rnd();
		expected = input[0];
		
		nn_array_network_process(nn, input, 0, NN_MODE_PREDICT);
		error = (nn->out_node->output - expected) / expected;
		
		printf("PREDICTION: generation %d\n", i);
		printf("\tinput = %f\n", input[0]);
		printf("\toutput = %f\n", nn->out_node->output);
		printf("\texpected = %f\n", expected);
		printf("\terror = %f\n", error);
		printf("---------------------------------------\n");
	}
	
	printf("freeing array network\n");
	nn_array_network_free(nn);
	
	return 0;
	
error:
	fprintf(stderr, "failed to run array network\n");
	if(nn) nn_array_network_free(nn);
	
	return ret;
}