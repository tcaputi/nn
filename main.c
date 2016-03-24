#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"

#define NODES_PER_LAYER 4
#define LAYERS 2
#define LEARNING_RATE 0.15
#define MOMENTUM 0.5
#define PRINT_GENERTION 50

#define TRAINING_GENERTIONS 2000
#define PREDICTION_TRIALS 10

static double rnd(void){
	return ((double)rand() / (double)(RAND_MAX));
}

static double sigmoid(double value){
	//return 1 / (1 + pow(M_E, -value));
	return tanh(value);
}

static double sigmoid_derivative(double value){
	//return 1 / (1 + pow(M_E, -value)) * (1 - (1 / (1 + pow(M_E, -value))));
	return 1.0 - value * value;
}

int main(int argc, char **argv){
	int ret, i;
	struct nn_array_network *nn = NULL;
	double weight_limit = sqrt(((double)6) / ((double)(2 * NODES_PER_LAYER)));
	double input[NODES_PER_LAYER];
	double expected;
	
	printf("allocating array network\n");
	ret = nn_array_network_alloc(NODES_PER_LAYER, LAYERS, sigmoid, sigmoid_derivative, LEARNING_RATE, MOMENTUM, weight_limit, -weight_limit, &nn);
	if(ret) goto error;
	
	for(i = 0; i < TRAINING_GENERTIONS; i++){
		input[0] = (rnd() >= 0.5) ? 1 : 0;
		input[1] = (rnd() >= 0.5) ? 1 : 0;
		input[2] = 0;
		input[3] = 0;
		expected = (double)((int)input[0] ^ (int)input[1]);
		
		nn_array_network_process(nn, input, expected, NN_MODE_TRAIN);
		
		if(i % PRINT_GENERTION == 0){
			printf("TRAINING: generation %d\n", i);
			printf("\tinput = %f, %f\n", input[0], input[1]);
			printf("\texpected = %f\n", expected);
			printf("\toutput = %f\n", nn->out_node->output);
			printf("\terror (RMS) = %f\n", nn->error);
			printf("---------------------------------------\n");
		}
	}
	
	for(i = 0; i < PREDICTION_TRIALS; i++){
		input[0] = (rnd() >= 0.5) ? 1 : 0;
		input[1] = (rnd() >= 0.5) ? 1 : 0;
		input[2] = 0;
		input[3] = 0;
		expected = (double)((int)input[0] ^ (int)input[1]);
		
		nn_array_network_process(nn, input, expected, NN_MODE_PREDICT);
		
		printf("PREDICTION: generation %d\n", i);
		printf("\tinput = %f, %f\n", input[0], input[1]);
		printf("\texpected = %f\n", expected);
		printf("\toutput = %f\n", nn->out_node->output);
		printf("\terror (RMS) = %f\n", nn->error);
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