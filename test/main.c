#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <nn.h>
#include <backprop.h>

#define NR_INPUTS 2
#define HIDDEN_NPL 4
#define HIDDEN_LAYERS 1
#define NR_OUTPUTS 1

#define LEARNING_RATE 0.15
#define MOMENTUM 0.5

#define PRINT_GENERTION 50
#define TRAINING_GENERTIONS 2000
#define PREDICTION_TRIALS 10

static double rnd(void){
	return ((double)rand() / (double)(RAND_MAX));
}

static double calculate_error(struct nn_array_network *nn, double *expected){
	int i;
	double total = 0;
	
	for(i = 0; i < nn->nr_outputs; i++){
		total += (expected[i] - nn->output_nodes[i].output) * (expected[i] - nn->output_nodes[i].output);
	}
	
	return sqrt(total / nn->nr_outputs);
}

int main(int argc, char **argv){
	int ret, i;
	struct nn_array_network nn;
	double weight_limit = sqrt(((double)6) / ((double)(2 * (HIDDEN_NPL + 1))));
	double input[NR_INPUTS];
	double expected[NR_OUTPUTS];
	
	printf("initializing array network\n");
	ret = nn_array_network_init(&nn, NR_INPUTS, NR_OUTPUTS, HIDDEN_NPL, HIDDEN_LAYERS, &backprop_node_ops, calculate_error, LEARNING_RATE, MOMENTUM, weight_limit, -weight_limit);
	if(ret) goto error;
	
	for(i = 0; i < TRAINING_GENERTIONS; i++){
		input[0] = (rnd() >= 0.5) ? 1 : 0;
		input[1] = (rnd() >= 0.5) ? 1 : 0;
		expected[0] = (double)((int)input[0] ^ (int)input[1]);
		
		nn_array_network_process(&nn, input, expected, NN_MODE_TRAIN);
		
		if(i % PRINT_GENERTION == 0){
			printf("TRAINING: generation %d\n", i);
			printf("\tinput = %f, %f\n", input[0], input[1]);
			printf("\texpected = %f\n", expected[0]);
			printf("\toutput = %f\n", nn.output_nodes[0].output);
			printf("\terror (RMS) = %f\n", nn.error);
			printf("---------------------------------------\n");
		}
	}
	
	for(i = 0; i < PREDICTION_TRIALS; i++){
		input[0] = (rnd() >= 0.5) ? 1 : 0;
		input[1] = (rnd() >= 0.5) ? 1 : 0;
		expected[0] = (double)((int)input[0] ^ (int)input[1]);
		
		nn_array_network_process(&nn, input, expected, NN_MODE_PREDICT);
		
		printf("PREDICTION: generation %d\n", i);
		printf("\tinput = %f, %f\n", input[0], input[1]);
		printf("\texpected = %f\n", expected[0]);
		printf("\toutput = %f\n", nn.output_nodes[0].output);
		printf("\terror (RMS) = %f\n", nn.error);
		printf("---------------------------------------\n");
	}
	
	printf("freeing array network\n");
	nn_array_network_destroy(&nn);
	
	return 0;
	
error:
	fprintf(stderr, "failed to run array network\n");	
	return ret;
}