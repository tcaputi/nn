#include <stdlib.h>
#include <stdio.h>
#include "nn.h"

void nn_array_network_free(struct nn_array_network *nn){
	int i;
	
	for(i = 0; i < nn->nodes_per_layer * nn->layers; i++){
		nn_node_free(nn->nodes[i]);
	}
	
	nn_node_free(nn->out_node);
	free(nn);
}

int nn_array_network_alloc(int npl, int layers, nn_transfer_fn *tfn, nn_transfer_fn *tdfn, double learning_rate, double momentum, double weight_ul, double weight_ll, struct nn_array_network **nn_out){
	int ret, i, j, k;
	struct nn_array_network *nn = NULL;
	struct nn_node **nodes = NULL;
	struct nn_node *out_node = NULL;
	
	printf("allocating array network struct\n");
	nn = malloc(sizeof(struct nn_array_network));
	if(!nn){
		ret = ENOMEM;
		goto error;
	}
	
	printf("allocating nodes array\n");
	nodes = calloc(1, npl * layers * sizeof(struct nn_node *));
	if(!nodes){
		ret = ENOMEM;
		goto error;
	}
	
	//input nodes have no inputs annd no tranfer function
	printf("allocating input nodes\n");
	for(i = 0; i < npl; i++){
		ret = nn_node_alloc(i, 0, npl, NULL, NULL, &nodes[i]);
		if(ret) goto error;
	}
	
	//allocate all hidden nodes
	printf("allocating hidden nodes\n");
	for(i = npl; i < npl * layers; i++){
		ret = nn_node_alloc(i, npl + 1, npl, tfn, tdfn, &nodes[i]);
		if(ret) goto error;
	}
	
	//allocate output node
	printf("allocating output node\n");
	ret = nn_node_alloc(1000000, npl + 1, 0, tfn, tdfn, &out_node);
	if(ret) goto error;
	
	//initialize all inputs to hidden nodes (input nodes don't have any)
	printf("initializing inputs\n");
	for(i = 1; i < layers; i++){
		for(j = 0; j < npl; j++){
			nn_node_init_bias(nodes[(i * npl) + j], weight_ul, weight_ll);
			for(k = 0; k < npl; k++){
				nn_node_init_connection(nodes[((i - 1) * npl) + k], nodes[(i * npl) + j], weight_ul, weight_ll);
			}
		}
	}
	
	nn_node_init_bias(out_node, weight_ul, weight_ll);
	for(k = 0; k < npl; k++){
		nn_node_init_connection(nodes[((layers - 1) * npl) + k], out_node, weight_ul, weight_ll);
	}
	
	for(i = 0; i < npl * layers; i++){
		nn_node_printf(nodes[i]);
	}
	nn_node_printf(out_node);
	
	printf("finishing setting up struct\n");
	nn->nodes_per_layer = npl;
	nn->layers = layers;
	nn->momentum = momentum;
	nn->learning_rate = learning_rate;
	nn->training_cases = 0;
	nn->nodes = nodes;
	nn->out_node = out_node;
	
	*nn_out = nn;
	return 0;

error:
	fprintf(stderr, "failed to allocate array network\n");
	if(nn) free(nn);
	
	for(i = 0; i < npl * layers; i++){
		if(nodes[i]) nn_node_free(nodes[i]);
	}
	
	if(nodes) free(nodes);
	if(out_node) nn_node_free(out_node);
	
	*nn_out = NULL;
	return ret;
}

void nn_array_network_process(struct nn_array_network *nn, double *values, double expected, nn_mode_t mode){
	int i;
	
	//printf("initializing inputs\n");
	for(i = 0; i < nn->nodes_per_layer; i++){
		nn->nodes[i]->output = values[i];
	}
	
	//printf("processing hidden nodes\n");
	for(i = nn->nodes_per_layer; i < nn->nodes_per_layer * nn->layers; i++){
		nn_node_process(nn->nodes[i]);
	}
	
	//printf("initializing output node\n");
	nn_node_process(nn->out_node);
	
	if(mode == NN_MODE_TRAIN){
		//printf("back-propogating error\n");
		nn_node_calculate_output_ndelta(nn->out_node, expected);
		nn_node_recalculate_weights(nn->out_node, nn->learning_rate, nn->momentum);
		
		for(i = nn->nodes_per_layer * nn->layers - 1; i >= nn->nodes_per_layer; i--){
			//printf("\tcalculating error error\n");
			nn_node_calculate_ndelta(nn->nodes[i]);
			
			//printf("\trecalculating weights\n");
			nn_node_recalculate_weights(nn->nodes[i], nn->learning_rate, nn->momentum);
		}
		nn->training_cases++;
	}
}
