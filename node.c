#include <stdlib.h>
#include <stdio.h>
#include "nn.h"

void nn_node_printf(struct nn_node *node){
	int i;
	
	if(!node){
		printf("node: NULL\n");
		return;
	}
	
	printf("node: %d\n", node->id);
	printf("\toutput: %f\n", node->output);
	printf("\tgradient: %f\n", node->gradient);
	printf("\tinput_total: %f\n", node->input_total);
	
	printf("\tinputs: %d\n", node->nr_inputs);
	if(node->nr_inputs != 0) printf("\t\tbias: weight = %f, delta_weight = %f\n", node->inputs[0].weight, node->inputs[0].delta_weight);
	for(i = 1; i < node->nr_inputs; i++){
		printf("\t\tid = %d: weight = %f, delta_weight = %f\n", node->inputs[i].node->id, node->inputs[i].weight, node->inputs[i].delta_weight);
	}
	
	printf("\toutputs: %d\n", node->nr_out_nodes);
	for(i = 0; i < node->nr_out_nodes; i++){
		printf("\t\tid = %d\n", node->out_nodes[i].node->id);
	}
	
	printf("\n");
}

void nn_node_free(struct nn_node *node){
	free(node->inputs);
	free(node->out_nodes);
	free(node);
}

int nn_node_alloc(int id, int nr_inputs, int nr_out_nodes, nn_transfer_fn *tfn, nn_transfer_fn *tdfn, struct nn_node **node_out){
	int ret;
	struct nn_node *node = NULL;
	struct nn_input *inputs = NULL;
	struct nn_output *out_nodes = NULL;
	
	node = calloc(1, sizeof(struct nn_node));
	if(!node){
		ret = ENOMEM;
		goto error;
	}
	
	if(nr_inputs != 0){
		inputs = malloc(nr_inputs * sizeof(struct nn_input));
		if(!inputs){
			ret = ENOMEM;
			goto error;
		}
	}
	
	if(nr_out_nodes != 0){
		out_nodes = malloc(nr_out_nodes * sizeof(struct nn_output));
		if(!out_nodes){
			ret = ENOMEM;
			goto error;
		}
	}
	
	node->id = id;
	node->output = 0;
	node->gradient = 0;
	node->input_total = 0;
	node->nr_inputs = 0;
	node->nr_out_nodes = 0;
	node->transfer_fn = tfn;
	node->transfer_derivative_fn = tdfn;
	node->total_inputs = nr_inputs;
	node->total_outputs = nr_out_nodes;
	node->inputs = inputs;
	node->out_nodes = out_nodes;
	
	*node_out = node;
	return 0;
	
error:
	fprintf(stderr, "failed to allocate node\n");
	if(inputs) free(inputs);
	if(out_nodes) free(out_nodes);
	if(node) free(node);
	
	*node_out = NULL;
	return ret;
}

static double generate_weight(double weight_ul, double weight_ll){
	return weight_ll + ((double)rand()) / (((double)(RAND_MAX)) / (weight_ul - weight_ll));
}

void nn_node_init_bias(struct nn_node *node, double weight_ul, double weight_ll){
	double weight = generate_weight(weight_ul, weight_ll);
	
	node->inputs[0].node = NULL;
	node->inputs[0].weight = weight;
	node->inputs[0].delta_weight = 0;
	node->nr_inputs++;
}

void nn_node_init_connection(struct nn_node *in_node, struct nn_node *out_node, double weight_ul, double weight_ll){
	struct nn_input *input = &out_node->inputs[out_node->nr_inputs];
	struct nn_output *output = &in_node->out_nodes[in_node->nr_out_nodes];
	double weight = generate_weight(weight_ul, weight_ll);
	
	input->weight = weight;
	input->delta_weight = 0;
	input->node = in_node;
	out_node->nr_inputs++;
	
	output->node = out_node;
	output->input = input;
	in_node->nr_out_nodes++;
}

void nn_node_process(struct nn_node *node){
	int i;
	
	node->input_total = node->inputs[0].weight;
	for(i = 1; i < node->nr_inputs; i++){
		node->input_total += node->inputs[i].weight * node->inputs[i].node->output;
	}
	
	node->output = node->transfer_fn(node->input_total);
}

void nn_node_calculate_output_gradient(struct nn_node *node, double expected){
	node->gradient = (expected - node->output) * node->transfer_derivative_fn(node->output);
}

void nn_node_calculate_gradient(struct nn_node *node){
	int i;
	double sum = 0;
	
	for(i = 0; i < node->nr_out_nodes; i++){
		sum += node->out_nodes[i].input->weight * node->out_nodes[i].node->gradient;
	}
	node->gradient = sum * node->transfer_derivative_fn(node->output);
}

void nn_node_recalculate_weights(struct nn_node *node, double learning_rate, double momentum){
	int i;
	double input_value;
	
	for(i = 0; i < node->nr_inputs; i++){
		input_value = (i != 0) ? node->inputs[i].node->output : 1;
		node->inputs[i].delta_weight = learning_rate * input_value * node->gradient + momentum * node->inputs[i].delta_weight;
		node->inputs[i].weight += node->inputs[i].delta_weight;
	}
}
