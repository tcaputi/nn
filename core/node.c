#include <stdlib.h>
#include <stdio.h>

#include <nn.h>

void nn_node_printf(struct nn_node *node){
	int i;
	
	if(!node){
		printf("node: NULL\n");
		return;
	}
		
	printf("node: %d\n", node->id);
	printf("\tvalue: %f\n", node->value);
	printf("\tgradient: %f\n", node->gradient);
	printf("\tinput_total: %f\n", node->input_total);
	
	printf("\tinputs: %d of %d\n", node->nr_inputs, node->total_inputs);
	if(node->nr_inputs != 0) printf("\t\tbias: weight = %f, delta_weight = %f\n", node->inputs[0].weight, node->inputs[0].delta_weight);
	for(i = 1; i < node->nr_inputs; i++){
		printf("\t\tid = %d: weight = %f, delta_weight = %f\n", node->inputs[i].node->id, node->inputs[i].weight, node->inputs[i].delta_weight);
	}
	
	printf("\toutputs: %d of %d\n", node->nr_outputs, node->total_outputs);
	for(i = 0; i < node->nr_outputs; i++){
		printf("\t\tid = %d\n", node->outputs[i].node->id);
	}
	
	printf("\n");
}

void nn_node_destroy(struct nn_node *node){
	if(node->inputs) free(node->inputs);
	if(node->outputs) free(node->outputs);
}

int nn_node_init(struct nn_node *node, int id, int nr_inputs, int nr_outputs, struct nn_node_ops *ops){
	int ret;
	struct nn_input *inputs = NULL;
	struct nn_output *outputs = NULL;
	
	if(nr_inputs != 0){
		inputs = malloc(nr_inputs * sizeof(struct nn_input));
		if(!inputs){
			ret = ENOMEM;
			goto error;
		}
	}
	
	if(nr_outputs != 0){
		outputs = malloc(nr_outputs * sizeof(struct nn_output));
		if(!outputs){
			ret = ENOMEM;
			goto error;
		}
	}
	
	node->id = id;
	node->value = 0;
	node->gradient = 0;
	node->input_total = 0;
	node->nr_inputs = 0;
	node->nr_outputs = 0;
	node->total_inputs = nr_inputs;
	node->total_outputs = nr_outputs;
	node->ops = ops;
	node->inputs = inputs;
	node->outputs = outputs;
	
	return 0;
	
error:
	fprintf(stderr, "failed to allocate node\n");
	if(inputs) free(inputs);
	if(outputs) free(outputs);
	
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
	struct nn_output *output = &in_node->outputs[in_node->nr_outputs];
	double weight = generate_weight(weight_ul, weight_ll);
	
	input->weight = weight;
	input->delta_weight = 0;
	input->node = in_node;
	out_node->nr_inputs++;
	
	output->node = out_node;
	output->input = input;
	in_node->nr_outputs++;
}

void nn_node_process(struct nn_node *node){
	int i;
	
	node->input_total = node->inputs[0].weight;
	for(i = 1; i < node->nr_inputs; i++){
		node->input_total += node->inputs[i].weight * node->inputs[i].node->value;
	}
	
	node->value = node->ops->transfer_fn(node->input_total);
}
