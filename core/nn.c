#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <nn.h>

void nn_array_network_destroy(struct nn_array_network *nn){
	int i;
	
	for(i = 0; i < nn->nr_inputs; i++){
		nn_node_destroy(&nn->input_nodes[i]);
	}
	free(nn->input_nodes);
	
	for(i = 0; i < nn->hidden_npl * nn->hidden_layers; i++){
		nn_node_destroy(&nn->hidden_nodes[i]);
	}
	free(nn->hidden_nodes);
	
	for(i = 0; i < nn->nr_outputs; i++){
		nn_node_destroy(&nn->output_nodes[i]);
	}
	free(nn->output_nodes);
}

int nn_array_network_init(struct nn_array_network *nn, unsigned nr_inputs, unsigned nr_outputs, unsigned npl, unsigned hidden_layers, struct nn_node_ops *ops, nn_calculate_error error_fn, double learning_rate, double momentum, double weight_ul, double weight_ll){
	int ret, i, j, k, id = 0;
	int nr_out_nodes, nr_in_nodes;
	
	printf("allocating input nodes\n");
	nn->input_nodes = calloc(1, nr_inputs * sizeof(struct nn_node));
	if(!nn->input_nodes){
		ret = ENOMEM;
		goto error;
	}
	
	printf("allocating hidden nodes\n");
	nn->hidden_nodes = calloc(1, npl * hidden_layers * sizeof(struct nn_node));
	if(!nn->hidden_nodes){
		ret = ENOMEM;
		goto error;
	}
	
	printf("allocating output nodes\n");
	nn->output_nodes = calloc(1, nr_outputs * sizeof(struct nn_node));
	if(!nn->output_nodes){
		ret = ENOMEM;
		goto error;
	}
	
	printf("initalizing input nodes\n");
	for(i = 0; i < nr_inputs; i++){
		ret = nn_node_init(&nn->input_nodes[i], id++, 0, npl, NULL);
		if(ret) goto error;
	}
	
	printf("initalizing hidden nodes\n");
	for(i = 0; i < hidden_layers; i++){
		for(j = 0; j < npl; j++){
			if(i == 0) nr_in_nodes = nr_inputs + 1;
			else nr_in_nodes = npl + 1;
			
			if(i == hidden_layers - 1) nr_out_nodes = nr_outputs;
			else nr_out_nodes = npl;
			
			ret = nn_node_init(&nn->hidden_nodes[i * npl + j], id++, nr_in_nodes, nr_out_nodes, &nn->node_ops);
			if(ret) goto error;
		}
	}
	
	printf("initalizing output nodes\n");
	for(i = 0; i < nr_outputs; i++){
		ret = nn_node_init(&nn->output_nodes[i], id++, npl + 1, 0, &nn->node_ops);
		if(ret) goto error;
	}
	
	printf("initializing first hidden layer connections\n");
	for(j = 0; j < npl; j++){
		nn_node_init_bias(&nn->hidden_nodes[j], weight_ul, weight_ll);
		for(k = 0; k < nr_inputs; k++){
			nn_node_init_connection(&nn->input_nodes[k], &nn->hidden_nodes[j], weight_ul, weight_ll);
		}
	}
	
	printf("initializing other hidden layer connections\n");
	for(i = 1; i < hidden_layers; i++){
		for(j = 0; j < npl; j++){
			nn_node_init_bias(&nn->hidden_nodes[i * npl + j], weight_ul, weight_ll);
			for(k = 0; k < npl; k++){
				nn_node_init_connection(&nn->hidden_nodes[(i - 1) * npl + k], &nn->hidden_nodes[i * npl + j], weight_ul, weight_ll);
			}
		}
	}
	
	printf("initializing output layer connections\n");
	for(j = 0; j < nr_outputs; j++){
		nn_node_init_bias(&nn->output_nodes[j], weight_ul, weight_ll);
		for(k = 0; k < npl; k++){
			nn_node_init_connection(&nn->hidden_nodes[(hidden_layers - 1) * npl + k], &nn->output_nodes[j], weight_ul, weight_ll);
		}
	}
	
	printf("finishing setting up struct\n");
	nn->nr_inputs = nr_inputs;
	nn->nr_outputs = nr_outputs;
	nn->hidden_layers = hidden_layers;
	nn->hidden_npl = npl;
	nn->error = 0;
	nn->momentum = momentum;
	nn->learning_rate = learning_rate;
	nn->training_cases = 0;
	nn->node_ops = *ops;
	nn->error_fn = error_fn;
	
	return 0;

error:
	fprintf(stderr, "failed to initialize array network\n");
	nn_array_network_destroy(nn);
	
	return ret;
}

void nn_array_network_process(struct nn_array_network *nn, double *values, double *expected, nn_mode_t mode){
	int i;
	
	for(i = 0; i < nn->nr_inputs; i++){
		nn->input_nodes[i].output = values[i];
	}
	
	for(i = 0; i < nn->hidden_layers * nn->hidden_npl; i++){
		nn_node_process(&nn->hidden_nodes[i]);
	}
	
	for(i = 0; i < nn->nr_outputs; i++){
		nn_node_process(&nn->output_nodes[i]);
	}
	
	nn->error = nn->error_fn(nn, expected);
	
	if(mode == NN_MODE_TRAIN){
		for(i = nn->nr_outputs - 1; i >= 0; i--){
			nn->output_nodes[i].gradient = nn->node_ops.output_gradient_fn(&nn->output_nodes[i], expected[i]);
		}
		
		for(i = nn->hidden_npl * nn->hidden_layers - 1; i >= 0; i--){
			nn->hidden_nodes[i].gradient = nn->node_ops.hidden_gradient_fn(&nn->hidden_nodes[i]);
		}
		
		for(i = nn->nr_outputs - 1; i >= 0; i--){
			nn->node_ops.recalculate_weights_fn(&nn->output_nodes[i], nn);
		}
		
		for(i = nn->hidden_npl * nn->hidden_layers - 1; i >= 0; i--){
			nn->node_ops.recalculate_weights_fn(&nn->hidden_nodes[i], nn);
		}
		
		nn->training_cases++;
	}
}
