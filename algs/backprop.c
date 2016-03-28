#include <backprop.h>

struct nn_node_ops backprop_node_ops = {
	.transfer_fn = bp_sigmoid,
	.transfer_derivative_fn = bp_sigmoid_derivative,
	.output_gradient_fn = bp_calculate_output_gradient,
	.hidden_gradient_fn = bp_calculate_hidden_gradient,
	.recalculate_weights_fn = bp_recalculate_weights,
};

double bp_sigmoid(double value){
	return tanh(value);
}

double bp_sigmoid_derivative(double value){
	return 1.0 - value * value;
}

double bp_calculate_output_gradient(struct nn_node *node, double expected){
	return (expected - node->value) * node->ops->transfer_derivative_fn(node->value);
}

double bp_calculate_hidden_gradient(struct nn_node *node){
	int i;
	double sum = 0;
	
	for(i = 0; i < node->nr_outputs; i++){
		sum += node->outputs[i].input->weight * node->outputs[i].node->gradient;
	}
	return sum * node->ops->transfer_derivative_fn(node->value);
}

void bp_recalculate_weights(struct nn_node *node, void *data){
	int i;
	struct nn_array_network *nn = data;
	double input_value;
	
	for(i = 0; i < node->nr_inputs; i++){
		input_value = (i != 0) ? node->inputs[i].node->value : 1;
		node->inputs[i].delta_weight = nn->learning_rate * input_value * node->gradient + nn->momentum * node->inputs[i].delta_weight;
		node->inputs[i].weight += node->inputs[i].delta_weight;
	}
}
