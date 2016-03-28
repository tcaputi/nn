#ifndef NN_NN_H
#define NN_NN_H

#include <errno.h>
#include <math.h>

struct nn_node;

//node operations
typedef double (nn_transfer_fn) (double value);
typedef double (nn_output_gradient_fn) (struct nn_node *node, double expected);
typedef double (nn_hidden_gradient_fn) (struct nn_node *node);
typedef void (nn_recalculate_weights_fn) (struct nn_node *node, void *data);

struct nn_node_ops {
	nn_transfer_fn *transfer_fn;
	nn_transfer_fn *transfer_derivative_fn;
	nn_output_gradient_fn *output_gradient_fn;
	nn_hidden_gradient_fn *hidden_gradient_fn;
	nn_recalculate_weights_fn *recalculate_weights_fn;
};

struct nn_input {
	double weight;
	double delta_weight;
	struct nn_node *node;
};

struct nn_output {
	struct nn_node *node;
	struct nn_input *input;
};

struct nn_node {
	int id;
	double value;
	double gradient;
	double input_total;
	int total_inputs;
	int total_outputs;
	int nr_inputs; //no inputs -> input node
	int nr_outputs; //no outputs -> output node
	struct nn_node_ops *ops;
	struct nn_input *inputs;
	struct nn_output *outputs;
};

struct nn_array_network;
typedef double (nn_calculate_error) (struct nn_array_network *nn, double *expected);

struct nn_array_network {
	unsigned nr_inputs;
	unsigned nr_outputs;
	unsigned hidden_layers;
	unsigned hidden_npl;
	double error;
	double momentum;
	double learning_rate;
	long training_cases;
	struct nn_node_ops node_ops;
	nn_calculate_error *error_fn;
	struct nn_node *input_nodes;
	struct nn_node *hidden_nodes;
	struct nn_node *output_nodes;
};

typedef enum nn_mode {
	NN_MODE_TRAIN,
	NN_MODE_PREDICT,
} nn_mode_t;

//node functions
void nn_node_printf(struct nn_node *node);
void nn_node_destroy(struct nn_node *node);
int nn_node_init(struct nn_node *node, int id, int nr_inputs, int nr_out_nodes, struct nn_node_ops *ops);
void nn_node_init_bias(struct nn_node *node, double weight_ul, double weight_ll);
void nn_node_init_connection(struct nn_node *in_node, struct nn_node *out_node, double weight_ul, double weight_ll);
void nn_node_process(struct nn_node *node);

//array network functions
void nn_array_network_destroy(struct nn_array_network *nn);
int nn_array_network_init(struct nn_array_network *nn, unsigned nr_inputs, unsigned nr_outputs, unsigned npl, unsigned hidden_layers, struct nn_node_ops *ops, nn_calculate_error error_fn, double learning_rate, double momentum, double weight_ul, double weight_ll);
void nn_array_network_calculate_error(struct nn_array_network *nn, double *expected);
void nn_array_network_process(struct nn_array_network *nn, double *values, double *expected, nn_mode_t mode);

#endif