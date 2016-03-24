#ifndef NN_NN_H
#define NN_NN_H

#include <errno.h>
#include <math.h>

struct nn_node;

typedef double (nn_transfer_fn) (double value);

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
	double output;
	double gradient;
	double input_total;
	nn_transfer_fn *transfer_fn;
	nn_transfer_fn *transfer_derivative_fn;
	int total_inputs;
	int total_outputs;
	int nr_inputs; //no inputs -> input node
	int nr_out_nodes; //no out_nodes -> output node
	struct nn_input *inputs;
	struct nn_output *out_nodes;
};

struct nn_array_network {
	int nodes_per_layer;
	int layers;
	double error;
	double momentum;
	double learning_rate;
	long training_cases;
	struct nn_node **nodes;
	struct nn_node *out_node;
};

typedef enum nn_mode {
	NN_MODE_TRAIN,
	NN_MODE_PREDICT,
} nn_mode_t;

//node functions
void nn_node_printf(struct nn_node *node);
void nn_node_free(struct nn_node *node);
int nn_node_alloc(int id, int nr_inputs, int nr_out_nodes, nn_transfer_fn *tfn, nn_transfer_fn *tdfn, struct nn_node **node_out);
void nn_node_init_bias(struct nn_node *node, double weight_ul, double weight_ll);
void nn_node_init_connection(struct nn_node *in_node, struct nn_node *out_node, double weight_ul, double weight_ll);
void nn_node_process(struct nn_node *node);
void nn_node_calculate_output_gradient(struct nn_node *node, double expected);
void nn_node_calculate_gradient(struct nn_node *node);
void nn_node_recalculate_weights(struct nn_node *node, double learning_rate, double momentum);

//array network functions
void nn_array_network_free(struct nn_array_network *nn);
int nn_array_network_alloc(int npl, int layers, nn_transfer_fn *tfn, nn_transfer_fn *tdfn, double learning_rate, double momentum, double weight_ul, double weight_ll, struct nn_array_network **nn_out);
void nn_array_network_process(struct nn_array_network *nn, double *values, double expected, nn_mode_t mode);

#endif