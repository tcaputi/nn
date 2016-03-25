#include <nn.h>

double bp_sigmoid(double value);
double bp_sigmoid_derivative(double value);
double bp_calculate_output_gradient(struct nn_node *node, double expected);
double bp_calculate_hidden_gradient(struct nn_node *node);
void bp_recalculate_weights(struct nn_node *node, void *data);
double bp_calculate_error(struct nn_array_network *nn, double *expected);

struct nn_node_ops backprop_node_ops;