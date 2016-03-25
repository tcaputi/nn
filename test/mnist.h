#include <stdint.h>

#define MNIST_IMAGE_SIZE 784
#define MNIST_LABEL_MAGIC 2049
#define MNIST_IMAGE_MAGIC 2051

struct mnist_image {
	uint8_t data[MNIST_IMAGE_SIZE]; //pixels of an image
};

struct mnist_data {
	struct mnist_image *images;
	uint8_t *labels;
	uint32_t nr_images;
};

struct mnist {
	struct mnist_data *test;
	struct mnist_data *train;
};

void mnist_image_print(struct mnist_image *mi);
void mnist_free(struct mnist *mnist);
int load_mnist_data(int lfd, int ifd, struct mnist_data **md_out);
int load_all_mnist_data(char *dir, struct mnist **mnist_out);