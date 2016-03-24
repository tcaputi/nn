#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <byteswap.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <linux/limits.h>
#include "mnist.h"

struct mnist_label_header {
	uint32_t magic;
	uint32_t nr_items;
};

struct mnist_image_header {
	uint32_t magic;
	uint32_t nr_images;
	uint32_t nr_rows;
	uint32_t nr_cols;
};

static void mnist_label_header_bswap(struct mnist_label_header *hdr){
	hdr->magic = __bswap_32(hdr->magic);
	hdr->nr_items = __bswap_32(hdr->nr_items);
}

static void mnist_image_header_bswap(struct mnist_image_header *hdr){
	hdr->magic = __bswap_32(hdr->magic);
	hdr->nr_images = __bswap_32(hdr->nr_images);
	hdr->nr_rows = __bswap_32(hdr->nr_rows);
	hdr->nr_cols = __bswap_32(hdr->nr_cols);
}

void mnist_image_print(struct mnist_image *mi){
	int i, j;
	
	for(i = 0; i < 28; i++){
		for(j = 0; j < 28; j++){
			printf("%02X", (int)mi->data[i * 28 + j]);
		}
		printf("\n");
	}
}

void mnist_free(struct mnist *mnist){
	free(mnist->train->images);
	free(mnist->train->labels);
	free(mnist->test->images);
	free(mnist->test->labels);
	free(mnist->train);
	free(mnist->test);
	free(mnist);
}

int load_mnist_data(int lfd, int ifd, struct mnist_data **md_out){
	int ret;
	struct mnist_label_header lhdr;
	struct mnist_image_header ihdr;
	struct mnist_data *md = NULL;
	struct mnist_image *mi = NULL;
	uint8_t *labels = NULL;
	
	md = malloc(sizeof(struct mnist_data));
	if(!md){
		ret = ENOMEM;
		fprintf(stderr, "failed to allocate mnist data\n");
		goto error;
	}
	
	ret = read(lfd, &lhdr, sizeof(struct mnist_label_header));
	if(ret != sizeof(struct mnist_label_header)){
		ret = EIO;
		fprintf(stderr, "failed to read label header\n");
		goto error;
	}
	
	mnist_label_header_bswap(&lhdr);
	
	if(lhdr.magic != MNIST_LABEL_MAGIC){
		ret = EINVAL;
		fprintf(stderr, "invalid label magic\n");
		goto error;
	}
	
	ret = read(ifd, &ihdr, sizeof(struct mnist_image_header));
	if(ret != sizeof(struct mnist_image_header)){
		ret = EIO;
		fprintf(stderr, "failed to read image header\n");
		goto error;
	}
	
	mnist_image_header_bswap(&ihdr);
	
	if(ihdr.magic != MNIST_IMAGE_MAGIC){
		ret = EINVAL;
		fprintf(stderr, "invalid image magic\n");
		goto error;
	}
	
	if(ihdr.nr_images != lhdr.nr_items){
		ret = EINVAL;
		fprintf(stderr, "image count != label count\n");
		goto error;
	}
	
	mi = malloc(ihdr.nr_images * sizeof(struct mnist_image));
	if(!mi){
		ret = ENOMEM;
		fprintf(stderr, "failed to allocate mnist image array\n");
		goto error;
	}
	
	labels = malloc(ihdr.nr_images * sizeof(uint8_t));
	if(!labels){
		ret = ENOMEM;
		fprintf(stderr, "failed to allocate mnist label array\n");
		goto error;
	}
	
	ret = read(lfd, labels, lhdr.nr_items * sizeof(uint8_t));
	if(ret != lhdr.nr_items * sizeof(uint8_t)){
		ret = EIO;
		fprintf(stderr, "failed to read label array\n");
		goto error;
	}
	
	ret = read(ifd, mi, ihdr.nr_images * sizeof(struct mnist_image));
	if(ret != ihdr.nr_images * sizeof(struct mnist_image)){
		ret = EIO;
		fprintf(stderr, "failed to read image array\n");
		goto error;
	}
	
	md->images = mi;
	md->labels = labels;
	md->nr_images = ihdr.nr_images;
	
	*md_out = md;
	return 0;
	
error:
	fprintf(stderr, "failed to load mnist data\n");
	if(mi) free(mi);
	if(labels) free(labels);
	if(md) free(md);
	
	*md_out = NULL;
	return ret;
}

int load_all_mnist_data(char *dir, struct mnist **mnist_out){
	int ret, trainl_fd = 0, traini_fd = 0, testl_fd = 0, testi_fd = 0;
	struct mnist *mnist = NULL;
	struct mnist_data *train = NULL, *test = NULL;
	char path[PATH_MAX];
	
	mnist = malloc(sizeof(struct mnist));
	if(!mnist){
		ret = ENOMEM;
		fprintf(stderr, "failed to allocate mnist structure\n");
		goto error;
	}
	
	strncpy(path, dir, sizeof(path));
	strncat(path, "/train-labels-idx1-ubyte", sizeof(path));
	
	trainl_fd = open(path, O_RDONLY, 0);
	if(trainl_fd < 0){
		ret = errno;
		errno = 0;
		fprintf(stderr, "failed to open training labels\n");
		goto error;
	}
	
	strncpy(path, dir, sizeof(path));
	strncat(path, "/train-images-idx3-ubyte", sizeof(path));
	
	traini_fd = open(path, O_RDONLY, 0);
	if(traini_fd < 0){
		ret = errno;
		errno = 0;
		fprintf(stderr, "failed to open training images\n");
		goto error;
	}
	
	ret = load_mnist_data(trainl_fd, traini_fd, &train);
	if(ret) goto error;
	
	strncpy(path, dir, sizeof(path));
	strncat(path, "/t10k-labels-idx1-ubyte", sizeof(path));
	
	testl_fd = open(path, O_RDONLY, 0);
	if(testl_fd < 0){
		ret = errno;
		errno = 0;
		fprintf(stderr, "failed to open test labels\n");
		goto error;
	}
	
	strncpy(path, dir, sizeof(path));
	strncat(path, "/t10k-images-idx3-ubyte", sizeof(path));
	
	testi_fd = open(path, O_RDONLY, 0);
	if(testi_fd < 0){
		ret = errno;
		errno = 0;
		fprintf(stderr, "failed to open test images\n");
		goto error;
	}
	
	ret = load_mnist_data(testl_fd, testi_fd, &test);
	if(ret) goto error;
	
	mnist->train = train;
	mnist->test = test;
	
	close(trainl_fd);
	close(traini_fd);
	close(testl_fd);
	close(testi_fd);
	
	*mnist_out = mnist;
	return 0;
	
error:
	fprintf(stderr, "failed to load mnist data from files\n");
	if(trainl_fd) close(trainl_fd);
	if(traini_fd) close(traini_fd);
	if(testl_fd) close(testl_fd);
	if(testi_fd) close(testi_fd);
	if(train){
		free(train->images);
		free(train->labels);
	}
	if(test){
		free(test->images);
		free(test->labels);
	}
	if(mnist) free(mnist);
	
	*mnist_out = NULL;
	return ret;
}