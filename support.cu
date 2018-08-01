/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "support.h"

Matrix allocateMatrix(unsigned height, unsigned width)
{
	Matrix mat;
	mat.height = height;
	mat.width = mat.pitch = width;
	mat.elements = (float*)malloc(height*width*sizeof(float));
	if(mat.elements == NULL) FATAL("Unable to allocate host");

	return mat;
}

void input(char *inputImageFile, Matrix **conv_weight, Matrix *conv_bias, Matrix *fc1_weight, Matrix *fc1_bias, Matrix *fc2_weight, Matrix *fc2_bias, Matrix *test_image) {

    FILE *fp;
    int height, width, depth;

    // weight for convolution layer
    fp=fopen("conv_w.txt", "r");
    fscanf(fp, "%d %d %d", &height, &width, &depth);
    // allocate matrix
    *conv_weight = (Matrix*)malloc(depth*sizeof(Matrix));
    for (unsigned int i=0; i < depth; i++) {
        (*conv_weight)[i] = allocateMatrix(height, width); 
        for (unsigned int j=0; j < height*width; j++) {
            fscanf(fp, "%f", &((*conv_weight)[i].elements[j]));
        }
    }
    fclose(fp);

    // bias for convolution layer
    fp=fopen("conv_b.txt", "r");
    fscanf(fp, "%d %d", &height, &width);
    // allocate matrix
    *conv_bias = allocateMatrix(height, width); 
    for (unsigned int i=0; i < height*width; i++) {
        fscanf(fp, "%f", &(conv_bias->elements[i]));
    }
    fclose(fp);

    // weight for first fully connected layer
    fp=fopen("fc1_w.txt", "r");
    fscanf(fp, "%d %d", &height, &width);
    // allocate matrix
    *fc1_weight = allocateMatrix(height, width); 
    for (unsigned int i=0; i < height*width; i++) {
        fscanf(fp, "%f", &(fc1_weight->elements[i]));
    }
    fclose(fp);

    // bias for first fully connected layer
    fp=fopen("fc1_b.txt", "r");
    fscanf(fp, "%d %d", &height, &width);
    // allocate matrix
    *fc1_bias = allocateMatrix(height, width); 
    for (unsigned int i=0; i < height*width; i++) {
        fscanf(fp, "%f", &(fc1_bias->elements[i]));
    }
    fclose(fp);

    // weight for second fully connected layer
    fp=fopen("fc2_w.txt", "r");
    fscanf(fp, "%d %d", &height, &width);
    // allocate matrix
    *fc2_weight = allocateMatrix(height, width); 
    for (unsigned int i=0; i < height*width; i++) {
        fscanf(fp, "%f", &(fc2_weight->elements[i]));
    }
    fclose(fp);

    // bias for second fully connected layer
    fp=fopen("fc2_b.txt", "r");
    fscanf(fp, "%d %d", &height, &width);
    // allocate matrix
    *fc2_bias = allocateMatrix(height, width); 
    for (unsigned int i=0; i < height*width; i++) {
        fscanf(fp, "%f", &(fc2_bias->elements[i]));
    }
    fclose(fp);

    // test image
    fp=fopen(inputImageFile, "r");
    // allocate matrix
    height = width = IMAGE_SIZE;
    *test_image = allocateMatrix(height, width); 
    for (unsigned int i=0; i < height*width; i++) {
        fscanf(fp, "%f", &(test_image->elements[i]));
    }
    fclose(fp);
}

void output(char *outputFileName, Matrix Labels) {
    FILE *fp;
    fp=fopen(outputFileName, "wb");
    fprintf(fp, "%d %d", Labels.height, Labels.width);
    for (unsigned int i=0; i < Labels.height * Labels.width; i++) {
        fprintf(fp, "\n%f", Labels.elements[i]);
    }
    fclose(fp);
}

void verify(int result, char *labelFile) {

    FILE *fp;
    if (result < 10) {
        fp=fopen(labelFile, "r");
	int val;
        for (unsigned int i=0; i <= result; i++) {
            fscanf(fp, "%d", &val);
        }
        fclose(fp);
        if (val == 1) {
            printf("\n\nPASS!");
	    return;
        }
    }
    printf("\n\nFAIL!");
}

void freeMatrix(Matrix mat)
{
	free(mat.elements);
	mat.elements = NULL;
}

void freeDeviceMatrix(Matrix mat)
{
	cudaFree(mat.elements);
	mat.elements = NULL;
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}


