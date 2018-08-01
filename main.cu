/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

 //final main

#include <stdio.h>
#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
	Timer timer;
	if (argc != 3) {
	    printf("\nInput files not specified");
	    exit(0);
	}
        char *inputImageFile = argv[1];
        char *labelFile = argv[2];

	// Initialize host variables ----------------------------------------------

	printf("\nSetting up the problem..."); fflush(stdout);
	startTime(&timer);
        
	// Allocate and initialize host variables ----------------------------------------------
	Matrix *conv_weight, conv_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, test_image;
	input(inputImageFile, &conv_weight, &conv_bias, &fc1_weight, &fc1_bias, &fc2_weight, &fc2_bias, &test_image);
	int result;

	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Allocate device variables ----------------------------------------------
	float *dev_conv_bias ,*dev_fc1_weight ,*dev_fc1_bias, *dev_fc2_weight ,*dev_fc2_bias ,*dev_test_image;
	float *dev_out,*mul1out ,*mul2out;


	//INSERT DEVICE ALLOCATION CODE HERE

	printf("Allocating device variables..."); fflush(stdout);
	startTime(&timer);

	cudaMalloc(&dev_conv_bias, sizeof(float)*conv_bias.width*conv_bias.height);
	cudaMalloc(&dev_fc1_weight, sizeof(float)*fc1_weight.width*fc1_weight.height);
	cudaMalloc(&dev_fc1_bias, sizeof(float)*fc1_bias.width*fc1_bias.height);
	cudaMalloc(&dev_fc2_weight, sizeof(float)*fc2_weight.width*fc2_weight.height);
	cudaMalloc(&dev_fc2_bias, sizeof(float)*fc2_bias.width*fc2_bias.height);
	cudaMalloc(&dev_test_image, sizeof(float)*test_image.width*test_image.height);

	cudaMalloc(&dev_out, 8*sizeof(float)*test_image.width*test_image.height);

	cudaMalloc(&mul1out, sizeof(float)*512);
	cudaMalloc(&mul2out, sizeof(float)*10);
	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	// Copy host variables to device ------------------------------------------
	printf("Copying data from host to device..."); fflush(stdout);
	startTime(&timer);

	float convmerged[800];
	for(int i=0; i<8;i++){
	for(int j=0; j<100 ;j++){
		convmerged[i*100+j] = conv_weight[i].elements[j];
	}
	}
	cudaMemcpyToSymbol(conv, convmerged, 800*4);

	cudaMemcpyToSymbol(bias, conv_bias.elements, sizeof(float)*conv_bias.width*conv_bias.height);

	cudaMemcpy(dev_conv_bias, conv_bias.elements, sizeof(float)*conv_bias.width*conv_bias.height,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_fc1_weight, fc1_weight.elements, sizeof(float)*fc1_weight.width*fc1_weight.height,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_fc1_bias, fc1_bias.elements, sizeof(float)*fc1_bias.width*fc1_bias.height,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_fc2_weight, fc2_weight.elements, sizeof(float)*fc2_weight.width*fc2_weight.height,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_fc2_bias, fc2_bias.elements, sizeof(float)*fc2_bias.width*fc2_bias.height,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_test_image, test_image.elements, sizeof(float)*test_image.width*test_image.height,cudaMemcpyHostToDevice);

	cudaMemset(dev_out, 0 , 8*sizeof(float)*test_image.width*test_image.height);

	cudaMemset(mul1out, 0 , sizeof(float)*512);
	cudaMemset(mul2out, 0 , sizeof(float)*10);
	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));

	dim3 dim_block(28,28,1);
	dim3 dim_grid(8,1,1);
	dim3 dimblock1(128,1,1);
	dim3 dimgrid1(49,1,1);
	dim3 dimblock2(128,1,1);
	dim3 dimgrid2(4,1,1);

	float *mul_images;
	cudaMalloc(&mul_images, sizeof(float)*fc1_weight.width*fc1_weight.height);


	// Launch kernel ----------------------------------------------------------
	printf("Launching kernel..."); fflush(stdout);
	startTime(&timer);

	convolution <<<dim_grid,dim_block>>>(dev_test_image, dev_out);
	Multiply0<<<dimgrid1 ,dimblock1>>>(dev_out ,dev_fc1_weight,6272 ,mul1out ,dev_fc1_bias);
	Multiply1<<<10,512>>>(mul1out,dev_fc2_weight,mul2out,dev_fc2_bias);
	cudaError_t cuda_ret;
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));



	// Copy device variables from host ----------------------------------------

	printf("Copying data from device to host..."); fflush(stdout);
	startTime(&timer);
	float A[10];
	cudaMemcpy(A, mul2out, 10*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	stopTime(&timer); 
	printf("%f s\n", elapsedTime(timer));


	int max = A[0];
	int index=0;
	for (int i=0; i<10;i++){
		printf("%f  " ,A[i]);
		if (max<A[i]){
			max=A[i];
			index=i;
		}
	}
	result=index;
	// Verify correctness -----------------------------------------------------
    	verify(result, labelFile);
	printf("\n");

	// Free host and device memory ------------------------------------------------------------

	cudaFree(dev_conv_bias);
	cudaFree(dev_fc1_weight);
	cudaFree(dev_fc1_bias);
	cudaFree(dev_fc2_weight);
	cudaFree(dev_fc2_bias);
	cudaFree(dev_out);

	cudaFree(mul1out);
	cudaFree(mul2out);



	return 0;
}


