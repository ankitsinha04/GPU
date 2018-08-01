/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
 
//final kernel

//INSERT KERNEL CODE HERE
//__device__ __constant__ float test_image[1024];

__constant__ float conv[800];
__constant__ float bias[8];


__global__ void convolution(float* test_image ,float* out )
{
	//int Row = blockIdx.y* blockDim.y + threadIdx.y; 
	//int Col = blockIdx.x* blockDim.x + threadIdx.x;

	int convid = blockIdx.x;


	int row = threadIdx.y;
	int col = threadIdx.x;
	int index = 8*(row*28+col);
	


	//loading the matrix into shared memory

	__shared__ float Matrix1[28][28];

	Matrix1[row][col]=test_image[row*28 + col];

	__syncthreads();

	float value = 0.0f;

	for (int i=-4 ;i<=5;i++){
		for(int j=-4 ;j<=5 ;j++){
			int ModRow = row +i;
			int ModCol = col +j;
			if(ModRow>=0 && ModCol>=0 && ModRow<28 && ModCol<28){
				int temp = (i+4)*10 + j+4;
				value += Matrix1[ModRow][ModCol]*conv[temp +100*convid];
			}
		}
	}

	out[index+convid] = (value+bias[convid])>=0? (value +bias[convid]) :0.0f;

}


__global__ void Multiply0(float *imageData , float* multiplier , int multiplier_height,float* matrixresult ,float* bias1){

	__shared__ float ds_M[128];

 	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	double Pvalue = 0.0;
	for (int m = 0; m < 49; m++) {
		ds_M[threadIdx.x] = imageData[128*m + threadIdx.x];
  		__syncthreads();
   		for (int k = 0; k < 128; k++)
			Pvalue += ds_M[k] * multiplier[(m*128+k)*512 +Col];
	}	

   	matrixresult[Col] = (Pvalue +bias1[Col] )>=0?(Pvalue +bias1[Col] ):0;
}

__global__ void Multiply1(float *imageData , float* multiplier , float* matrixresult , float* bias){
	
	__shared__ float ds_M[512][10];
	int col = blockIdx.x;
	int row = threadIdx.x;

	ds_M[row][col] = multiplier[row*10+col]* imageData[row];

	__syncthreads();


	if (threadIdx.x==0){
		float value=0.0;
			for (int i=0 ; i<512 ; i++)
				value+=ds_M[i][blockIdx.x];
		

		matrixresult[col]= value+ bias[col];
	}

		

	
	}



__global__ void Mul0(float *imageData , float* multiplier ,float* matrixresult){
		
	int row = 7*threadIdx.x;
	int col = 2*blockIdx.x;

	for(int i=0; i<7 ;i++){
		if((row+i)<6272){
		matrixresult[(row+i)*512+col] = multiplier[(row+i)*512+col]*imageData[col];
		matrixresult[(row+i)*512+col+1] = multiplier[(row+i)*512+col+1]*imageData[col+1];
		}
	}
	}

