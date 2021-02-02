#include "im2Gray.h"

#define BLOCK 256



/*
 
  Given an input image d_in, perform the grayscale operation 
  using the luminance formula i.e. 
  o[i] = 0.224f*r + 0.587f*g + 0.111*b; 
  
  Your kernel needs to check for boundary conditions 
  and write the output pixels in gray scale format. 

  you may vary the BLOCK parameter.
 
 */
__global__ 
void im2Gray(uchar4 *d_in, unsigned char *d_grey, int numRows, int numCols){

 /*
   Your kernel here: Make sure to check for boundary conditions
  */

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < numCols && y < numRows){
    int grayOffset= y*numCols + x;
    int rgbOffset= grayOffset*3;
    //unsigned char r = d_in[rgbOffset]; 
    //unsigned char g = d_in[rgbOffset + 1];
    //unsigned char b = d_in[rgbOffset + 2];
    char r = d_in[rgbOffset]; 
    char g = d_in[rgbOffset + 1];
    char b = d_in[rgbOffset + 2];
    d_grey[grayOffset] = 0.299f*r + 0.587f*g + 0.114f*b;
  } 
}




void launch_im2gray(uchar4 *d_in, unsigned char* d_grey, size_t numRows, size_t numCols){
    // configure launch params here 
    
    dim3 block(BLOCK,1,1);
    dim3 grid((numRows*numCols + 1)/BLOCK,1,1);

    im2Gray<<<grid,block>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
}





