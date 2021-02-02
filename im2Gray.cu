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
    int grayOffset = y*numCols + x;
    int rgbOffset = grayOffset*3;
    //unsigned char r = d_in[rgbOffset]; 
    //unsigned char g = d_in[rgbOffset + 1];
    //unsigned char b = d_in[rgbOffset + 2];
    uchar4 r = d_in[rgbOffset]; 
    uchar4 g = d_in[rgbOffset + 1];
    uchar4 b = d_in[rgbOffset + 2];
    float4 val1 = 0.299;
    float4 val2 = 0.587;
    float4 val3 = 0.114;
    d_grey[grayOffset] = val1 * make_float4(r) + val2 * make_float4(g) + val3 * make_float4(b);
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





