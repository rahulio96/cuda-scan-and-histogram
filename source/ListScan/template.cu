#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define BLOCK_SIZE 512 //do not change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *aux, int len) {
    //@@ Modify the body of this kernel to generate the scanned blocks
    //@@ Make sure to use the workefficient version of the parallel scan
    //@@ Also make sure to store the block sum to the aux array 

    // Data Loading
    __shared__ float XY[2 * BLOCK_SIZE];
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
        XY[threadIdx.x] = input[i];
    }

    if (i + blockDim.x < numElements) {
        XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];
    }

    // Reduction
    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride * 2 - 1) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE) {
            XY[index] += XY[index - stride];
        }
    }

    // Post-Reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < 2 * BLOCK_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();
    if (i < numElements) {
        // MIGHT NEED TO CHANGE
        output[i] = XY[threadIdx.x];
    }

    if (i + blockDim.x < numElements) {
        output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
    }
}

__global__ void addScannedBlockSums(float *output, float *aux, int len) {
	//@@ Modify the body of this kernel to add scanned block sums to 
	//@@ all values of the scanned blocks

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output 1D list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAuxArray, *deviceAuxScannedArray;
  int numElements; // number of elements in the input/output list. 
				   
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating device memory.");
  //@@ Allocate device memory
  //you can assume that deviceAuxArray size would not need to be more than BLOCK_SIZE*2 (i.e., 1024)
  int size = numElements * sizeOf(float);
  int auxSize = BLOCK_SIZE * 2 * sizeOf(float);

  cudaMalloc((void**)&deviceInput, size);
  cudaMalloc((void**)&deviceOutput, size);
  cudaMalloc((void**)&deviceAuxArray, auxSize);
  cudaMalloc((void**)&deviceAuxScannedArray, auxSize);

  wbTime_stop(GPU, "Allocating device memory.");

  wbTime_start(GPU, "Clearing output device memory.");
  //@@ zero out the deviceOutput using cudaMemset() by uncommenting the below line
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output device memory.");

  wbTime_start(GPU, "Copying input host memory to device.");
  //@@ Copy input host memory to device	
  cudaMemcpy(deviceInput, hostInput, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutput, hostOutput, size, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input host memory to device.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock((num_elements - 1) / BLOCK_SIZE + 1, 1, 1);
  dim3 dimGrid(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  //@@ You need to launch scan kernel twice: 1) for generating scanned blocks 
  //@@ (hint: pass deviceAuxArray to the aux parameter)
  //@@ and 2) for generating scanned aux array that has the scanned block sums. 
  //@@ (hint: pass NULL to the aux parameter)
  //@@ Then you should call addScannedBlockSums kernel.
  scan << <dimGrid, dimBlock> >> (deviceInput, deviceOutput, deviceAuxArray, numElements);
  scan << <dimGrid, dimBlock> >> (deviceInput, deviceOutput, NULL, numElements);
  addScannedBlockSums << <dimGrid, dimBlock> >> (deviceOutput, deviceAuxScannedArray, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output device memory to host");
  //@@ Copy results from device to host	
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output device memory to host");

  wbTime_start(GPU, "Freeing device memory");
  //@@ Deallocate device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxArray);
  cudaFree(deviceAuxScannedArray);
  wbTime_stop(GPU, "Freeing device memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
