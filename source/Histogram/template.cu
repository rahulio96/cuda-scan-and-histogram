#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define NUM_BINS 4096
#define BLOCK_SIZE 512 

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void histogram(unsigned int *input, unsigned int *bins,
	unsigned int num_elements,
	unsigned int num_bins) {
	//@@ Write the kernel that computes the histogram
	//@@ Make sure to use the privitization technique and each integer will map into a single bin in both shared and global memory
	//(hint: since NUM_BINS=4096 is larger than maximum allowed number of threads per block, 
	//be aware that threads would need to initialize more than one shared memory bin 
	//and update more than one global memory bin)

	__shared__ unsigned int private_histo[NUM_BINS];
	
	// Init to shared memory bins to 0
	for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
		private_histo[i] = 0;
	}
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	// Add to shared memory
	while (i < num_elements) {
		unsigned int bin = input[i];
		atomicAdd(&(private_histo[bin]), 1);
		i += stride;
	}

	__syncthreads();

	// Transfer from shared memory to output bin
	// Update global memory bin
	for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
		atomicAdd(&(bins[i]), private_histo[i]);
	}
}

__global__ void saturate(unsigned int *bins, unsigned int num_bins) {
	//@@ Write the kernel that applies saturtion to counters (i.e., if the bin value is more than 127, make it equal to 127)
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x;

	while (i < num_bins) {
		if (bins[i] > 127) {
			bins[i] = 127;
		}
		i += stride;
	}
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating device memory");
  //@@ Allocate device memory here
  int size = sizeof(unsigned int);
  cudaMalloc((void**)&deviceInput, inputLength * size);
  cudaMalloc((void**)&deviceBins, NUM_BINS * size);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating device memory");

  wbTime_start(GPU, "Copying input host memory to device");
  //@@ Copy input host memory to device
  cudaMemcpy(deviceInput, hostInput, inputLength * size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, NUM_BINS * size, cudaMemcpyHostToDevice);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input host memory to device");
	
  wbTime_start(GPU, "Clearing the bins on device");
  //@@ zero out the deviceBins using cudaMemset() 
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));
  wbTime_stop(GPU, "Clearing the bins on device");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((inputLength - 1) / BLOCK_SIZE + 1, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Invoke kernels: first call histogram kernel and then call saturate kernel
  histogram << <dimGrid, dimBlock >> > (deviceInput, deviceBins, inputLength, NUM_BINS);
  saturate << <dimGrid, dimBlock >> > (deviceBins, NUM_BINS);
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output device memory to host");
  //@@ Copy output device memory to host
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * size, cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output device memory to host");

  wbTime_start(GPU, "Freeing device memory");
  //@@ Free the device memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  wbTime_stop(GPU, "Freeing device memory");

  wbSolution(args, hostBins, NUM_BINS);

  free(hostBins);
  free(hostInput);
  return 0;
}
