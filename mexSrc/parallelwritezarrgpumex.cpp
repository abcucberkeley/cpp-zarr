// parallelWriteZarrGPU(file, data, ...): parallelWriteZarr's write of a gpuArray (the Parallel
// Computing Toolbox's GPU arrays), with the same options, for data that is on the GPU anyway: the
// chunks are put together and compressed on the GPU with nvCOMP and their files written here
// (gpuwritezarr.h). Data of another class than the array's data type is converted as
// parallelWriteZarr converts it, in host memory, and written from there (cast it on the GPU, as
// uint16(data), to keep the write on the GPU). Needs an NVIDIA GPU and nvCOMP (see gpureadzarr.h;
// setenv('CPPZARR_NVCOMP_LIBRARY', <libnvcomp file>) before the first write tells it where nvCOMP
// is).
#define CPPZARR_MEX_GPU
#include "parallelwritezarrmex.cpp"
