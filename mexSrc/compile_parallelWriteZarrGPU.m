% parallelWriteZarrGPU needs the Parallel Computing Toolbox (its GPU MEX API: gpu/mxGPUArray.h and
% the mwgpu library) and no CUDA toolkit: cpp-zarr loads the CUDA driver and nvCOMP when it first
% writes on the GPU. Linux only for now.
gpuInclude = ['-I' fullfile(matlabroot, 'toolbox', 'parallel', 'gpu', 'extern', 'include')];
if isunix && ~ismac
    if ~exist('cppZarrInstall', 'var')
        cppZarrInstall = '/clusterfs/fiona/matthewmueller/cppZarrTest/c-zarr/jenkinsBuild/install';
    end
    releaseFolder = '../linux';
    if ~exist(releaseFolder, 'dir')
        mkdir(releaseFolder);
    end
    mex('-outdir', releaseFolder, '-output', 'parallelWriteZarrGPU.mexa64', '-v', 'CXXOPTIMFLAGS=-DNDEBUG -O2', ...
        'LDOPTIMFLAGS=-Wl,-rpath,''$ORIGIN'' -O2 -DNDEBUG', 'CXXFLAGS=$CXXFLAGS -O2', ['-I' cppZarrInstall '/include'], ...
        gpuInclude, ['-L' cppZarrInstall '/lib64'], '-lcppZarr', '-lmwgpu', 'parallelwritezarrgpumex.cpp');
    % (as for parallelReadZarr: MATLAB preloads its own libstdc++)
    system(['patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.32 ' releaseFolder '/parallelWriteZarrGPU.mexa64']);
else
    % (Windows: not built yet, untested; macOS: MATLAB has no GPU arrays there)
    error('parallelWriteZarrGPU is built for Linux only');
end
