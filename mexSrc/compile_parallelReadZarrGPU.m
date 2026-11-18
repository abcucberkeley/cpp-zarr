% parallelReadZarrGPU needs the Parallel Computing Toolbox (its GPU MEX API: gpu/mxGPUArray.h and
% the mwgpu library) and no CUDA toolkit: cpp-zarr loads the CUDA driver and nvCOMP when it first
% reads on the GPU. Linux only for now.
gpuInclude = ['-I' fullfile(matlabroot, 'toolbox', 'parallel', 'gpu', 'extern', 'include')];
if isunix && ~ismac
    if ~exist('cppZarrInstall', 'var')
        cppZarrInstall = '../jenkinsBuild/install';
    end
    releaseFolder = '../linux';
    if ~exist(releaseFolder, 'dir')
        mkdir(releaseFolder);
    end
    % (as for parallelReadZarr: libstdc++ linked in statically, MATLAB's LINKLIBS without -lstdc++ and
    % -lMatlabDataArray, with the libraries of this MEX file, which LINKLIBS replaces, and no debug information)
    linkLibs = ['LINKLIBS=-L' cppZarrInstall '/lib64 -lcppZarr -lmwgpu -Wl,--as-needed -Wl,-rpath-link,$MATLABROOT/bin/$ARCH ' ...
                '-L"$MATLABROOT/bin/$ARCH" -lmx -lmex -lm -lmat'];
    mex('-outdir', releaseFolder, '-output', 'parallelReadZarrGPU.mexa64', '-v', 'CXXOPTIMFLAGS=-DNDEBUG -O2', ...
        'LDOPTIMFLAGS=-Wl,-rpath,''$ORIGIN'' -O2 -DNDEBUG', 'CXXFLAGS=$CXXFLAGS -O2', 'LDFLAGS=$LDFLAGS -static-libstdc++ -Wl,-S', linkLibs, ['-I' cppZarrInstall '/include'], ...
        gpuInclude, 'parallelreadzarrgpumex.cpp');
    system(['patchelf --force-rpath --set-rpath ''$ORIGIN'' ' releaseFolder '/parallelReadZarrGPU.mexa64']);
else
    % (Windows: not built yet, untested; macOS: MATLAB has no GPU arrays there)
    error('parallelReadZarrGPU is built for Linux only');
end
