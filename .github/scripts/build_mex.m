function build_mex(installDir, outDir)
%BUILD_MEX Compile the cpp-zarr mex files for the GitHub Actions tests.
%   BUILD_MEX(installDir, outDir) builds parallelReadZarr, parallelWriteZarr,
%   createZarrFile and convertZarrToV3 into outDir, linked against the cpp-zarr
%   library that CMake installed into installDir. The compile_*.m scripts in mexSrc build the
%   release binaries from each Jenkins agent's own paths, so CI uses this.
    inc = fullfile(installDir, 'include');
    lib = fullfile(installDir, 'lib');
    src = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'mexSrc');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    mexes = {'parallelreadzarrmex.cpp',  'parallelReadZarr'
             'parallelwritezarrmex.cpp', 'parallelWriteZarr'
             'createzarrfilemex.cpp',    'createZarrFile'
             'convertzarrtov3mex.cpp',   'convertZarrToV3'};
    % Same optimization and OpenMP flags as the release compile scripts
    % (parallelWriteZarr has an OpenMP loop of its own)
    args = {'-outdir', outDir, ['-I' inc], ['-L' lib], 'CXXOPTIMFLAGS=-O2 -DNDEBUG'};

    if ispc
        % Use the MinGW that built the library. Select it explicitly, since
        % MATLAB would otherwise prefer Visual Studio when both are installed.
        mingw = getenv('MW_MINGW64_LOC');
        gxx = fullfile(mingw, 'bin', 'g++.exe');
        xml = dir(fullfile(matlabroot, 'bin', 'win64', 'mexopts', 'mingw64_g++.xml'));
        assert(~isempty(xml), 'build_mex:noMinGW', 'MATLAB''s MinGW C++ config was not found');
        mex(['-setup:' fullfile(xml(1).folder, xml(1).name)], 'C++');
        [status, gccEh] = system(['"' gxx '" -print-file-name=libgcc_eh.a']);
        assert(status == 0, 'build_mex:noGcc', 'Could not run %s', gxx);
        % -lcppZarr.dll makes the linker pick up the import library libcppZarr.dll.a.
        args = [args, {['CXX=' gxx], 'CXXFLAGS=$CXXFLAGS -fopenmp', 'LDFLAGS=$LDFLAGS -fopenmp', ...
                       strtrim(gccEh), '-lcppZarr.dll'}];
        % The mex files load these at run time. MATLAB looks next to the mex first.
        copyfile(fullfile(installDir, 'bin', 'libcppZarr.dll'), outDir);
        runtime = {'libgomp-1.dll', 'libwinpthread-1.dll', 'libstdc++-6.dll', 'libgcc_s_seh-1.dll'};
        for k = 1:numel(runtime)
            copyfile(fullfile(mingw, 'bin', runtime{k}), outDir);
        end
    elseif ismac
        % The library is built with Homebrew GCC, and the mex files pass
        % std::string and std::vector across the library boundary, so they need
        % GCC's libstdc++ as well rather than Apple clang's libc++. MATLAB's
        % default flags are clang-only (-stdlib=libc++, -weak-lmx), so replace
        % them. mex adds -l libraries to LINKLIBS, so cpp-zarr goes in the override.
        gxx = getenv('CXX');
        if isempty(gxx)
            gxx = '/opt/homebrew/bin/g++-13';
        end
        [status, sdk] = system('xcrun --show-sdk-path');
        assert(status == 0, 'build_mex:noSdk', 'xcrun could not find the macOS SDK');
        sdk = strtrim(sdk);
        args = [args, {['CXX=' gxx], ...
                       ['CXXFLAGS=-fno-common -fexceptions -fopenmp -std=c++11 -isysroot ' sdk], ...
                       ['LDFLAGS=-bundle -Wl,-twolevel_namespace -fopenmp -isysroot ' sdk ' -Wl,-rpath,' lib], ...
                       ['LINKLIBS=-L"' fullfile(matlabroot, 'bin', 'maca64') '" -lmx -lmex -lmat -L"' lib '" -lcppZarr']}];
    else
        % The rpath lets the mex files find libcppZarr at run time.
        args = [args, {'CXXFLAGS=$CXXFLAGS -fopenmp', ...
                       ['LDFLAGS=$LDFLAGS -fopenmp -Wl,-rpath,' lib], '-lcppZarr'}];
    end

    for i = 1:size(mexes, 1)
        mex(args{:}, '-output', mexes{i, 2}, fullfile(src, mexes{i, 1}));
    end
end
