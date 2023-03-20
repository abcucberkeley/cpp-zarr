
if isunix
    mex -v CXXOPTIMFLAGS="-O3 -DNDEBUG" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O3 -DNDEBUG" CXXFLAGS='$CXXFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' -I'/usr/local/include' -luuid createzarrfile.cpp helperfunctions.cpp zarr.cpp
    % Need to change the library name because matlab preloads their own version
    % of libstdc++
    % Setting it to libstdc++.so.6.0.30 as of MATLAB R2022b
    system('export LD_LIBRARY_PATH="";patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.30 createzarrfile.mexa64');
    movefile('createzarrfile.mexa64','../cpp-zarr_linux/createZarrFile.mexa64');

end