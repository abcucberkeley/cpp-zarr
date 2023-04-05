% ADD --disable-new-dtags
if isunix
    mex -v CXXOPTIMFLAGS="-O2 -DNDEBUG" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -O2 -fopenmp' LDFLAGS='$LDFLAGS -O2 -fopenmp' -I'/clusterfs/fiona/matthewmueller/cppZarrTest' -luuid createzarrfile.cpp helperfunctions.cpp zarr.cpp
    % Need to change the library name because matlab preloads their own version
    % of libstdc++
    % Setting it to libstdc++.so.6.0.30 as of MATLAB R2022b
    system('patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.30 createzarrfile.mexa64');
    %system('export LD_LIBRARY_PATH="";patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.30 createzarrfile.mexa64');
    movefile('createzarrfile.mexa64','../cpp-zarr_linux/createZarrFile.mexa64');

end