debug = false;

if isunix
    if debug
        mex -v LDOPTIMFLAGS="-g -O0 -Wall -Wextra -Wl',-rpath='''$ORIGIN''''" CXXFLAGS='$CXXFLAGS -Wall -Wextra -g -O0 -fopenmp' LDFLAGS='$LDFLAGS -g -O0 -fopenmp' -I'/home/matt/c-zarr/cBlosc2/include' -L'/home/matt/c-zarr/cBlosc2/lib' -lblosc2 -lz -luuid parallelreadzarr.cpp zarr.cpp helperfunctions.cpp
    else
        mex -v CXXOPTIMFLAGS="-DNDEBUG -O2" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -fopenmp -O2' LDFLAGS='$LDFLAGS -fopenmp -O2' -I'/home/matt/c-zarr/cBlosc2/include' -L'/home/matt/c-zarr/cBlosc2/lib' -lblosc2 -lz -luuid parallelreadzarr.cpp zarr.cpp helperfunctions.cpp
    end
    % Need to change the library name because matlab preloads their own version
    % of libstdc++
    % Setting it to libstdc++.so.6.0.30 as of MATLAB R2022b
    system('export LD_LIBRARY_PATH="";patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.30 parallelreadzarr.mexa64');
    movefile('parallelreadzarr.mexa64','../cpp-zarr_linux/parallelReadZarr.mexa64');
end
