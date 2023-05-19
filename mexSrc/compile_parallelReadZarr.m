debug = false;
% ADD --disable-new-dtags
if isunix
    if debug
        mex -v LDOPTIMFLAGS="-g -O0 -Wall -Wextra -Wl',-rpath='''$ORIGIN''''" CXXFLAGS='$CXXFLAGS -Wall -Wextra -g -O0 -fopenmp' LDFLAGS='$LDFLAGS -g -O0 -fopenmp' -I'/home/matt/c-zarr/cBlosc2/include' -L'/home/matt/c-zarr/cBlosc2/lib' -lblosc2 -lz -luuid parallelreadzarr.cpp zarr.cpp helperfunctions.cpp
    else
        mex -v CXXOPTIMFLAGS="-DNDEBUG -O2" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -fopenmp -O2' LDFLAGS='$LDFLAGS -fopenmp -O2' -I'/clusterfs/fiona/matthewmueller/cppZarrTest' -I'/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.8.0/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.8.0/lib64' -lblosc2 -lz -luuid parallelreadzarr.cpp zarr.cpp helperfunctions.cpp parallelreadzarrwrapper.cpp
        %mex -v CXXOPTIMFLAGS="-DNDEBUG -O3" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O3 -DNDEBUG" CXXFLAGS='$CXXFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp -O3' -I'/clusterfs/fiona/matthewmueller/cppZarrTest' -I'/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.8.0/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.8.0/lib64' -lblosc2 -lz -luuid parallelreadzarr.cpp zarr.cpp helperfunctions.cpp parallelreadzarrwrapper.cpp
        %mex -v CXXOPTIMFLAGS="-DNDEBUG -O2" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -fopenmp -O2' LDFLAGS='$LDFLAGS -fopenmp -O2' -I'/home/matt/c-zarr/cBlosc2/include' -L'/home/matt/c-zarr/cBlosc2/lib' -lblosc2 -lz -luuid parallelreadzarr.cpp zarr.cpp helperfunctions.cpp
    end
    % Need to change the library name because matlab preloads their own version
    % of libstdc++
    % Setting it to libstdc++.so.6.0.30 as of MATLAB R2022b
    %system('patchelf --replace-needed libc.so.6 libc.so.6.0.0 parallelreadzarr.mexa64');
    system('patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.30 parallelreadzarr.mexa64');
    %system('export LD_LIBRARY_PATH="";patchelf --replace-needed libc.so.6 libc.so.6.0.0 parallelreadzarr.mexa64');
    %system('export LD_LIBRARY_PATH="";patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.30 parallelreadzarr.mexa64');
    movefile('parallelreadzarr.mexa64','../cpp-zarr_linux/parallelReadZarr.mexa64');
end
