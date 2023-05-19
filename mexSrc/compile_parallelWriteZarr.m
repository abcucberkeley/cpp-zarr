debug = false;
% ADD --disable-new-dtags
if isunix
    if debug
        mex -g -v LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O0 -g" CXXFLAGS='$CXXFLAGS -fopenmp -O0 -g' LDFLAGS='$LDFLAGS -fopenmp -O0 -g' -I'/clusterfs/fiona/matthewmueller/cppZarrTest' -I'/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.8.0/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.8.0/lib64' -I'/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/lib' -lblosc -lblosc2 -lz -luuid parallelwritezarr.cpp helperfunctions.cpp zarr.cpp parallelreadzarrwrapper.cpp
        %mex -v LDOPTIMFLAGS="-g -O0 -Wall -Wextra -Wl',-rpath='''$ORIGIN''''" CXXFLAGS='$CXXFLAGS -Wall -Wextra -g -O0 -fopenmp' LDFLAGS='$LDFLAGS -g -O0 -fopenmp' '-I/home/matt/c-zarr/cBlosc/include' '-I/home/matt/c-zarr/cBlosc2/include' '-L/home/matt/c-zarr/cBlosc/lib' -lblosc '-L/home/matt/c-zarr/cBlosc2/lib' -lblosc2 -lz -luuid parallelwritezarr.cpp helperfunctions.cpp zarr.cpp parallelreadzarrwrapper.cpp
    else
        %mex -v CXXOPTIMFLAGS="-DNDEBUG -O2" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -fopenmp -O2' LDFLAGS='$LDFLAGS -fopenmp -O2' '-I/home/matt/c-zarr/cBlosc/include' '-I/home/matt/c-zarr/cBlosc2/include' '-L/home/matt/c-zarr/cBlosc/lib' -lblosc '-L/home/matt/c-zarr/cBlosc2/lib' -lblosc2 -lz -luuid parallelwritezarr.cpp helperfunctions.cpp zarr.cpp parallelwritezarrread.cpp
        mex -v CXXOPTIMFLAGS="-DNDEBUG -O2" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -fopenmp -O2' LDFLAGS='$LDFLAGS -fopenmp -O2' -I'/clusterfs/fiona/matthewmueller/cppZarrTest' -I'/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.8.0/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.8.0/lib64' -I'/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/lib' -lblosc -lblosc2 -lz -luuid parallelwritezarr.cpp helperfunctions.cpp zarr.cpp parallelreadzarrwrapper.cpp
    end
    % Need to change the library name because matlab preloads their own version
    % of libstdc++
    % Setting it to libstdc++.so.6.0.30 as of MATLAB R2022b
    system('patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.30 parallelwritezarr.mexa64');
    %system('export LD_LIBRARY_PATH="";patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.30 parallelWriteZarr.mexa64');
    movefile('parallelwritezarr.mexa64','../cpp-zarr_linux/parallelWriteZarr.mexa64');
end