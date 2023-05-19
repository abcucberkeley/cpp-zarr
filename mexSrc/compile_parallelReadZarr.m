debug = false;
% ADD --disable-new-dtags
if isunix && ~ismac
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
    mkdir('../cpp-zarr_linux');
    movefile('parallelreadzarr.mexa64','../cpp-zarr_linux/parallelReadZarr.mexa64');
elseif ismac
    %mex -v CXX="/usr/local/bin/g++-12" CXXOPTIMFLAGS='-O2 -DNDEBUG' LDOPTIMFLAGS='-O2 -DNDEBUG' CXXFLAGS='$CXXFLAGS -O2 -fopenmp' LDFLAGS='$LDFLAGS -O2 -fopenmp' '-I/usr/local/include/' -L'/usr/local/lib/' -lstdc++ -lblosc2 -lz -luuid parallelreadzarr.cpp helperfunctions.cpp zarr.cpp
    mex -v CXX="/usr/local/bin/g++-12" CXXOPTIMFLAGS='-O2 -DNDEBUG' LDOPTIMFLAGS='-O2 -DNDEBUG' CXXFLAGS='-fno-common -arch x86_64 -mmacosx-version-min=10.15 -fexceptions -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++11 -O2 -fopenmp -DMATLAB_DEFAULT_RELEASE=R2017b  -DUSE_MEX_CMD   -DMATLAB_MEX_FILE' LDFLAGS='$LDFLAGS -O2 -fopenmp' '-I/usr/local/include/' -lstdc++ -lblosc2 -lz -luuid parallelreadzarr.cpp helperfunctions.cpp zarr.cpp parallelreadzarrwrapper.cpp

    % We need to change all the current paths to be relative to the mex file
    system('install_name_tool -change /usr/local/opt/gcc@12/lib/gcc/12/libstdc++.6.dylib @loader_path/libstdc++.6.0.30.dylib parallelreadzarr.mexmaci64');
    system('install_name_tool -change /usr/local/opt/ossp-uuid/lib/libuuid.16.dylib @loader_path/libuuid.16.22.0.dylib parallelreadzarr.mexmaci64');
    system('install_name_tool -change /usr/local/opt/gcc@12/lib/gcc/12/libgomp.1.dylib @loader_path/libgomp.1.dylib parallelreadzarr.mexmaci64');
    system('install_name_tool -change @rpath/libblosc2.2.dylib @loader_path/libblosc2.2.8.0.dylib parallelreadzarr.mexmaci64');
    system('install_name_tool -change /usr/lib/libz.1.dylib @loader_path/libz.1.2.13.dylib parallelreadzarr.mexmaci64');
    system('install_name_tool -change /usr/local/opt/gcc@12/lib/gcc/12/libgcc_s.1.1.dylib @loader_path/libgcc_s.1.1.0.dylib parallelreadzarr.mexmaci64');
   

    system('chmod 777 parallelreadzarr.mexmaci64');
    mkdir('../cpp-zarr_mac');
    movefile('parallelreadzarr.mexmaci64','../cpp-zarr_mac/parallelReadZarr.mexmaci64');

end
