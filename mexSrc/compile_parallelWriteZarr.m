debug = false;
% ADD --disable-new-dtags
if isunix && ~ismac
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
    mkdir('../cpp-zarr_linux');
    movefile('parallelwritezarr.mexa64','../cpp-zarr_linux/parallelWriteZarr.mexa64');
elseif ismac
    if debug
        %mex -v -g CXX="/usr/local/opt/llvm/bin/clang++" CXXOPTIMFLAGS="" LDOPTIMFLAGS='-g -O0 -Wall -Wextra' CXXFLAGS='$CXXFLAGS -g -O0 -Wall -Wextra -fopenmp' LDFLAGS='$LDFLAGS -g -O0 -fopenmp' '-I/usr/local/include/' -L'/usr/local/lib/' -lblosc -lblosc2 -lz -luuid parallelwritezarr.cpp helperfunctions.cpp zarr.cpp parallelwritezarrread.cpp
        %mex -v -g CXXOPTIMFLAGS='' LDOPTIMFLAGS='-g -O0' CXXFLAGS='$CXXFLAGS -O0 -Xpreprocessor -fopenmp' LDFLAGS='$LDFLAGS -g -O0 -Xpreprocessor -fopenmp' '-I/usr/local/opt/libomp/include' '-I/usr/include' '-I/usr/local/include/' -L'/usr/local/lib/' -L'/usr/local/opt/libomp/lib' -lomp -lblosc -lblosc2 -lz -luuid parallelwritezarr.cpp helperfunctions.cpp zarr.cpp parallelwritezarrread.cpp
    else
        mex -v CXX="/usr/local/bin/g++-12" CXXOPTIMFLAGS='-O2 -DNDEBUG' LDOPTIMFLAGS='-O2 -DNDEBUG' CXXFLAGS='-fno-common -arch x86_64 -mmacosx-version-min=10.15 -fexceptions -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++11 -O2 -fopenmp -DMATLAB_DEFAULT_RELEASE=R2017b  -DUSE_MEX_CMD   -DMATLAB_MEX_FILE' LDFLAGS='$LDFLAGS -O2 -fopenmp' '-I/usr/local/include/' -lstdc++ -lblosc -lblosc2 -lz -luuid parallelwritezarr.cpp helperfunctions.cpp zarr.cpp parallelreadzarrwrapper.cpp
    end
    
    % We need to change all the current paths to be relative to the mex file
    system('install_name_tool -change /usr/local/opt/gcc@12/lib/gcc/12/libstdc++.6.dylib @loader_path/libstdc++.6.0.30.dylib parallelwritezarr.mexmaci64');
    system('install_name_tool -change /usr/local/opt/ossp-uuid/lib/libuuid.16.dylib @loader_path/libuuid.16.22.0.dylib parallelwritezarr.mexmaci64');
    system('install_name_tool -change /usr/local/opt/gcc@12/lib/gcc/12/libgomp.1.dylib @loader_path/libgomp.1.dylib parallelwritezarr.mexmaci64');
    system('install_name_tool -change @rpath/libblosc.1.dylib @loader_path/libblosc.1.21.0.0.dylib parallelwritezarr.mexmaci64');
    system('install_name_tool -change @rpath/libblosc2.2.dylib @loader_path/libblosc2.2.8.0.dylib parallelwritezarr.mexmaci64');
    system('install_name_tool -change /usr/lib/libz.1.dylib @loader_path/libz.1.2.13.dylib parallelwritezarr.mexmaci64');
    system('install_name_tool -change /usr/local/opt/gcc@12/lib/gcc/12/libgcc_s.1.1.dylib @loader_path/libgcc_s.1.1.0.dylib parallelwritezarr.mexmaci64');
    

    system('chmod 777 parallelwritezarr.mexmaci64');
    mkdir('../cpp-zarr_mac');
    movefile('parallelwritezarr.mexmaci64','../cpp-zarr_mac/parallelWriteZarr.mexmaci64');
elseif ispc
    mex CXX="C:/mingw64/bin/g++" -v CXXOPTIMFLAGS="-DNDEBUG -O2" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -fopenmp -O2' LDFLAGS='$LDFLAGS -fopenmp -O2' -I'C:/Program Files (x86)/nlohmann_json/include' -I'C:/Program Files (x86)/blosc2/include' '-LC:/Program Files (x86)/blosc2/lib' -I'C:/Program Files (x86)/blosc/include' '-LC:/Program Files (x86)/blosc/lib' -I'C:/Program Files (x86)/zlib/include' '-LC:/Program Files (x86)/zlib/lib' -lblosc.dll -lblosc2.dll -lzlib.dll parallelwritezarr.cpp helperfunctions.cpp zarr.cpp parallelreadzarrwrapper.cpp
    
    mkdir('../cpp-zarr_windows');
    movefile('parallelwritezarr.mexw64','../cpp-zarr_windows/parallelWriteZarr.mexw64');
end