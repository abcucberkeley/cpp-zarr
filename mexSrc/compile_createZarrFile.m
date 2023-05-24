% ADD --disable-new-dtags
if isunix && ~ismac
    mex -v CXXOPTIMFLAGS="-O2 -DNDEBUG" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -O2 -fopenmp' LDFLAGS='$LDFLAGS -O2 -fopenmp' -I'/clusterfs/fiona/matthewmueller/cppZarrTest' -luuid createzarrfile.cpp helperfunctions.cpp zarr.cpp
    % Need to change the library name because matlab preloads their own version
    % of libstdc++
    % Setting it to libstdc++.so.6.0.30 as of MATLAB R2022b
    system('patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.30 createzarrfile.mexa64');
    %system('export LD_LIBRARY_PATH="";patchelf --replace-needed libstdc++.so.6 libstdc++.so.6.0.30 createzarrfile.mexa64');
    mkdir('../cpp-zarr_linux');
    movefile('createzarrfile.mexa64','../cpp-zarr_linux/createZarrFile.mexa64');
elseif ismac
    % Might have to do this part in terminal. First change the library
    % linked to libstdc++
    % system('install_name_tool -change @rpath/libgcc_s.1.1.dylib @loader_path/libgcc_s.1.1.0.dylib ../cpp-zarr_mac/libstdc++.6.0.30.dylib');
    
    %mex -v CXX="/usr/local/bin/g++-12" CXXOPTIMFLAGS='-O2 -DNDEBUG' LDOPTIMFLAGS='-O2 -DNDEBUG' CXXFLAGS='-fno-common -arch x86_64 -mmacosx-version-min=10.15 -fexceptions -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++11 -O2 -fopenmp -DMATLAB_DEFAULT_RELEASE=R2017b  -DUSE_MEX_CMD   -DMATLAB_MEX_FILE' LDFLAGS='$LDFLAGS -O2 -fopenmp' '-I/usr/local/include/' -lstdc++ -luuid createzarrfile.cpp helperfunctions.cpp zarr.cpp
    mex -v CXX="/usr/local/bin/g++-12" CXXOPTIMFLAGS='-O2 -DNDEBUG' LDOPTIMFLAGS='-O2 -DNDEBUG' CXXFLAGS='-fno-common -arch arm64 -mmacosx-version-min=10.15 -fexceptions -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++11 -O2 -fopenmp -DMATLAB_DEFAULT_RELEASE=R2017b  -DUSE_MEX_CMD   -DMATLAB_MEX_FILE' LDFLAGS='$LDFLAGS -O2 -fopenmp' '-I/usr/local/include/' -lstdc++ -luuid createzarrfile.cpp helperfunctions.cpp zarr.cpp
    %mex -v CXX='/usr/local/bin/g++-12' CXXOPTIMFLAGS='-O2 -DNDEBUG' LDOPTIMFLAGS='-O2 -DNDEBUG' CXXFLAGS='$CXXFLAGS -O2 -fopenmp' LDFLAGS='$LDFLAGS -O2 -fopenmp' '-I/usr/local/include/' -L'/usr/local/lib/' -luuid createZarrFile.cpp
    %mex -v CXX="/usr/local/opt/llvm/bin/clang++" CXXOPTIMFLAGS='-O2 -DNDEBUG' LDOPTIMFLAGS='-O2 -DNDEBUG' CXXFLAGS='$CXXFLAGS -O2 -fopenmp' LDFLAGS='$LDFLAGS -O2 -fopenmp' '-I/usr/local/include/' -L'/usr/local/lib/' -luuid createZarrFile.cpp helperfunctions.cpp zarr.cpp

    % We need to change all the current paths to be relative to the mex file
    system('install_name_tool -change /usr/local/opt/gcc@12/lib/gcc/12/libstdc++.6.dylib @loader_path/libstdc++.6.0.30.dylib createzarrfile.mexmaci64');
    system('install_name_tool -change /usr/local/opt/gcc@12/lib/gcc/12/libgcc_s.1.1.dylib @loader_path/libgcc_s.1.1.0.dylib createzarrfile.mexmaci64');
    system('install_name_tool -change /usr/local/opt/ossp-uuid/lib/libuuid.16.dylib @loader_path/libuuid.16.22.0.dylib createzarrfile.mexmaci64');
    system('install_name_tool -change /usr/local/opt/gcc@12/lib/gcc/12/libgomp.1.dylib @loader_path/libgomp.1.dylib createzarrfile.mexmaci64');
    

    system('chmod 777 createzarrfile.mexmaci64');
    mkdir('../cpp-zarr_mac');
    movefile('createzarrfile.mexmaci64','../cpp-zarr_mac/createZarrFile.mexmaci64');
elseif ispc
    mex CXX="C:/mingw64/bin/g++" -v CXXOPTIMFLAGS="-DNDEBUG -O2" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -fopenmp -O2' LDFLAGS='$LDFLAGS -fopenmp -O2' -I'C:/Program Files (x86)/nlohmann_json/include' createzarrfile.cpp helperfunctions.cpp zarr.cpp
    
    mkdir('../cpp-zarr_windows');
    movefile('createzarrfile.mexw64','../cpp-zarr_windows/createZarrFile.mexw64');
end