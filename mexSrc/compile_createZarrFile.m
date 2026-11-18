% Builds against the cpp-zarr install in ../jenkinsBuild/install, where the release builds install it (cmake
% --install into jenkinsBuild/install), and is run from this folder (mexSrc): -outdir and the other paths below
% are relative to it
% ADD --disable-new-dtags
if isunix && ~ismac
    releaseFolder = '../linux';
    if ~exist(releaseFolder, 'dir')
        mkdir(releaseFolder);
    end
    mex -outdir ../linux -output createZarrFile.mexa64 -v CXXOPTIMFLAGS="-O2 -DNDEBUG" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -O2 -fopenmp' LDFLAGS='$LDFLAGS -O2 -fopenmp -static-libstdc++ -Wl,-S' LINKLIBS='-L../jenkinsBuild/install/lib64 -lcppZarr -Wl,--as-needed -Wl,-rpath-link,$MATLABROOT/bin/$ARCH -L"$MATLABROOT/bin/$ARCH" -lmx -lmex -lm -lmat' -I'../jenkinsBuild/install/include' createzarrfilemex.cpp
    % libstdc++ is linked in statically (-static-libstdc++), so the MEX file needs no libstdc++ at run time, neither
    % MATLAB's (the 6.0.28 of R2024a and older releases is too old for GCC 13's code) nor a copy next to it. LINKLIBS
    % is MATLAB's own without its -lstdc++, which bound the MEX file to the libstdc++ MATLAB has loaded, and without
    % -lMatlabDataArray (the MEX files use the C API); a LINKLIBS given replaces the -L and -l options as well, so
    % the library is named in it. Release builds are linked with -Wl,-S: libstdc++.a's debug information names the
    % GCC's build folder. The release folder ships libgomp alone with the library (libgcc_s is MATLAB's or the
    % system's: none newer than GCC 4.2's is needed)
    system(['patchelf --force-rpath --set-rpath ''$ORIGIN'' ' releaseFolder '/createZarrFile.mexa64']);
elseif ismac
    % (-Wl,-S: no debug map, which named the objects of the static libstdc++.a linked in by their absolute paths
    % and is of no use without them, as the library is linked)
    % Might have to do this part in terminal. First change the library
    % linked to libstdc++
    % system('install_name_tool -change @rpath/libgcc_s.1.1.dylib @loader_path/libgcc_s.1.1.0.dylib libstdc++.6.0.32.dylib');
    if computer == "MACI64"
        releaseFolder = '../mac';
        if ~exist(releaseFolder, 'dir')
            mkdir(releaseFolder);
        end
        mex -outdir ../mac -output createZarrFile.mexmaci64 -v CXX="/usr/local/bin/g++-13" CXXOPTIMFLAGS='-O2 -DNDEBUG' LDOPTIMFLAGS='-O2 -DNDEBUG' CXXFLAGS='-fno-common -arch arm64 -mmacosx-version-min=10.15 -fexceptions -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++11 -O2 -fopenmp -DMATLAB_DEFAULT_RELEASE=R2017b  -DUSE_MEX_CMD   -DMATLAB_MEX_FILE' LDFLAGS='$LDFLAGS -O2 -fopenmp -Wl,-S' -I'../jenkinsBuild/install/include/' -L'../jenkinsBuild/install/lib' /usr/local/opt/gcc/lib/gcc/current/libstdc++.a -lcppZarr createzarrfilemex.cpp
    
        % We need to change all the current paths to be relative to the mex file
        %system('install_name_tool -change /usr/local/opt/gcc/lib/gcc/current/libstdc++.6.dylib @loader_path/libstdc++.6.0.32.dylib ../mac/createZarrFile.mexmaci64');
        system('install_name_tool -change /usr/local/opt/gcc/lib/gcc/current/libgcc_s.1.1.dylib @loader_path/libgcc_s.1.1.0.dylib ../mac/createZarrFile.mexmaci64');
        system('install_name_tool -change /usr/local/opt/gcc/lib/gcc/current/libgomp.1.dylib @loader_path/libgomp.1.dylib ../mac/createZarrFile.mexmaci64');    
        system('install_name_tool -change @rpath/libcppZarr.dylib @loader_path/libcppZarr.dylib ../mac/createZarrFile.mexmaci64');
    
        system('chmod 777 ../mac/createZarrFile.mexmaci64');
    else
        releaseFolder = '../macArm';
        if ~exist(releaseFolder, 'dir')
            mkdir(releaseFolder);
        end
        mex -outdir ../macArm -output createZarrFile.mexmaca64 -v CXX="/opt/homebrew/bin/g++-13" CXXOPTIMFLAGS='-O2 -DNDEBUG' LDOPTIMFLAGS='-O2 -DNDEBUG' CXXFLAGS='-fno-common -arch arm64 -mmacosx-version-min=10.15 -fexceptions -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++11 -O2 -fopenmp -DMATLAB_DEFAULT_RELEASE=R2017b  -DUSE_MEX_CMD   -DMATLAB_MEX_FILE' LDFLAGS='$LDFLAGS -O2 -fopenmp -Wl,-S' -I'../jenkinsBuild/install/include/' -L'../jenkinsBuild/install/lib' /opt/homebrew/opt/gcc@13/lib/gcc/13/libstdc++.a -lcppZarr createzarrfilemex.cpp
    
        % We need to change all the current paths to be relative to the mex file
        %system('install_name_tool -change /usr/local/opt/gcc/lib/gcc/current/libstdc++.6.dylib @loader_path/libstdc++.6.0.32.dylib ../mac/createZarrFile.mexmaci64');
        system('install_name_tool -change /opt/homebrew/opt/gcc@13/lib/gcc/13/libgcc_s.1.1.dylib @loader_path/libgcc_s.1.1.0.dylib ../macArm/createZarrFile.mexmaca64');
        system('install_name_tool -change /opt/homebrew/opt/gcc@13/lib/gcc/13/libgomp.1.dylib @loader_path/libgomp.1.dylib ../macArm/createZarrFile.mexmaca64');    
        system('install_name_tool -change @rpath/libcppZarr.dylib @loader_path/libcppZarr.dylib ../macArm/createZarrFile.mexmaca64');
    
        system('chmod 777 ../macArm/createZarrFile.mexmaca64'); 
    end
elseif ispc
    setenv('MW_MINGW64_LOC','C:/mingw64');
    releaseFolder = '../windows';
    if ~exist(releaseFolder, 'dir')
        mkdir(releaseFolder);
    end
    mex -outdir ../windows -output createZarrFile.mexa64 -v CXX="C:/mingw64/bin/g++" CXXOPTIMFLAGS="-DNDEBUG -O2" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O2 -DNDEBUG" CXXFLAGS='$CXXFLAGS -fopenmp -O2' LDFLAGS='$LDFLAGS -fopenmp -O2' -I'../jenkinsBuild/install/include' -L'../jenkinsBuild/install/lib' C:\mingw64\lib\gcc\x86_64-w64-mingw32\12.2.0\libgcc_eh.a -lcppZarr.dll createzarrfilemex.cpp
end
