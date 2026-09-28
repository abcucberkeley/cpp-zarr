function test_matlab(mexDir)
% Smoke test for the compiled MATLAB mex files: write small random volumes,
% read them back, and check the data survives the round trip. Also checks a
% bounding-box (region) read and createZarrFile. Errors (non-zero exit under
% `matlab -batch`) on any failure so Jenkins fails the build.
%
% With no argument it auto-detects the platform's mex folder (matching the
% compile_*.m scripts), so Jenkins can call it quote-free right after compiling:
%   matlab -batch "compile_createZarrFile;compile_parallelReadZarr;compile_parallelWriteZarr;addpath ../tests;test_matlab;exit"
% Or pass an explicit folder: test_matlab('../linux')
    if nargin < 1
        if ispc
            mexDir = '../windows';
        elseif ismac
            if strcmp(computer, 'MACI64')
                mexDir = '../mac';
            else
                mexDir = '../macArm';
            end
        else
            mexDir = '../linux';
        end
    end
    addpath(mexDir);

    tmp = tempname;
    mkdir(tmp);
    cleanup = onCleanup(@() rmdir(tmp, 's')); %#ok<NASGU>

    rng(1234567);
    sz = [40 24 18];                       % d0, d1, d2
    types = {'uint8','int8','uint16','int16','uint32','int32','uint64','int64','single','double'};
    dtypes = {'<u1','<i1','<u2','<i2','<u4','<i4','<u8','<i8','<f4','<f8'};

    for k = 1:numel(types)
        t = types{k};
        if any(strcmp(t, {'single','double'}))
            data = cast((rand(sz) - 0.5) * 1000, t);
        elseif t(1) == 'u'
            data = cast(randi([0 200], sz), t);
        else
            data = cast(randi([-100 100], sz), t);   % negatives exercise signed types
        end

        f = fullfile(tmp, ['rt_' t '.zarr']);
        parallelWriteZarr(f, data);

        back = parallelReadZarr(f);
        assert(isa(back, t),            'dtype mismatch for %s: got %s', t, class(back));
        assert(isequal(size(back), sz), 'size mismatch for %s', t);
        assert(isequal(back, data),     'round-trip data mismatch for %s', t);

        % Bounding-box (region) read: 1-indexed [x0 y0 z0 x1 y1 z1], first half of d0.
        h = floor(sz(1)/2);
        sub = parallelReadZarr(f, 'bbox', [1 1 1 h sz(2) sz(3)]);
        assert(isequal(sub, data(1:h, :, :)), 'region read mismatch for %s', t);

        % A non-default compressor should also round trip.
        fz = fullfile(tmp, ['rt_' t '_zstd.zarr']);
        parallelWriteZarr(fz, data, 'cname', 'zstd');
        assert(isequal(parallelReadZarr(fz), data), 'zstd round-trip mismatch for %s', t);

        % C order: createZarrFile sets the order, then bbox writes go through the
        % C-order writer and reads through the C-order reader. The second write is
        % a box that is not chunk-aligned; the data around it must be kept.
        fcz = fullfile(tmp, ['rt_' t '_C.zarr']);
        createZarrFile(fcz, 'shape', sz, 'dtype', dtypes{k}, 'chunks', [16 16 16], 'order', 'C');
        parallelWriteZarr(fcz, data, 'bbox', [1 1 1 sz]);
        assert(isequal(parallelReadZarr(fcz), data), 'C-order round-trip mismatch for %s', t);
        box = [6 4 3 29 13 11];
        patch = data(box(1):box(4), box(2):box(5), box(3):box(6)) + 1;
        parallelWriteZarr(fcz, patch, 'bbox', box);
        expect = data;
        expect(box(1):box(4), box(2):box(5), box(3):box(6)) = patch;
        assert(isequal(parallelReadZarr(fcz), expect), 'C-order bbox write mismatch for %s', t);

        fprintf('PASS  %s\n', t);
    end

    % createZarrFile writes only .zarray metadata; reading it back (no chunks on
    % disk) must return an array of the fill value (0) with the right shape/type.
    fc = fullfile(tmp, 'meta_only.zarr');
    createZarrFile(fc, 'shape', sz, 'dtype', '<u2', 'chunks', [16 16 16]);
    z = parallelReadZarr(fc);
    assert(isequal(size(z), sz) && isa(z, 'uint16') && ~any(z(:)), 'createZarrFile mismatch');
    fprintf('PASS  createZarrFile\n');

    disp('MATLAB round-trip tests PASSED');
end
