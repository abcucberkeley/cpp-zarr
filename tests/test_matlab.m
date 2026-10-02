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

    % N-dimensional arrays. 2D data is a true 2D array; the old 3D-style
    % arguments (3-value chunks, 6-value bbox) still work on it.
    a2 = cast(randi([0 200], [70 45]), 'uint16');
    f2 = fullfile(tmp, 'nd2.zarr');
    parallelWriteZarr(f2, a2, 'chunks', [32 16 99]);
    meta = jsondecode(fileread(fullfile(f2, '.zarray')));
    assert(isequal(meta.shape(:)', [70 45]) && isequal(meta.chunks(:)', [32 16]), '2D metadata mismatch');
    assert(isequal(parallelReadZarr(f2), a2), '2D round-trip mismatch');
    assert(isequal(parallelReadZarr(f2, 'bbox', [3 5 40 30]), a2(3:40, 5:30)), '2D bbox read mismatch');
    assert(isequal(parallelReadZarr(f2, 'bbox', [3 5 1 40 30 1]), a2(3:40, 5:30)), '2D 6-value bbox read mismatch');
    p = a2(10:50, 7:40) + 1;
    parallelWriteZarr(f2, p, 'bbox', [10 7 50 40]);
    e2 = a2; e2(10:50, 7:40) = p;
    assert(isequal(parallelReadZarr(f2), e2), '2D bbox write mismatch');
    parallelWriteZarr(f2, p - 1, 'bbox', [10 7 1 50 40 1]);
    assert(isequal(parallelReadZarr(f2), a2), '2D 6-value bbox write mismatch');
    % a 2D region written into a [m n 1] array (how 2D data used to be stored)
    f21 = fullfile(tmp, 'nd2_as3d.zarr');
    createZarrFile(f21, 'shape', [70 45 1], 'dtype', '<u2', 'chunks', [32 16 1]);
    parallelWriteZarr(f21, a2, 'bbox', [1 1 70 45]);
    assert(isequal(parallelReadZarr(f21), a2), '2D region into [m n 1] array mismatch');
    fprintf('PASS  2D\n');

    for nd = [4 5]
        sz = [9 7 6 5 4];  sz = sz(1:nd);
        ch = [4 3 4 2 3];  ch = ch(1:nd);
        a = cast(randi([0 200], sz), 'uint16');
        lo = 2 * ones(1, nd);  hi = sz - 1;
        idx = arrayfun(@(d) lo(d):hi(d), 1:nd, 'UniformOutput', false);
        for order = {'F', 'C'}
            f = fullfile(tmp, sprintf('nd%d_%s.zarr', nd, order{1}));
            if order{1} == 'F'
                parallelWriteZarr(f, a, 'chunks', ch);
            else
                createZarrFile(f, 'shape', sz, 'dtype', '<u2', 'chunks', ch, 'order', 'C');
                parallelWriteZarr(f, a, 'bbox', [ones(1, nd) sz]);
            end
            assert(isequal(parallelReadZarr(f), a), '%dD %s round-trip mismatch', nd, order{1});
            assert(isequal(parallelReadZarr(f, 'bbox', [lo hi]), a(idx{:})), '%dD %s bbox read mismatch', nd, order{1});
            p = a(idx{:}) + 1;
            parallelWriteZarr(f, p, 'bbox', [lo hi]);
            e = a;  e(idx{:}) = p;
            assert(isequal(parallelReadZarr(f), e), '%dD %s bbox write mismatch', nd, order{1});
        end
        % default chunks: 256 on the first three axes, 1 after
        fdef = fullfile(tmp, sprintf('nd%d_default.zarr', nd));
        parallelWriteZarr(fdef, a);
        meta = jsondecode(fileread(fullfile(fdef, '.zarray')));
        assert(isequal(meta.chunks(:)', [256 256 256 ones(1, nd - 3)]), '%dD default chunks mismatch', nd);
        assert(isequal(parallelReadZarr(fdef), a), '%dD default-chunk round-trip mismatch', nd);
        fprintf('PASS  %dD\n', nd);
    end

    % A 0-dimensional array (scalar): created with an empty shape, read as 1x1
    f0 = fullfile(tmp, 'nd0.zarr');
    createZarrFile(f0, 'shape', [], 'dtype', '<f8');
    meta = jsondecode(fileread(fullfile(f0, '.zarray')));
    assert(isempty(meta.shape), '0-D metadata mismatch');
    assert(isequal(parallelReadZarr(f0), 0), '0-D read mismatch');
    parallelWriteZarr(f0, 3.25, 'bbox', [1 1]);
    assert(isequal(parallelReadZarr(f0), 3.25), '0-D write mismatch');
    fprintf('PASS  0D\n');

    % Many dimensions (70), with partial chunks and a region read
    sz70 = ones(1, 70);  sz70([3 36 67 70]) = [5 3 4 6];
    ch70 = ones(1, 70);  ch70([3 36 67 70]) = [2 2 3 4];
    a70 = cast(randi([0 200], sz70), 'uint16');
    f70 = fullfile(tmp, 'nd70.zarr');
    parallelWriteZarr(f70, a70, 'chunks', ch70);
    assert(isequal(parallelReadZarr(f70), a70), '70-D round-trip mismatch');
    lo = ones(1, 70);  hi = sz70;  lo(3) = 2;  hi(70) = 5;
    idx = arrayfun(@(d) lo(d):hi(d), 1:70, 'UniformOutput', false);
    assert(isequal(parallelReadZarr(f70, 'bbox', [lo hi]), a70(idx{:})), '70-D bbox read mismatch');
    fprintf('PASS  70D\n');

    % 1D arrays read back as column vectors
    f1 = fullfile(tmp, 'nd1.zarr');
    createZarrFile(f1, 'shape', 100, 'dtype', '<u2', 'chunks', 32);
    v = cast((1:100)', 'uint16');
    parallelWriteZarr(f1, v, 'bbox', [1 100]);
    assert(isequal(parallelReadZarr(f1), v), '1D round-trip mismatch');
    assert(isequal(parallelReadZarr(f1, 'bbox', [11 50]), v(11:50)), '1D bbox read mismatch');
    fprintf('PASS  1D\n');

    % Sharding (chunk_shape): round trip, crop writes across shards (with and without
    % temporary files), and a rewrite with all zeros, in both orders; shards in
    % subfolders; and a non-sharded rewrite with all zeros (old chunks must not
    % survive it)
    a = cast(randi([1 200], [70 45 33]), 'uint16');
    for order = {'F', 'C'}
        fs = fullfile(tmp, ['shard_' order{1} '.zarr']);
        createZarrFile(fs, 'shape', size(a), 'dtype', '<u2', 'chunks', [32 32 20], 'chunk_shape', [16 16 10], 'order', order{1});
        parallelWriteZarr(fs, a, 'bbox', [1 1 1 size(a)]);
        assert(isequal(parallelReadZarr(fs), a), 'sharded %s round-trip mismatch', order{1});
        p = a(6:61, 8:40, 4:29) + 1;
        parallelWriteZarr(fs, p, 'bbox', [6 8 4 61 40 29]);
        e = a;  e(6:61, 8:40, 4:29) = p;
        assert(isequal(parallelReadZarr(fs), e), 'sharded %s crop write mismatch', order{1});
        p = a(3:50, 20:44, 2:15) + 2;
        parallelWriteZarr(fs, p, 'bbox', [3 20 2 50 44 15], 'uuid', 0);
        e(3:50, 20:44, 2:15) = p;
        assert(isequal(parallelReadZarr(fs), e), 'sharded %s crop write without uuid mismatch', order{1});
        parallelWriteZarr(fs, zeros(size(a), 'uint16'), 'bbox', [1 1 1 size(a)]);
        assert(~any(parallelReadZarr(fs), 'all'), 'sharded %s zero rewrite mismatch', order{1});
    end
    fsub = fullfile(tmp, 'shard_subfolders.zarr');
    createZarrFile(fsub, 'shape', size(a), 'dtype', '<u2', 'chunks', [16 16 16], 'chunk_shape', [8 8 8], 'subfolders', [2 2 1]);
    parallelWriteZarr(fsub, a, 'bbox', [1 1 1 size(a)]);
    assert(isequal(parallelReadZarr(fsub), a), 'sharded subfolders round-trip mismatch');
    fz = fullfile(tmp, 'zero_rewrite.zarr');
    parallelWriteZarr(fz, a, 'chunks', [16 16 16]);
    parallelWriteZarr(fz, zeros(size(a), 'uint16'), 'chunks', [16 16 16]);
    assert(~any(parallelReadZarr(fz), 'all'), 'zero rewrite mismatch');
    fprintf('PASS  sharding and zero rewrites\n');

    % createZarrFile writes only .zarray metadata; reading it back (no chunks on
    % disk) must return an array of the fill value (0) with the right shape/type.
    fc = fullfile(tmp, 'meta_only.zarr');
    createZarrFile(fc, 'shape', sz, 'dtype', '<u2', 'chunks', [16 16 16]);
    z = parallelReadZarr(fc);
    assert(isequal(size(z), sz) && isa(z, 'uint16') && ~any(z(:)), 'createZarrFile mismatch');
    fprintf('PASS  createZarrFile\n');

    disp('MATLAB round-trip tests PASSED');
end
