function test_matlab(mexDir)
% Smoke test for the compiled MATLAB mex files: write small random volumes,
% read them back, and check the data survives the round trip. Also checks a
% bounding-box (region) read, createZarrFile, Zarr v3 writing and
% convertZarrToV3. Errors (non-zero exit under `matlab -batch`) on any failure so
% Jenkins fails the build.
%
% With no argument it auto-detects the platform's mex folder (matching the
% compile_*.m scripts), so Jenkins can call it quote-free right after compiling:
%   matlab -batch "compile_createZarrFile;compile_parallelReadZarr;compile_parallelWriteZarr;compile_convertZarrToV3;addpath ../tests;test_matlab;exit"
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

    % Zarr v3 ('zarr_format', 3): C order and c/0/0/0 chunk files by default, F order
    % when asked; sharded arrays from createZarrFile with bbox writes into them; an
    % existing array keeps its format; a v2 array converted with convertZarrToV3
    a = cast(randi([1 200], [70 45 33]), 'uint16');
    for order = {'', 'F'}
        fv = fullfile(tmp, ['v3_' order{1} '.zarr']);
        if isempty(order{1})
            parallelWriteZarr(fv, a, 'zarr_format', 3, 'chunks', [32 16 16]);
        else
            parallelWriteZarr(fv, a, 'zarr_format', 3, 'chunks', [32 16 16], 'order', order{1}, 'cname', 'gzip');
        end
        meta = jsondecode(fileread(fullfile(fv, 'zarr.json')));
        transposed = contains(fileread(fullfile(fv, 'zarr.json')), '"transpose"');
        assert(meta.zarr_format == 3 && ~isfile(fullfile(fv, '.zarray')) && isfile(fullfile(fv, 'c', '0', '0', '0')) && ...
               transposed == strcmp(order{1}, 'F'), 'v3 %s metadata mismatch', order{1});
        assert(isequal(parallelReadZarr(fv), a), 'v3 %s round-trip mismatch', order{1});
        p = a(6:61, 8:40, 4:29) + 1;
        parallelWriteZarr(fv, p, 'bbox', [6 8 4 61 40 29]);
        e = a;  e(6:61, 8:40, 4:29) = p;
        assert(isequal(parallelReadZarr(fv), e), 'v3 %s bbox write mismatch', order{1});
    end
    for order = {'C', 'F'}
        fs = fullfile(tmp, ['v3_shard_' order{1} '.zarr']);
        createZarrFile(fs, 'shape', size(a), 'dtype', '<u2', 'chunks', [32 32 20], 'chunk_shape', [16 16 10], ...
                       'order', order{1}, 'zarr_format', 3);
        parallelWriteZarr(fs, a, 'bbox', [1 1 1 size(a)]);
        assert(isequal(parallelReadZarr(fs), a), 'v3 sharded %s round-trip mismatch', order{1});
        p = a(6:61, 8:40, 4:29) + 1;
        parallelWriteZarr(fs, p, 'bbox', [6 8 4 61 40 29], 'uuid', 0);
        e = a;  e(6:61, 8:40, 4:29) = p;
        assert(isequal(parallelReadZarr(fs), e), 'v3 sharded %s crop write mismatch', order{1});
    end
    % (the last one: kept as v3 by writes and createZarrFile without zarr_format)
    parallelWriteZarr(fs, a);
    assert(isfile(fullfile(fs, 'zarr.json')) && ~isfile(fullfile(fs, '.zarray')) && isequal(parallelReadZarr(fs), a), ...
           'v3 rewrite mismatch');
    createZarrFile(fs, 'shape', [10 7 5], 'dtype', '<u2');
    assert(isfile(fullfile(fs, 'zarr.json')) && ~isfile(fullfile(fs, '.zarray')), 'v3 createZarrFile did not keep the format');
    for attempt = 1 : 4
        try
            switch attempt
                case 1, parallelWriteZarr(fs, a(1:2, 1:2, 1:2), 'bbox', [1 1 1 2 2 2], 'zarr_format', 2);
                case 2, parallelWriteZarr(fs, a, 'zarr_format', 2);
                case 3, createZarrFile(fs, 'shape', [10 7 5], 'dtype', '<u2', 'zarr_format', 2);
                case 4, parallelWriteZarr(fullfile(tmp, 'v3_bad.zarr'), a, 'zarr_format', 4);
            end
            refused = false;
        catch ME
            refused = contains(ME.message, 'Zarr v3') || contains(ME.message, 'zarr_format');
        end
        assert(refused, 'Zarr format mismatch %d was not refused', attempt);
    end
    fc2 = fullfile(tmp, 'v2_to_v3.zarr');
    parallelWriteZarr(fc2, a, 'chunks', [32 16 16]);
    fid = fopen(fullfile(fc2, '.zattrs'), 'w');  fprintf(fid, '{"units": "um"}');  fclose(fid);
    convertZarrToV3(fc2);
    meta = jsondecode(fileread(fullfile(fc2, 'zarr.json')));
    assert(~isfile(fullfile(fc2, '.zarray')) && ~isfile(fullfile(fc2, '.zattrs')) && strcmp(meta.attributes.units, 'um') && ...
           isequal(parallelReadZarr(fc2), a), 'converted array mismatch');
    p = a(3:50, 20:44, 2:15) + 2;
    parallelWriteZarr(fc2, p, 'bbox', [3 20 2 50 44 15]);
    e = a;  e(3:50, 20:44, 2:15) = p;
    assert(isequal(parallelReadZarr(fc2), e), 'converted array bbox write mismatch');
    try
        convertZarrToV3(fc2);
        refused = false;
    catch ME
        refused = contains(ME.message, 'not a Zarr v2 array');
    end
    assert(refused, 'converting a v3 array was not refused');
    % shard and inner chunk sizes on every axis of a 4D array (v2 and v3)
    a4 = cast(randi([1 200], [8 12 16 20]), 'uint16');
    for fmt = [2 3]
        f4 = fullfile(tmp, sprintf('shard4d_v%d.zarr', fmt));
        parallelWriteZarr(f4, a4, 'chunks', [4 6 8 10], 'chunk_shape', [2 3 4 5], 'zarr_format', fmt);
        if fmt == 2
            meta = jsondecode(fileread(fullfile(f4, '.zarray')));
            shardShape = meta.chunks(:)';
            innerShape = meta.codecs.configuration.chunk_shape(:)';
        else
            meta = jsondecode(fileread(fullfile(f4, 'zarr.json')));
            shardShape = meta.chunk_grid.configuration.chunk_shape(:)';
            innerShape = meta.codecs.configuration.chunk_shape(:)';
        end
        assert(isequal(shardShape, [4 6 8 10]) && isequal(innerShape, [2 3 4 5]), '4D v%d shard shapes mismatch', fmt);
        assert(isequal(parallelReadZarr(f4), a4), '4D v%d sharded round-trip mismatch', fmt);
    end
    fsub = fullfile(tmp, 'v2_subfolders.zarr');
    createZarrFile(fsub, 'shape', [70 45 33], 'dtype', '<u2', 'chunks', [16 16 16], 'subfolders', [2 2 1]);
    try
        convertZarrToV3(fsub);
        refused = false;
    catch ME
        refused = contains(ME.message, 'subfolders') && isfile(fullfile(fsub, '.zarray'));
    end
    assert(refused, 'converting an array with subfolders was not refused');
    fprintf('PASS  Zarr v3 writing and conversion\n');

    % createZarrFile writes only .zarray metadata; reading it back (no chunks on
    % disk) must return an array of the fill value (0) with the right shape/type.
    fc = fullfile(tmp, 'meta_only.zarr');
    createZarrFile(fc, 'shape', sz, 'dtype', '<u2', 'chunks', [16 16 16]);
    z = parallelReadZarr(fc);
    assert(isequal(size(z), sz) && isa(z, 'uint16') && ~any(z(:)), 'createZarrFile mismatch');
    fprintf('PASS  createZarrFile\n');

    % Test arrays written by zarr-python 3 and TensorStore (tests/test_arrays): Zarr v3
    % arrays, and v2 arrays with codecs and fill values cpp-zarr does not write itself.
    % Full and region reads match what zarr-python reads, arrays cpp-zarr cannot read
    % are rejected, and copies of them can be written into and converted
    v3dir = fullfile(fileparts(mfilename('fullpath')), 'test_arrays');
    if isfile(fullfile(v3dir, 'arrays.json'))
        checkTestArrays(v3dir, tmp);
    else
        % made by tests/make_test_arrays.py; CI makes them and requires them
        assert(isempty(getenv('CPPZARR_REQUIRE_TEST_ARRAYS')), 'test arrays not found in %s', v3dir);
        fprintf('SKIP  test arrays not found (make them with tests/make_test_arrays.py)\n');
    end

    disp('MATLAB round-trip tests PASSED');
end

function checkTestArrays(v3dir, tmp)
% Read every test array (arrays.json) and compare with the values zarr-python reads;
% check the rejected ones are rejected; write into copies of them and convert those
    fixtures = jsondecode(fileread(fullfile(v3dir, 'arrays.json'))).arrays;
    classes = struct('u1', 'uint8', 'i1', 'int8', 'u2', 'uint16', 'i2', 'int16', 'u4', 'uint32', ...
                     'i4', 'int32', 'u8', 'uint64', 'i8', 'int64', 'f4', 'single', 'f8', 'double', 'b1', 'logical');
    nRead = 0;  nRejected = 0;
    for k = 1 : numel(fixtures)
        if iscell(fixtures), fx = fixtures{k}; else, fx = fixtures(k); end
        fz = fullfile(v3dir, [fx.name '.zarr']);
        if isfield(fx, 'error')
            try
                parallelReadZarr(fz);
                rejected = false;
            catch ME
                rejected = contains(ME.message, fx.error);
            end
            assert(rejected, 'test array %s was not rejected', fx.name);
            nRejected = nRejected + 1;
            continue;
        end
        cls = classes.(fx.dtype(2:3));
        fid = fopen(fullfile(v3dir, [fx.name '.bin']), 'r', 'ieee-le');
        if strcmp(cls, 'logical'), v = fread(fid, Inf, 'uint8=>uint8') ~= 0; else, v = fread(fid, Inf, [cls '=>' cls]); end
        fclose(fid);
        shp = fx.shape(:)';
        % the expected values are in C order: reverse the axes for MATLAB's F order
        if numel(shp) >= 2
            e = permute(reshape(v, fliplr(shp)), numel(shp):-1:1);
        else
            e = v;
        end
        a = parallelReadZarr(fz);
        assert(isa(a, cls) && isequaln(a, e), 'test array %s read mismatch', fx.name);
        if ~isempty(shp)
            s = ones(1, numel(shp));  t = shp;
            s(shp > 2) = 2;  t(shp > 2) = shp(shp > 2) - 1;
            idx = arrayfun(@(p, q) p:q, s, t, 'UniformOutput', false);
            assert(isequaln(parallelReadZarr(fz, 'bbox', [s t]), e(idx{:})), 'test array %s bbox read mismatch', fx.name);
        end
        nRead = nRead + 1;
    end
    % Writes into copies of them: a region, then the whole array, read back; arrays in
    % the opposite byte order are refused and left as they were. Then the v2 ones
    % converted to Zarr v3 (numcodecs zlib cannot be) read the same.
    nWritten = 0;
    for k = 1 : numel(fixtures)
        if iscell(fixtures), fx = fixtures{k}; else, fx = fixtures(k); end
        if isfield(fx, 'error') || strcmp(fx.dtype(2:3), 'b1'), continue; end
        cls = classes.(fx.dtype(2:3));
        fw = fullfile(tmp, ['ta_' fx.name '.zarr']);
        copyfile(fullfile(v3dir, [fx.name '.zarr']), fw);
        e = parallelReadZarr(fw);
        shp = size(e);
        if isempty(fx.shape), shp = []; elseif isscalar(fx.shape), shp = fx.shape; end
        bigEndian = isfile(fullfile(fw, 'zarr.json')) && contains(fileread(fullfile(fw, 'zarr.json')), '"big"');
        before = dir(fullfile(fw, '**', '*'));
        try
            if ~isempty(shp)
                s = ones(1, numel(shp));  t = shp;
                s(shp > 2) = 2;  t(shp > 2) = shp(shp > 2) - 1;
                idx = arrayfun(@(p, q) p:q, s, t, 'UniformOutput', false);
                p = cast(randi([0 100], size(e(idx{:}))), cls);
                parallelWriteZarr(fw, p, 'bbox', [s t]);
                e(idx{:}) = p;
                assert(isequaln(parallelReadZarr(fw), e), 'region write into test array %s mismatch', fx.name);
                w = cast(randi([0 100], size(e)), cls);
                parallelWriteZarr(fw, w, 'bbox', [ones(1, numel(shp)) shp]);
            else
                w = cast(randi([0 100]), cls);
                parallelWriteZarr(fw, w, 'bbox', [1 1]);
            end
            assert(~bigEndian && isequaln(parallelReadZarr(fw), w), 'write into test array %s mismatch', fx.name);
        catch ME
            after = dir(fullfile(fw, '**', '*'));
            assert(bigEndian && contains(ME.message, 'byte order') && isequal(sort({before.name}), sort({after.name})) && ...
                   isequal(sort([before.bytes]), sort([after.bytes])), 'write into test array %s failed: %s', fx.name, ME.message);
            continue;
        end
        if startsWith(fx.name, 'v2_')
            try
                convertZarrToV3(fw);
                converted = true;
            catch ME
                converted = false;
                assert(contains(fx.name, 'zlib') && contains(ME.message, 'Cannot convert'), 'converting %s failed: %s', fx.name, ME.message);
            end
            assert(~converted || (~contains(fx.name, 'zlib') && isequaln(parallelReadZarr(fw), w)), 'converted %s mismatch', fx.name);
        end
        nWritten = nWritten + 1;
    end
    fprintf('PASS  zarr-python and TensorStore test arrays (%d read, %d rejected, %d written into)\n', nRead, nRejected, nWritten);
end
