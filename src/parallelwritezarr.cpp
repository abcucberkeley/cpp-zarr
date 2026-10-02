#include <algorithm>
#include <cstdint>
#include <cstring>
#include <omp.h>
#ifdef _WIN32
#include <sys/time.h>
#else
#include <uuid/uuid.h>
#endif
#include <fstream>
#include "blosc.h"
#include "parallelreadzarr.h"
#include "parallelwritezarr.h"
#include "helperfunctions.h"
#include "zarr.h"
#include "zlib.h"

// Open a chunk or shard file for writing, making its folder first if it is missing
// (an array whose metadata another program wrote may not have its folders yet)
static bool openForWrite(std::ofstream &file, const std::string &path, const std::ios::openmode mode){
    file.open(path, std::ios::binary | mode);
    if(file.is_open()) return true;
    const size_t slash = path.find_last_of("/\\");
    if(slash == std::string::npos) return false;
    mkdirRecursive(path.substr(0, slash).c_str());
    file.clear();
    file.open(path, std::ios::binary | mode);
    return file.is_open();
}

uint8_t parallelWriteZarr(zarr &Zarr, void* zarrArr,
                          const std::vector<uint64_t> &startCoords,
                          const std::vector<uint64_t> &endCoords,
                          const std::vector<uint64_t> &writeShape,
                          const uint64_t bits, const bool useUuid,
                          const bool crop, const bool sparse){
    // The input is in F order: the first axis is contiguous
    std::vector<uint64_t> inStrides(writeShape.size());
    uint64_t acc = 1;
    for(uint64_t d = 0; d < writeShape.size(); d++){ inStrides[d] = acc; acc *= writeShape[d]; }
    return parallelWriteZarr(Zarr, zarrArr, startCoords, endCoords, writeShape, inStrides, bits, useUuid, crop, sparse);
}

uint8_t parallelWriteZarr(zarr &Zarr, void* zarrArr,
                          const std::vector<uint64_t> &startCoords,
                          const std::vector<uint64_t> &endCoords,
                          const std::vector<uint64_t> &writeShape,
                          const std::vector<uint64_t> &inStrides,
                          const uint64_t bits, const bool useUuid,
                          const bool crop, const bool sparse){
    // A 0-dimensional array is a single element: sharding does not apply
    if(Zarr.get_ndims() == 0 && Zarr.get_shard()){
        Zarr.set_errString("Sharding is not supported for 0-dimensional arrays\n");
        return 1;
    }
    // Only what the writer produces can be written: not Zarr v3 arrays (yet), and
    // not v2 arrays read with another compressor (numcodecs zlib or zstd) or byte
    // order (big-endian on this machine), which would end up with mixed chunks
    if(Zarr.get_zarr_format() == 3){
        Zarr.set_errString("Writing Zarr v3 arrays is not supported yet: "+Zarr.get_fileName()+"\n");
        return 1;
    }
    const std::string compressor = Zarr.get_compressor();
    if(!(compressor == "none" || compressor == "blosc" || (compressor == "gzip" && Zarr.get_cname() == "gzip"))){
        Zarr.set_errString("Writing arrays with the \""+Zarr.get_cname()+"\" compressor is not supported\n");
        return 1;
    }
    if(oppositeEndianness(Zarr.get_dtype())){
        Zarr.set_errString("Writing arrays of data type \""+Zarr.get_dtype()+"\" (opposite byte order) is not supported\n");
        return 1;
    }
    const uint64_t bytes = (bits/8);

    int32_t numWorkers = omp_get_max_threads();

    int32_t nBloscThreads = 1;
    if(numWorkers>Zarr.get_numChunks()){
        nBloscThreads = std::ceil(((double)numWorkers)/((double)Zarr.get_numChunks()));
        numWorkers = Zarr.get_numChunks();
    }

    int32_t batchSize = (Zarr.get_numChunks()-1)/numWorkers+1;
    // A sharded array is written with its inner chunk shape. The shard shape is
    // kept to read existing data when cropping, and restored on return.
    std::vector<uint64_t> shardChunks;
    if(Zarr.get_shard()){
        // batchSize has to align to a shard
        if(batchSize <= Zarr.get_numChunksPerShard()) batchSize = Zarr.get_numChunksPerShard();
        else batchSize += (Zarr.get_numChunksPerShard()-(batchSize%Zarr.get_numChunksPerShard()));

        std::vector<uint64_t> innerChunks(Zarr.get_ndims());
        for(uint64_t d = 0; d < innerChunks.size(); d++){
            shardChunks.push_back(Zarr.get_chunks(d));
            innerChunks[d] = Zarr.get_chunk_shape(d);
        }
        Zarr.set_chunks(innerChunks);
    }
    struct RestoreChunks {
        zarr &z;
        const std::vector<uint64_t> chunks;
        ~RestoreChunks(){ if(!chunks.empty()) z.set_chunks(chunks); }
    } restoreChunks{Zarr, shardChunks};

    const uint64_t nDims = Zarr.get_ndims();
    std::vector<uint64_t> chunkDims(nDims);
    uint64_t s = 1;
    for(uint64_t d = 0; d < nDims; d++){
        chunkDims[d] = Zarr.get_chunks(d);
        s *= chunkDims[d];
    }
    const uint64_t sB = s*bytes;

    if(writeShape.size() != nDims || inStrides.size() != nDims){
        Zarr.set_errString("The input has "+std::to_string(writeShape.size())+" dimensions and "+
                           std::to_string(inStrides.size())+" strides but the array has "+
                           std::to_string(nDims)+" dimensions\n");
        return 1;
    }

    // Element strides of an uncompressed chunk (F or C order) and of an F-order
    // chunk-sized region (existing data read back when cropping)
    std::vector<uint64_t> chunkStrides(nDims), chunkFStrides(nDims);
    {
        uint64_t acc = 1;
        for(uint64_t d = 0; d < nDims; d++){ chunkFStrides[d] = acc; acc *= chunkDims[d]; }
        if(Zarr.get_order() == "C"){
            acc = 1;
            for(int64_t d = (int64_t)nDims-1; d >= 0; d--){ chunkStrides[d] = acc; acc *= chunkDims[d]; }
        }
        else chunkStrides = chunkFStrides;
    }

    // Parse the fill value once (stoi would throw on Infinity-normalized fills,
    // and the C-order path below fills per element).
    const int fillValue = fillValueToInt(Zarr.get_fill_value());

    // The fill value for elements outside the written region: F-order chunks
    // have always been filled byte-wise (memset), C-order chunks per element.
    uint8_t fillElem[8] = {0};
    {
        const std::string &dtype = Zarr.get_dtype();
        const char kind = dtype.size() == 3 ? dtype[1] : '\0';
        if(kind == 'f' && bytes == 4){ const float v = (float)fillValue; memcpy(fillElem, &v, 4); }
        else if(kind == 'f' && bytes == 8){ const double v = (double)fillValue; memcpy(fillElem, &v, 8); }
        else if(bytes == 1){ const uint8_t v = (uint8_t)fillValue; memcpy(fillElem, &v, 1); }
        else if(bytes == 2){ const uint16_t v = (uint16_t)fillValue; memcpy(fillElem, &v, 2); }
        else if(bytes == 4){ const uint32_t v = (uint32_t)fillValue; memcpy(fillElem, &v, 4); }
        else if(bytes == 8){ const uint64_t v = (uint64_t)(int64_t)fillValue; memcpy(fillElem, &v, 8); }
        if(Zarr.get_order() != "C") memset(fillElem, fillValue, sizeof(fillElem));
    }

    // blosc (both blosc1 and blosc2) stores each chunk as a single frame with
    // 32-bit size fields, so one chunk cannot exceed BLOSC_MAX_BUFFERSIZE
    // (~2 GB) uncompressed. Exceeding it makes blosc_compress_ctx fail and, left
    // unchecked, wrote empty chunk files -- silently losing data. Reject it up
    // front with an actionable message. (gzip uses zlib's streaming API and is
    // not subject to this limit, so it is allowed through.)
    if(Zarr.get_cname() != "gzip" && Zarr.get_cname() != "none" && sB > (uint64_t)BLOSC_MAX_BUFFERSIZE){
        std::string dimsStr, productStr;
        for(uint64_t d = 0; d < nDims; d++){
            dimsStr += (d ? "x" : "")+std::to_string(chunkDims[d]);
            productStr += (d ? "*chunk[" : "chunk[")+std::to_string(d)+"]";
        }
        Zarr.set_errString("Chunk is too large for the \""+Zarr.get_cname()+
            "\" compressor: "+dimsStr+
            " x "+std::to_string(bytes)+" bytes/element = "+std::to_string(sB)+
            " bytes exceeds the blosc limit of "+std::to_string((uint64_t)BLOSC_MAX_BUFFERSIZE)+
            " bytes (~2 GB). Reduce the chunk size so that "+
            productStr+"*dtypeBytes < "+
            std::to_string((uint64_t)BLOSC_MAX_BUFFERSIZE+1)+", or use the gzip compressor.\n");
        return 1;
    }

    const std::string uuid(generateUUID());
    // A shard goes to a temporary file that is then moved into place when asked
    // (useUuid), and always when cropping: the shard's current contents are read
    // while its new ones are written
    const bool shardTmpFile = useUuid || crop;
    // Where a shard's index goes (the start or, by default, the end of its file)
    // and whether a CRC32C of it follows, as the array's metadata says
    const bool indexAtStart = Zarr.get_shardIndexAtStart();
    const bool indexChecksum = Zarr.get_shardIndexChecksum();

    // Sparse writes leave all-zero chunks unwritten, when that is how a missing
    // chunk reads back (a fill value of zero)
    void* zeroChunkUnc = NULL;
    if(sparse && fillValueIsZero(Zarr.get_fill_value())){
        zeroChunkUnc = calloc(s,bytes);
    }

    int err = 0;
    std::string errString;
    #pragma omp parallel for
    for(int32_t w = 0; w < numWorkers; w++){
        void* chunkUnC = malloc(sB);
        // gzip/deflate can expand incompressible data beyond blosc's
        // BLOSC_MAX_OVERHEAD guarantee, so size the compressed-chunk buffer to a
        // bound big enough for every codec (compressBound also covers gzip).
        const uint64_t chunkCCap = (uint64_t)compressBound(sB) + 64;
        void* chunkC = malloc(chunkCCap);
        void* cRegion = nullptr;
        if(!chunkUnC || !chunkC){
            #pragma omp critical
            {
                err = 1;
                errString = "Not enough memory for a chunk of "+std::to_string(sB)+" bytes\n";
            }
            free(chunkUnC);
            free(chunkC);
            continue;
        }
        // Sharding: the index of the shard being written, an (offset, nbytes) pair
        // per inner chunk (empty unless the chunk is written), where its next inner
        // chunk goes, and whether this write has started its file (a shard is always
        // written whole, never appended to an earlier file)
        const uint64_t emptyEntry = std::numeric_limits<uint64_t>::max();
        std::vector<uint64_t> shardIndex(Zarr.get_shard() ? 2*Zarr.get_numChunksPerShard() : 0, emptyEntry);
        const uint64_t indexBytes = shardIndex.size()*sizeof(uint64_t)+(indexChecksum ? sizeof(uint32_t) : 0);
        int64_t currChunk = -1;
        uint64_t shardBytes = 0;
        bool shardStarted = false;
        std::string shardPath;
        // Write the shard's index (and its CRC32C) after its inner chunks, or in the
        // room left for it at the start, then move the file into place
        auto finishShard = [&]() -> bool {
            const uint32_t crc = crc32c(reinterpret_cast<const uint8_t*>(shardIndex.data()), shardIndex.size()*sizeof(uint64_t));
            const std::string tmpPath = shardTmpFile ? shardPath+uuid : shardPath;
            std::ofstream file;
            const std::ios::openmode mode = !shardStarted ? std::ios::trunc :
                                            indexAtStart ? std::ios::in | std::ios::out : std::ios::app;
            if(!openForWrite(file, tmpPath, mode)){
                #pragma omp critical
                {
                    err = 1;
                    errString = "Check permissions or filepath. Cannot write to path: "+tmpPath+"\n";
                }
                return false;
            }
            file.write(reinterpret_cast<const char*>(shardIndex.data()), shardIndex.size()*sizeof(uint64_t));
            if(indexChecksum) file.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
            file.close();
            if(!file){
                if(shardTmpFile) remove(tmpPath.c_str());
                #pragma omp critical
                {
                    err = 1;
                    errString = "Could not write all of "+tmpPath+" (is the disk full?)\n";
                }
                return false;
            }
            if(shardTmpFile && !renameReplace(tmpPath, shardPath)){
                remove(tmpPath.c_str());
                #pragma omp critical
                {
                    err = 1;
                    errString = "Cannot move the temporary file into place: "+shardPath+"\n";
                }
                return false;
            }
            return true;
        };
        std::vector<uint64_t> cAV;
        std::vector<uint64_t> boxLo(nDims), boxHi(nDims), boxExt(nDims), inArray(nDims);
        const std::vector<uint64_t> zeros(nDims, 0);
        for(int64_t f = w*batchSize; f < (w+1)*batchSize; f++){
            if(f>=Zarr.get_numChunks()  || err) break;

            if(Zarr.get_shard()){
                currChunk++;
                // Moving on to the next shard: finish the previous one
                if(currChunk == (int64_t)Zarr.get_numChunksPerShard()){
                    if(!finishShard()) break;
                    currChunk = 0;
                }
                if(currChunk == 0){
                    std::fill(shardIndex.begin(), shardIndex.end(), emptyEntry);
                    shardBytes = indexAtStart ? indexBytes : 0;
                    shardStarted = false;
                    const std::vector<uint64_t> sAV = Zarr.chunkToShard(Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f)));
                    shardPath = Zarr.get_fileName()+"/"+Zarr.get_subfoldersString(sAV)+"/"+
                                Zarr.chunkNameToShardName(Zarr.get_chunkNames(f));
                }
                // Inner chunks past the array's edge are not stored
                const std::vector<uint64_t> pAV = Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f));
                bool pad = false;
                for(uint64_t d = 0; d < nDims; d++) pad = pad || pAV[d]*chunkDims[d] >= Zarr.get_shape(d);
                if(pad) continue;
            }
            
            cAV = Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f));
            cRegion = nullptr;

            // When cropping into an existing array, a chunk the written region only
            // partly covers (inside the array) needs its current contents
            bool partial = false;
            for(uint64_t d = 0; d < nDims; d++){
                partial = partial || cAV[d]*chunkDims[d] < startCoords[d] ||
                          ((cAV[d]+1)*chunkDims[d] > endCoords[d] && endCoords[d] < Zarr.get_shape(d));
            }
            if(crop && partial){
                std::vector<uint64_t> cStart(nDims), cEnd(nDims);
                for(uint64_t d = 0; d < nDims; d++){
                    cStart[d] = cAV[d]*chunkDims[d];
                    cEnd[d] = (cAV[d]+1)*chunkDims[d];
                }
                if(Zarr.get_shard()){
                    // read through the array's own (shard) chunk shape
                    zarr shardZarr(Zarr);
                    shardZarr.set_chunks(shardChunks);
                    cRegion = parallelReadZarrWriteWrapper(std::move(shardZarr), crop, cStart, cEnd);
                }
                else cRegion = parallelReadZarrWriteWrapper(Zarr, crop, cStart, cEnd);
                if(!cRegion){
                    err = 1;
                    errString = "Error in Writer Read. Chunk: "+Zarr.get_chunkNames(f)+"\n";
                    break;
                }
            }
            // Assemble the uncompressed chunk (F or C order) from the F-order input.
            // Elements inside the written region come from the input; outside it,
            // from the chunk's existing contents when cropping into an existing
            // array (cRegion, F order), and otherwise the fill value.
            uint64_t srcOff = 0, dstOff = 0;
            bool fullChunk = true;
            for(uint64_t d = 0; d < nDims; d++){
                const uint64_t b = cAV[d]*chunkDims[d];
                const uint64_t lo = std::max(b, startCoords[d]);
                const uint64_t hi = std::min(b+chunkDims[d], endCoords[d]);
                boxLo[d] = lo-b;
                boxHi[d] = hi > lo ? hi-b : lo-b;
                boxExt[d] = boxHi[d]-boxLo[d];
                fullChunk = fullChunk && boxLo[d] == 0 && boxHi[d] == chunkDims[d];
                srcOff += (lo-startCoords[d])*inStrides[d];
                dstOff += boxLo[d]*chunkStrides[d];
            }
            if(!fullChunk){
                if(cRegion){
                    // Existing data inside the array; fill beyond the array's edge
                    for(uint64_t d = 0; d < nDims; d++){
                        const uint64_t b = cAV[d]*chunkDims[d];
                        inArray[d] = Zarr.get_shape(d) > b ? std::min(chunkDims[d], Zarr.get_shape(d)-b) : 0;
                    }
                    copyBoxND(bytes, cRegion, chunkUnC, inArray, chunkFStrides, chunkStrides);
                    fillOutsideBoxND(bytes, chunkUnC, chunkDims, chunkStrides, zeros, inArray, fillElem);
                }
                else fillOutsideBoxND(bytes, chunkUnC, chunkDims, chunkStrides, boxLo, boxHi, fillElem);
            }
            copyBoxND(bytes, (const uint8_t*)zarrArr+srcOff*bytes, (uint8_t*)chunkUnC+dstOff*bytes,
                      boxExt, inStrides, chunkStrides);

            // An all-zero chunk is left unwritten (it reads back as zeros) and any
            // earlier copy of it removed; in a shard its index entry stays empty
            const std::string subfolderName = Zarr.get_subfoldersString(cAV);
            if(zeroChunkUnc && !memcmp(zeroChunkUnc,chunkUnC,sB)){
                if(!Zarr.get_shard()) remove((Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.get_chunkNames(f)).c_str());
                free(cRegion);
                cRegion = nullptr;
                continue;
            }

            // Use the same blosc compress as Zarr
            int64_t csize = 0;
            // Uncompressed arrays store the chunk as is
            const void* chunkOut = chunkC;

            if(Zarr.get_cname() == "none"){
                chunkOut = chunkUnC;
                csize = sB;
            }
            else if(Zarr.get_cname() != "gzip"){
                /*
                if(numWorkers<=Zarr.get_numChunks()){
                    csize = blosc_compress_ctx(Zarr.get_clevel(), BLOSC_SHUFFLE, bytes, sB, chunkUnC, chunkC, sB+BLOSC_MAX_OVERHEAD,Zarr.get_cname().c_str(),0,1);
                }
                else{
                    csize = blosc_compress_ctx(Zarr.get_clevel(), BLOSC_SHUFFLE, bytes, sB, chunkUnC, chunkC, sB+BLOSC_MAX_OVERHEAD,Zarr.get_cname().c_str(),0,numWorkers);
                }
                */
                csize = blosc_compress_ctx(Zarr.get_clevel(), BLOSC_SHUFFLE, bytes, sB, chunkUnC, chunkC, sB+BLOSC_MAX_OVERHEAD,Zarr.get_cname().c_str(),0,nBloscThreads);
                // A non-positive return means blosc could not compress the chunk
                // (e.g. it exceeds BLOSC_MAX_BUFFERSIZE). Fail loudly instead of
                // writing an empty/garbage chunk. (The chunk-size guard above
                // should already have caught the oversize case.)
                if(csize <= 0){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Compression error (blosc returned "+std::to_string(csize)+
                            "). ChunkName: "+Zarr.get_fileName()+"/"+subfolderName+"/"+
                            Zarr.get_chunkNames(f)+"\n";
                    }
                    break;
                }
            }
            else{
                csize = chunkCCap;
                z_stream stream;
                stream.zalloc = Z_NULL;
                stream.zfree = Z_NULL;
                stream.opaque = Z_NULL;

                stream.next_in = (uint8_t*)chunkUnC;
                stream.next_out = (uint8_t*)chunkC;

                stream.avail_in = sB;
                stream.avail_out = csize;
                int cErr = deflateInit2(&stream, Zarr.get_clevel(), Z_DEFLATED, MAX_WBITS + 16, MAX_MEM_LEVEL, Z_DEFAULT_STRATEGY);
                if(cErr){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Compression error. Error code: "+
                            std::to_string(cErr)+" ChunkName: "+
                            Zarr.get_fileName()+"/"+subfolderName+"/"+
                            Zarr.get_chunkNames(f)+"\n";
                    }
                    break;
                }

                cErr = deflate(&stream, Z_FINISH);

                if(cErr != Z_STREAM_END){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Compression error. Error code: "+
                            std::to_string(cErr)+" ChunkName: "+
                            Zarr.get_fileName()+"/"+subfolderName+"/"+
                            Zarr.get_chunkNames(f)+"\n";                }
                    break;
                }

                if(deflateEnd(&stream)){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Compression error. Error code: "+
                            std::to_string(cErr)+" ChunkName: "+
                            Zarr.get_fileName()+"/"+subfolderName+"/"+
                            Zarr.get_chunkNames(f)+"\n";
                    }
                    break;
                }
                csize = csize - stream.avail_out;
            }
            
            // Default write
            if(!Zarr.get_shard()){
                std::string fileName(Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.get_chunkNames(f));
                std::string fileNameFinal;
                if(useUuid){
                    fileNameFinal = std::string(fileName);
                    fileName.append(uuid);
                }
                std::ofstream file;
                if(!openForWrite(file, fileName, std::ios::trunc)){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Check permissions or filepath. Cannot write to path: "+
                            fileName+"\n";
                    }
                    break;
                }
                file.write(reinterpret_cast<const char*>(chunkOut),csize);
                file.close();
                if(!file){
                    if(useUuid) remove(fileName.c_str());
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Could not write all of "+fileName+" (is the disk full?)\n";
                    }
                    break;
                }
                if(useUuid && !renameReplace(fileName, fileNameFinal)){
                    remove(fileName.c_str());
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Cannot move the temporary file into place: "+
                            fileNameFinal+"\n";
                    }
                    break;
                }
            }
            // Sharding: add the inner chunk to its shard's file and record where it is
            else{
                const std::string tmpPath = shardTmpFile ? shardPath+uuid : shardPath;
                std::ofstream file;
                if(!openForWrite(file, tmpPath, shardStarted ? std::ios::app : std::ios::trunc)){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Check permissions or filepath. Cannot write to path: "+
                            tmpPath+"\n";
                    }
                    break;
                }
                // (room for an index at the start, written when the shard is finished)
                if(!shardStarted && indexAtStart){
                    const std::vector<char> room(indexBytes, 0);
                    file.write(room.data(), room.size());
                }
                file.write(reinterpret_cast<const char*>(chunkOut),csize);
                file.close();
                if(!file){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Could not write all of "+tmpPath+" (is the disk full?)\n";
                    }
                    break;
                }
                shardIndex[currChunk*2] = shardBytes;
                shardIndex[(currChunk*2)+1] = csize;
                shardBytes += csize;
                shardStarted = true;
            }
            free(cRegion);
            cRegion = nullptr;
        }

        // Finish this worker's last shard, unless the write failed: a partly written
        // shard is never moved into place (its temporary file is removed)
        if(Zarr.get_shard() && currChunk >= 0){
            if(!err) finishShard();
            else if(shardTmpFile) remove((shardPath+uuid).c_str());
        }
        free(chunkUnC);
        free(chunkC);

    }
    free(zeroChunkUnc);

    if(err) {
        Zarr.set_errString(errString);
        return 1;
    }
    else return 0;
}
