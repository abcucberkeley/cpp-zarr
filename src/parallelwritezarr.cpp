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

uint32_t crc32c(const uint8_t* data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    // CRC32C
    const uint32_t polynomial = 0x82F63B78;

    for (size_t i = 0; i < length; ++i) {
        crc ^= data[i];
        for (size_t j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (-(crc & 1) & polynomial);
        }
    }

    return ~crc;
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
    const uint64_t bytes = (bits/8);

    int32_t numWorkers = omp_get_max_threads();

    int32_t nBloscThreads = 1;
    if(numWorkers>Zarr.get_numChunks()){
        nBloscThreads = std::ceil(((double)numWorkers)/((double)Zarr.get_numChunks()));
        numWorkers = Zarr.get_numChunks();
    }

    int32_t batchSize = (Zarr.get_numChunks()-1)/numWorkers+1;
    if(Zarr.get_shard()){
        // batchSize has to align to a shard
        if(batchSize <= Zarr.get_numChunksPerShard()) batchSize = Zarr.get_numChunksPerShard();
        else batchSize += (Zarr.get_numChunksPerShard()-(batchSize%Zarr.get_numChunksPerShard()));

        // The chunk size is actually the inner chunk size now
        std::vector<uint64_t> innerChunks(Zarr.get_ndims());
        for(uint64_t d = 0; d < innerChunks.size(); d++) innerChunks[d] = Zarr.get_chunk_shape(d);
        Zarr.set_chunks(innerChunks);
    }

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

    void* zeroChunkUnc = NULL;
    if(sparse){
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
        int64_t currChunk = -1;
        uint64_t shardFooterSize;
        uint64_t* shardFooter = nullptr;
        std::vector<uint64_t> cAV;
        if(Zarr.get_shard()){
            shardFooterSize = Zarr.get_numChunksPerShard()*2;
            shardFooter = (uint64_t*)malloc(shardFooterSize*sizeof(uint64_t));
        }
        uint64_t lastF = 0;
        bool unWritten = true;
        std::vector<uint64_t> boxLo(nDims), boxHi(nDims), boxExt(nDims), inArray(nDims);
        const std::vector<uint64_t> zeros(nDims, 0);
        for(int64_t f = w*batchSize; f < (w+1)*batchSize; f++){
            if(f>=Zarr.get_numChunks()  || err) break;
            lastF = f;

            if(Zarr.get_shard()){
                unWritten = true;
                currChunk++;
                std::vector<uint64_t> pAV = Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f));
                bool pad = false;
                for(uint64_t d = 0; d < nDims; d++) pad = pad || pAV[d] > endCoords[d]/Zarr.get_chunk_shape(d);
                
                if(currChunk == Zarr.get_numChunksPerShard() || pad){
                    if(pad){
                        shardFooter[currChunk*2] = std::numeric_limits<uint64_t>::max();
                        shardFooter[(currChunk*2)+1] = std::numeric_limits<uint64_t>::max();

                        // Edge case for when we can't write because padded chunks are in the middle
                        if(currChunk != Zarr.get_numChunksPerShard()){
                            continue;
                        }
                    }
                    unWritten = false;
                    
                    // calculate CRC32C
                    uint32_t shardFooterCRC32C = crc32c(reinterpret_cast<uint8_t*>(shardFooter), shardFooterSize*sizeof(uint64_t));
                    // Not sure how sharding interacts with the subfolders at the moment (cAV needs to be converted)
                    const std::string subfolderName = Zarr.get_subfoldersString(cAV);
                    std::string fileName(Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.chunkNameToShardName(Zarr.get_chunkNames(f-1)));
                    std::string fileNameFinal;
                    if(useUuid){
                        fileNameFinal = std::string(fileName);
                        fileName.append(uuid);
                    }
                    std::ofstream file(fileName, std::ios::binary | std::ios::app);

                    if(!file.is_open()){
                            #pragma omp critical
                            {
                                err = 1;
                                errString = "Check permissions or filepath. Cannot write to path: "+
                                    fileName+"\n";
                            }
                            break;
                    }
                    file.write(reinterpret_cast<char*>(shardFooter),shardFooterSize*sizeof(uint64_t));
                    file.write(reinterpret_cast<char*>(&shardFooterCRC32C),sizeof(uint32_t));
                    file.close();
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
                    /*
                    if(pad){
                        f += (Zarr.get_numChunksPerShard()-currChunk-1);
                        currChunk = -1;
                        continue;
                    }
                    */
                    currChunk = 0;
                }
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
                cRegion = parallelReadZarrWriteWrapper(Zarr, crop, cStart, cEnd);
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

            if(sparse){
                const bool allZeros = memcmp(zeroChunkUnc,chunkUnC,sB);
                if(!allZeros){
                    if(Zarr.get_shard()){
                        shardFooter[(currChunk*2)] = std::numeric_limits<uint64_t>::max();
                        shardFooter[(currChunk*2)+1] = std::numeric_limits<uint64_t>::max();
                    } 
                    free(cRegion);
                    cRegion = nullptr;
                    continue;
                }

            }

            // Use the same blosc compress as Zarr
            const std::string subfolderName = Zarr.get_subfoldersString(cAV);
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
                    std::ofstream file(fileName, std::ios::binary | std::ios::trunc);
    
                    if(!file.is_open()){
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
            // Sharding
            else{
                std::string fileName(Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.chunkNameToShardName(Zarr.get_chunkNames(f)));
                if(useUuid){    
                    fileName.append(uuid);
                }
                std::ofstream file;

                if(currChunk > 0) {
                    int64_t currInd = currChunk;
                    while(currInd >= 0 && (shardFooter[((currInd-1)*2)] == std::numeric_limits<uint64_t>::max() &&
                       shardFooter[((currInd-1)*2)+1] == std::numeric_limits<uint64_t>::max())){
                       currInd--;
                    }
                    uint64_t shardOffset = 0; 
                    if(currInd){
                        shardOffset = shardFooter[((currInd-1)*2)]+
                                      shardFooter[((currInd-1)*2)+1];
                    }
                    shardFooter[(currChunk*2)] = shardOffset;
                    shardFooter[(currChunk*2)+1] = csize;
                    file = std::ofstream(fileName, std::ios::binary | std::ios::app);

                }
                else{
                    shardFooter[0] = 0;
                    shardFooter[1] = csize;
                    file = std::ofstream(fileName, std::ios::binary | std::ios::trunc);
                }
                if(!file.is_open()){
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
            
            }
            free(cRegion);
            cRegion = nullptr;
        }

        if(Zarr.get_shard() && unWritten && (((uint64_t)((w)*batchSize)))<(Zarr.get_numChunks())){
            // calculate CRC32C
            uint32_t shardFooterCRC32C = crc32c(reinterpret_cast<uint8_t*>(shardFooter), shardFooterSize*sizeof(uint64_t));
            uint64_t f = lastF;
            std::vector<uint64_t> pAV = Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f));

            bool pad = false;
            for(uint64_t d = 0; d < nDims; d++) pad = pad || pAV[d] > endCoords[d]/Zarr.get_chunk_shape(d);

            if(pad){
                for(uint64_t i = currChunk; i < Zarr.get_numChunksPerShard(); i++){
                    shardFooter[i*2] = std::numeric_limits<uint64_t>::max();
                    shardFooter[(i*2)+1] = std::numeric_limits<uint64_t>::max();
                }
            }
            const std::string subfolderName = Zarr.get_subfoldersString(cAV);
            std::string fileName(Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.chunkNameToShardName(Zarr.get_chunkNames(f)));
            std::string fileNameFinal;
            if(useUuid){
                fileNameFinal = std::string(fileName);
                fileName.append(uuid);
            }
            
            std::ofstream file(fileName, std::ios::binary | std::ios::app);

            if(!file.is_open()){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Check permissions or filepath. Cannot write to path: "+
                            fileName+"\n";
                    }
                    continue;
            }
            file.write(reinterpret_cast<char*>(shardFooter),shardFooterSize*sizeof(uint64_t));
            file.write(reinterpret_cast<char*>(&shardFooterCRC32C),sizeof(uint32_t));
            file.close();
            if(useUuid && !renameReplace(fileName, fileNameFinal)){
                remove(fileName.c_str());
                #pragma omp critical
                {
                    err = 1;
                    errString = "Cannot move the temporary file into place: "+
                        fileNameFinal+"\n";
                }
            }
        }
        free(shardFooter);
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
