#ifndef CPPZARR_HELPERFUNCTIONS_H
#define CPPZARR_HELPERFUNCTIONS_H
#include <string>

#ifndef _WIN32
const char* expandTilde(const char* path);
#endif
    
#ifdef _WIN32
char* strndup (const char *s, size_t n);

int _vscprintf_so(const char * format, va_list pargs);

int vasprintf(char **strp, const char *fmt, va_list ap);

int asprintf(char *strp[], const char *fmt, ...);
#endif

std::string generateUUID();

void mkdirRecursive(const char *dir);

bool fileExists(const std::string &fileName);

void makeDimensionFolders(const std::string &fileName);

bool isLittleEndian();

bool oppositeEndianness(const std::string &dtype);

// Parse a zarr fill_value string for memset-style fills. Returns 0 for anything
// unparsable or outside int range (e.g. the int64 min/max strings that
// Infinity/-Infinity fill values normalize to), instead of throwing like stoi.
// Note: memset-based fills are only exact for 0 (and -1) on multi-byte dtypes;
// any other nonzero fill repeats its low byte across the element.
int fillValueToInt(const std::string &fillValue);

void swapArrayEndianness(void* array, const size_t elementSize, const size_t numElements);
#endif
