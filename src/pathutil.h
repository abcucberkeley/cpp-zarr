#ifndef CPPZARR_PATHUTIL_H
#define CPPZARR_PATHUTIL_H
#include <cstdio>
#include <fstream>
#include <string>

// Opening files by their path, inside the library. Paths are UTF-8 on every platform. On Windows they are opened
// with Windows' wide-character functions, so that names in any script open: its narrow functions read a path's
// bytes in the ANSI code page, so that an array in a folder named "café µm" or "日本" could not be read or written
// from Python, Java or C++, which pass UTF-8. A path whose bytes are not valid UTF-8 is read in the ANSI code page,
// as callers that passed such paths before had them read (MATLAB's mxArrayToString, a narrow argv). cpp-tiff reads
// paths by the same rule. (Not installed: the library's functions take paths as std::string or char*, by this rule.)
#ifdef _WIN32
// The path in UTF-16, by that rule (empty for an empty path)
std::wstring widePath(const std::string &path);
#endif
void openPath(std::ifstream &f, const std::string &path, std::ios::openmode mode = std::ios::in);
void openPath(std::ofstream &f, const std::string &path, std::ios::openmode mode = std::ios::out);
FILE* fopenPath(const std::string &path, const char* mode);
// As remove and mkdir (0, or -1 with errno set)
int removePath(const std::string &path);
int mkdirPath(const std::string &path);

#endif
