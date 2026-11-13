#include "pathutil.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <direct.h>
#if defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE__WFOPEN)
#error "this libstdc++ has no streams opened by wide-character paths, which paths on Windows need"
#endif

std::wstring widePath(const std::string &path){
    if(path.empty()) return std::wstring();
    UINT codePage = CP_UTF8;
    DWORD flags = MB_ERR_INVALID_CHARS;
    int n = MultiByteToWideChar(codePage, flags, path.data(), (int)path.size(), NULL, 0);
    if(n <= 0){
        codePage = CP_ACP;
        flags = 0;
        n = MultiByteToWideChar(codePage, flags, path.data(), (int)path.size(), NULL, 0);
    }
    if(n <= 0) return std::wstring();
    std::wstring wide((size_t)n, L'\0');
    if(MultiByteToWideChar(codePage, flags, path.data(), (int)path.size(), &wide[0], n) != n) return std::wstring();
    return wide;
}

void openPath(std::ifstream &f, const std::string &path, std::ios::openmode mode){
    const std::wstring wide = widePath(path);
    if(wide.empty()) f.open(path, mode);
    else f.open(wide.c_str(), mode);
}

void openPath(std::ofstream &f, const std::string &path, std::ios::openmode mode){
    const std::wstring wide = widePath(path);
    if(wide.empty()) f.open(path, mode);
    else f.open(wide.c_str(), mode);
}

FILE* fopenPath(const std::string &path, const char* mode){
    const std::wstring wide = widePath(path);
    if(wide.empty()) return fopen(path.c_str(), mode);
    std::wstring wideMode;
    for(const char* m = mode; *m; m++) wideMode += (wchar_t)(unsigned char)*m;
    return _wfopen(wide.c_str(), wideMode.c_str());
}

int removePath(const std::string &path){
    const std::wstring wide = widePath(path);
    return wide.empty() ? remove(path.c_str()) : _wremove(wide.c_str());
}

int mkdirPath(const std::string &path){
    const std::wstring wide = widePath(path);
    return wide.empty() ? _mkdir(path.c_str()) : _wmkdir(wide.c_str());
}
#else
#include <sys/stat.h>
#include <sys/types.h>

void openPath(std::ifstream &f, const std::string &path, std::ios::openmode mode){
    f.open(path, mode);
}

void openPath(std::ofstream &f, const std::string &path, std::ios::openmode mode){
    f.open(path, mode);
}

FILE* fopenPath(const std::string &path, const char* mode){
    return fopen(path.c_str(), mode);
}

int removePath(const std::string &path){
    return remove(path.c_str());
}

int mkdirPath(const std::string &path){
    return mkdir(path.c_str(), 0777);
}
#endif
