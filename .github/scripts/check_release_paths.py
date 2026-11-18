#!/usr/bin/env python3
# No release archive names a folder of the machine that built it. Every wheel and every MATLAB release archive
# (.whl, .zip, .tar, .tar.gz, .tgz; whether one is a zip or a tar file is decided by its contents, so a tar file named
# .zip is read too, with a note) is opened and every regular member (modules, libraries bundled with them, mex files, RECORD,
# METADATA, delvewheel's DELVEWHEEL record, .py and .m files and the rest) is searched as bytes, with every member's
# own name (as /<name>, so that home/<user>/x is a home folder too), link target, owner names and pax header values as
# text, for:
#   - a user's home folder (C:\Users\<user>\, /home/<user>/, /<root>/home/<user>/, /Users/<user>/);
#   - a temporary build environment (/tmp/build-env-..., ...\Temp\build-env-..., /tmp/pip-...) and cibuildwheel's own
#     folders (cibw-run-..., pypa\cibuildwheel\Cache);
#   - a Spack build stage (spack-stage-..., /spack-temporary/, /spack-stage/), a per-user stage folder under /tmp,
#     /var/tmp or /local (/tmp/.../stage/<user>/), Bazel's per-user output root (_bazel_<user>/), and macOS's per-user
#     temporary folders (/private/var/folders/..., /var/folders/xx/<id>/);
#   - any pattern given with --forbid (a build machine's or account's name).
# The released 1.7.3 Windows wheels' modules held 11-13 such strings each, of the form
# C:\Users\<user>\AppData\Local\Temp\build-env-xxxx\Lib\site-packages\pybind11\include/... and
# C:\Users\<user>\AppData\Local\pypa\cibuildwheel\Cache\..., from __FILE__ in pybind11's and Python's headers, and the
# 1.7.3 Linux modules held /tmp/build-env-xxxxxxxx/... in their debug information. The libgomp.so.1 that the 1.8.0
# Linux MATLAB archives ship, built by a Spack GCC, holds its assertions' file names, of the form
# /local/spack-temporary/gcc/stage/<username>/spack-stage-gcc-13.2.0-<hash>/spack-src/libgomp/task.c. The neutral
# folders the builds now use on purpose (C:\cibw-tmp, C:\mex-tmp, relative names such as tmp/..., python-include,
# pybind11-include), a bare /tmp or /tmp/<name> and a bare /var/folders, which system libraries hold, are no
# failure. Compressed ELF sections (SHF_COMPRESSED, .zdebug_*) are searched after decompression, so compressed debug
# information can hide nothing, and an archive inside an archive (by its name) is opened and searched the same way.
# A tar member's owner or group name, other than root, is a finding too: the release's tar files are made with
# --numeric-owner, which stores none (the build account's name had been stored with every member of the 1.8.0 Linux and
# Mac archives' first builds); a numeric owner is searched by --forbid patterns only.
# usage: check_release_paths.py [--forbid REGEX ...] <archive or folder> ...  (a folder means every *.whl, *.zip,
#        *.tar, *.tar.gz and *.tgz in it, not in its subfolders)
# exit: 0 if all is clean, 1 on any finding, 2 on bad arguments or a missing or unreadable archive
import argparse
import io
import os
import re
import struct
import sys
import tarfile
import zipfile
import zlib

# A name in a path: no separator, no NUL or control byte, none of Windows' reserved characters (so a placeholder
# such as <user> is no name). Windows names may hold spaces, POSIX ones here not (names in strings end at a space).
_WIN_NAME = rb'[^\\/\x00-\x1f"<>|:*?]+'
_POSIX_NAME = rb'[A-Za-z0-9_.][A-Za-z0-9_.-]*'
# A user's folder name, but not a placeholder such as /Users/username/ in a comment that tells where a file lives
_NOT_PLACEHOLDER = rb'(?!(?i:user|username|yourname|your_?username|name)[\\/])'
_WIN_USER = _NOT_PLACEHOLDER + _WIN_NAME
_POSIX_USER = _NOT_PLACEHOLDER + _POSIX_NAME
# The start of an absolute POSIX path, not the middle of one (/usr/local/ is no /local/)
_ROOT = rb'(?<![A-Za-z0-9_.-])/'
PATTERNS = [
    ("Windows user folder", rb'(?i)[A-Z]:[\\/]+Users[\\/]+' + _WIN_USER + rb'[\\/]'),
    ("POSIX home folder", _ROOT + rb'home/' + _POSIX_USER + rb'/'),
    # A site's home folders under another root (/<root>/home/<user>/, /<root>/home/groups/<group>/), as in a GCC
    # built by hand in a shared software folder
    ("POSIX home folder under a site root", _ROOT + _POSIX_NAME + rb'/home/' + _POSIX_USER + rb'/'),
    # Not after a drive letter's colon, which the Windows pattern reports
    ("macOS user folder", rb'(?<![A-Za-z0-9_.:-])/Users/' + _POSIX_USER + rb'/'),
    ("temporary build environment", rb'/tmp/+(?:build-env-|pip-)'),
    ("temporary build environment", rb'(?i)[\\/]Temp[\\/]+(?:build-env-|pip-)'),
    ("cibuildwheel temp folder", rb'cibw-run-'),
    ("cibuildwheel cache", rb'(?i)pypa[\\/]+cibuildwheel[\\/]+Cache'),
    # Spack's build stage folders (spack-stage-<package>-<version>-<hash>) and its stage roots, which name the user
    # who built: /local/spack-temporary/<compiler>/stage/<username>/spack-stage-..., $TMP/<username>/spack-stage/...
    ("Spack build stage", rb'spack-stage-'),
    ("Spack build stage", rb'[\\/]spack-(?:temporary|stage)[\\/]'),
    # A user's stage folder under a temporary or node-local root: /tmp/stage/<user>/, /local/<a>/<b>/stage/<user>/
    ("per-user stage folder", _ROOT + rb'(?:tmp|var/tmp|local)/+(?:' + _POSIX_NAME + rb'/+){0,3}?'
     rb'(?:stage|stages|staging|build-stage)/+' + _POSIX_USER + rb'/'),
    # Bazel's output root: ~/.cache/bazel/_bazel_<user>/ or /tmp/_bazel_<user>/
    ("Bazel per-user output root", rb'/_bazel_' + _POSIX_USER + rb'/'),
    # macOS's per-user temporary folders ($TMPDIR): /private/var/folders/xx/<id>/T/ or /var/folders/xx/<id>/T/
    ("macOS per-user temporary folder", rb'/private/var/folders/'),
    ("macOS per-user temporary folder", rb'/var/folders/[A-Za-z0-9_+-]{2}/[A-Za-z0-9_+-]{2,}/'),
]
# Folder arguments: these names are read; what a file is (zip or tar) is decided by its contents
ARCHIVE_SUFFIXES = (".whl", ".zip", ".tar", ".tar.gz", ".tgz")
MAX_PER_MEMBER = 5
SHOW = 120


def _printable(c):
    return 0x20 <= c < 0x7f


def string_start(data, pos):
    """Where the printable run that holds pos starts (looking back at most 4096 bytes)."""
    lo = pos
    while lo > 0 and pos - lo < 4096 and _printable(data[lo - 1]):
        lo -= 1
    return lo


def printable_context(data, lo, start, end):
    """The printable run from lo (at most 40 bytes before the match), cut to SHOW characters."""
    lo = max(lo, start - 40)
    hi = end
    while hi < len(data) and hi - lo < SHOW and _printable(data[hi]):
        hi += 1
    more = hi < len(data) and _printable(data[hi])
    text = "".join(chr(c) if _printable(c) else "\\x%02x" % c for c in data[lo:hi])
    return text + ("..." if more else "")


def elf_compressed_sections(data):
    """Yield (name, bytes) for each compressed section of an ELF file, decompressed; nothing for other files."""
    if data[:4] != b"\x7fELF" or len(data) < 64:
        return
    is64 = data[4] == 2
    end = "<" if data[5] == 1 else ">"
    try:
        if is64:
            shoff, = struct.unpack_from(end + "Q", data, 0x28)
            shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x3a)
            fmt, chfmt = end + "IIQQQQIIQQ", end + "IIQQ"
        else:
            shoff, = struct.unpack_from(end + "I", data, 0x20)
            shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x2e)
            fmt, chfmt = end + "IIIIIIIIII", end + "III"
        sections = [struct.unpack_from(fmt, data, shoff + i * shentsize) for i in range(shnum)]
        stroff = sections[shstrndx][4]
    except (struct.error, IndexError):
        return
    for sec in sections:
        name_off, flags, offset, size = sec[0], sec[2], sec[4], sec[5]
        nul = data.find(b"\0", stroff + name_off)
        name = data[stroff + name_off:nul].decode("latin-1", "replace")
        raw = data[offset:offset + size]
        try:
            if flags & 0x800:  # SHF_COMPRESSED: an Elf_Chdr, then the data
                ch_type = struct.unpack_from(chfmt, raw, 0)[0]
                body = raw[struct.calcsize(chfmt):]
                if ch_type == 1:  # ELFCOMPRESS_ZLIB
                    yield name, zlib.decompress(body)
                else:
                    yield name, None  # ELFCOMPRESS_ZSTD, which the standard library cannot read before 3.14
            elif name.startswith(".zdebug") and raw[:4] == b"ZLIB":
                yield name, zlib.decompress(raw[12:])
        except (struct.error, zlib.error):
            yield name, None


def scan_bytes(data, patterns):
    """Return the contexts of the first MAX_PER_MEMBER strings with matches and the number of such strings (a
    string that several patterns match, as C:\\Users\\<user>\\AppData\\Local\\Temp\\build-env-... does, is one
    finding)."""
    found = {}
    for label, rx in patterns:
        for m in rx.finditer(data):
            key = string_start(data, m.start())
            if key not in found:
                found[key] = (m.start(), m.end(), [label])
            else:
                start, end, labels = found[key]
                if label not in labels:
                    labels.append(label)
                if m.start() < start:  # show the string from its first match
                    found[key] = (m.start(), m.end(), labels)
    shown = []
    for key in sorted(found)[:MAX_PER_MEMBER]:
        start, end, labels = found[key]
        shown.append("%s [%s]" % (printable_context(data, key, start, end), ", ".join(labels)))
    return shown, len(found)

def _is_archive_name(name):
    return name.lower().endswith(ARCHIVE_SUFFIXES)


def _open_archive(fileobj):
    """Return ("zip", ZipFile) or ("tar", TarFile) for a seekable file object, by its contents; None for neither."""
    if zipfile.is_zipfile(fileobj):
        fileobj.seek(0)
        return "zip", zipfile.ZipFile(fileobj)
    fileobj.seek(0)
    try:
        return "tar", tarfile.open(fileobj=fileobj, mode="r:*")
    except tarfile.ReadError:
        return None


def archive_entries(fileobj, prefix, depth=0):
    """Yield (name, kind, data) for an archive: kind "text" for a member's name, link target and pax values (data is
    text bytes), "owner" for a tar member's owner names, "file" for a regular member's contents, and "note" for a
    remark (data is a message). An archive member (by its name) is opened too, its members named outer!inner."""
    opened = _open_archive(fileobj)
    if opened is None:
        raise tarfile.ReadError("neither a zip nor a tar file")
    kind, ar = opened
    with ar:
        if kind == "zip":
            for info in ar.infolist():
                name = prefix + info.filename
                yield name, "text", b"/" + info.filename.encode("utf-8", "surrogateescape")
                if info.comment:
                    yield name, "text", info.comment
                if info.is_dir():
                    continue
                yield from _member_data(name, ar.read(info), depth)
        else:
            for m in ar:
                name = prefix + m.name
                yield name, "text", b"/" + m.name.encode("utf-8", "surrogateescape")
                if m.linkname:
                    yield name, "text", m.linkname.encode("utf-8", "surrogateescape")
                for k, v in m.pax_headers.items():
                    if k not in ("path", "linkpath", "uname", "gname"):  # already searched
                        yield name, "text", ("%s=%s" % (k, v)).encode("utf-8", "surrogateescape")
                yield name, "owner", ("%s/%s" % (m.uname or m.uid, m.gname or m.gid)).encode("utf-8", "surrogateescape")
                if m.isfile():
                    yield from _member_data(name, ar.extractfile(m).read(), depth)


def _member_data(name, data, depth):
    yield name, "file", data
    if _is_archive_name(name) and depth < 3:
        try:
            yield from archive_entries(io.BytesIO(data), name + "!", depth + 1)
        except (OSError, EOFError, zlib.error, zipfile.BadZipFile, tarfile.TarError) as e:
            yield name, "note", ("could not be opened as an archive and was searched as bytes only: %s" % e).encode()


def check_archive(path, patterns, forbid):
    """Print the findings of one archive and its summary line; return its number of findings."""
    findings, members, notes, owners, named = 0, 0, [], set(), set()
    with open(path, "rb") as f:
        kind = _open_archive(f)
        if kind is None:
            raise tarfile.ReadError("neither a zip nor a tar file")
        kind[1].close()
        want = "zip" if path.lower().endswith((".whl", ".zip")) else "tar"
        if kind[0] != want:
            notes.append("is a %s file, not the %s file its name says" % (kind[0], want))
        f.seek(0)
        for name, what, data in archive_entries(f, ""):
            if what == "note":
                notes.append("%s: %s" % (name, data.decode()))
                continue
            if what == "owner":
                text = data.decode("utf-8", "replace")
                owners.add(text)
                if [n for n in text.split("/") if n and not n.isdigit() and n != "root"]:
                    # one finding for each owner, at its first member
                    if text not in named:
                        named.add(text)
                        findings += 1
                        print("%s: %s (owner): %s is named (tar --numeric-owner stores no names)" % (path, name, text))
                    continue
                parts = [(name + " (owner)", data)]
                pats = forbid
            elif what == "text":
                parts = [(name + " (name)", data)]
                pats = patterns
            else:
                members += 1
                parts = [(name, data)]
                for sec, body in elf_compressed_sections(data):
                    if body is None:
                        notes.append("%s: section %s could not be decompressed and was not searched" % (name, sec))
                    else:
                        parts.append(("%s[%s]" % (name, sec), body))
                pats = patterns
            for pname, body in parts:
                shown, total = scan_bytes(body, pats)
                findings += total
                for s in shown:
                    print("%s: %s: %s" % (path, pname, s))
                if total > len(shown):
                    print("%s: %s: ... and %d more" % (path, pname, total - len(shown)))
    owners -= {"0/0", "root/root", "root/0", "0/root"}
    if owners:
        notes.append("members are owned by (user/group) %s" % ", ".join(sorted(owners)))
    for n in notes:
        print("%s: note: %s" % (path, n))
    print("%s: %s, %d findings in %d members" % (path, "FAIL" if findings else "ok", findings, members))
    return findings


def expand_paths(paths, suffixes=ARCHIVE_SUFFIXES, noun="archives"):
    """The archives that the arguments name: a file as it is, a folder as its files (not its subfolders') whose names
    end with one of suffixes, sorted. On a missing path or a folder with no such file, print why and return None."""
    archives = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(os.path.join(p, f) for f in os.listdir(p)
                           if f.lower().endswith(suffixes) and os.path.isfile(os.path.join(p, f)))
            if not found:
                print("no %s in %s" % (noun, p), file=sys.stderr)
                return None
            archives += found
        elif os.path.isfile(p):
            archives.append(p)
        else:
            print("no archive or folder at %s" % p, file=sys.stderr)
            return None
    return archives


def main(argv=None, suffixes=ARCHIVE_SUFFIXES, noun="archives"):
    ap = argparse.ArgumentParser(description="Fail if any release archive names a build machine's folders.")
    ap.add_argument("--forbid", action="append", default=[], metavar="REGEX",
                    help="another pattern to fail on, such as a build machine's or account's name (repeatable)")
    ap.add_argument("paths", nargs="+", metavar="archive or folder")
    args = ap.parse_args(argv)  # exits 2 on bad arguments

    forbid = []
    for i, rx in enumerate(args.forbid):
        try:
            forbid.append(("--forbid %d" % (i + 1), re.compile(rx.encode("utf-8"))))
        except re.error as e:
            print("bad --forbid pattern %r: %s" % (rx, e), file=sys.stderr)
            return 2
    patterns = [(label, re.compile(rx)) for label, rx in PATTERNS] + forbid

    archives = expand_paths(args.paths, suffixes, noun)
    if archives is None:
        return 2

    bad = 0
    for a in archives:
        try:
            bad += check_archive(a, patterns, forbid) > 0
        except (OSError, EOFError, zlib.error, zipfile.BadZipFile, tarfile.TarError) as e:
            print("%s: cannot read: %s" % (a, e), file=sys.stderr)
            return 2
    print("%d %s, %d with findings" % (len(archives), noun, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
