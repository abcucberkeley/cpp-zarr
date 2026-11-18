#!/usr/bin/env python3
# No ELF file in a release archive needs a symbol version newer than the oldest runtime it must load with. Every
# member of every archive (.whl, .zip, .tar, .tar.gz, .tgz, read by their contents, and archives inside them, as
# check_release_paths.py reads them) that is an ELF file (by its magic, whatever its name) has its version needs
# (.gnu.version_r, SHT_GNU_verneed) read, 32- or 64-bit, either byte order, and every needed version whose prefix
# (the name up to its last '_', as GLIBCXX in GLIBCXX_3.4.21) is one given with --max fails if it is newer than that
# --max, compared number by number (3.4.9 < 3.4.21 < 3.4.28).
# Why: a Linux MATLAB mex file that needs a libstdc++ at all loads MATLAB's own libstdc++.so.6, not one shipped
# next to it, since MATLAB puts its own first. The mex files link libstdc++ statically and need none (their LINKLIBS
# leave out the -lstdc++ that MATLAB's mex links by default, where -static-libstdc++ takes no effect); before that
# they needed GLIBCXX_3.4.29. Several MATLAB releases still supported ship a libstdc++ that provides at most
# GLIBCXX_3.4.28 and CXXABI_1.3.12, so a mex file that needs a newer version (from a newer compiler or a newer C++
# library feature) does not load in them. A needed version of a --max prefix that is no
# number (as CXXABI_FLOAT128) is shown as a note.
# usage: check_release_abi.py --max GLIBCXX_3.4.28 --max CXXABI_1.3.12 <archive or folder> ...  (a folder means every
#        *.whl, *.zip, *.tar, *.tar.gz and *.tgz in it, not in its subfolders)
# exit: 0 if all is clean, 1 on any finding (or an ELF member whose version needs cannot be read), 2 on bad
#       arguments or a missing, unreadable or non-archive input
import argparse
import os
import re
import struct
import sys
import tarfile
import zipfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_release_paths  # noqa: E402

SHT_GNU_VERNEED = 0x6ffffffe
_VERSION = re.compile(r"^(.+)_([0-9]+(?:\.[0-9]+)*)$")


def split_version(name):
    """(prefix, (numbers...)) for a version name such as GLIBCXX_3.4.21; (prefix, None) when what follows the last
    '_' is no dotted number; (None, None) when there is no '_'."""
    m = _VERSION.match(name)
    if m:
        return m.group(1), tuple(int(n) for n in m.group(2).split("."))
    if "_" in name:
        return name.rsplit("_", 1)[0], None
    return None, None


def _cstr(data, off):
    if off < 0 or off >= len(data):
        raise ValueError("string offset %d out of range" % off)
    nul = data.find(b"\0", off)
    if nul < 0:
        raise ValueError("unterminated string at %d" % off)
    return data[off:nul].decode("latin-1")


def elf_version_needs(data):
    """Return [(file, version)] of an ELF file's version needs, in file order; raise ValueError on a malformed file.
    A file with no .gnu.version_r section needs none."""
    if len(data) < 52 or data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    if data[4] not in (1, 2) or data[5] not in (1, 2):
        raise ValueError("unknown ELF class %d or data encoding %d" % (data[4], data[5]))
    is64 = data[4] == 2
    e = "<" if data[5] == 1 else ">"
    try:
        if is64:
            shoff, = struct.unpack_from(e + "Q", data, 0x28)
            shentsize, shnum = struct.unpack_from(e + "HH", data, 0x3a)
            shfmt = e + "IIQQQQIIQQ"  # name type flags addr offset size link info addralign entsize
        else:
            shoff, = struct.unpack_from(e + "I", data, 0x20)
            shentsize, shnum = struct.unpack_from(e + "HH", data, 0x2e)
            shfmt = e + "IIIIIIIIII"
        if shoff == 0:
            raise ValueError("no section headers")
        if shnum == 0:  # more than 0xff00 sections: the count is section 0's sh_size
            shnum = struct.unpack_from(shfmt, data, shoff)[5]
        if shentsize < struct.calcsize(shfmt) or shoff + shnum * shentsize > len(data):
            raise ValueError("section headers out of range")
        sections = [struct.unpack_from(shfmt, data, shoff + i * shentsize) for i in range(shnum)]
        needs = []
        for sec in sections:
            if sec[1] != SHT_GNU_VERNEED:
                continue
            offset, size, link, count = sec[4], sec[5], sec[6], sec[7]
            if link >= len(sections) or offset + size > len(data):
                raise ValueError(".gnu.version_r out of range")
            stroff = sections[link][4]
            pos, seen = offset, 0
            while seen < count:  # Elf_Verneed: version cnt file aux next (same in 32 and 64 bits)
                if pos + 16 > offset + size:
                    raise ValueError(".gnu.version_r entry out of range")
                _, cnt, vfile, aux, nxt = struct.unpack_from(e + "HHIII", data, pos)
                fname = _cstr(data, stroff + vfile)
                apos = pos + aux
                for _ in range(cnt):  # Elf_Vernaux: hash flags other name next
                    if apos + 16 > offset + size:
                        raise ValueError(".gnu.version_r aux entry out of range")
                    _, _, _, vname, anxt = struct.unpack_from(e + "IHHII", data, apos)
                    needs.append((fname, _cstr(data, stroff + vname)))
                    if anxt == 0:
                        break
                    apos += anxt
                seen += 1
                if nxt == 0:
                    break
                pos += nxt
        return needs
    except struct.error as ex:
        raise ValueError("truncated: %s" % ex)


def parse_max(values):
    """{prefix: (name, numbers)} from the --max arguments; raise ValueError on a bad or repeated one."""
    maxes = {}
    for v in values:
        prefix, nums = split_version(v)
        if nums is None:
            raise ValueError("bad --max %r: not of the form NAME_1.2.3" % v)
        if prefix in maxes:
            raise ValueError("--max %s given twice" % prefix)
        maxes[prefix] = (v, nums)
    return maxes


def check_archive(path, maxes):
    """Print the findings of one archive and its summary line; return its number of findings."""
    findings, elves, notes = 0, 0, []
    highest = {}  # prefix -> (numbers, name)
    odd = set()
    with open(path, "rb") as f:
        for name, what, data in check_release_paths.archive_entries(f, ""):
            if what == "note":
                notes.append("%s: %s" % (name, data.decode()))
                continue
            if what != "file" or data[:4] != b"\x7fELF":
                continue
            elves += 1
            try:
                needs = elf_version_needs(data)
            except ValueError as ex:
                findings += 1
                print("%s: %s: version needs cannot be read: %s" % (path, name, ex))
                continue
            for fname, ver in needs:
                prefix, nums = split_version(ver)
                if prefix not in maxes:
                    continue
                if nums is None:
                    if (name, ver) not in odd:
                        odd.add((name, ver))
                        notes.append("%s: needs %s from %s, which is no version number" % (name, ver, fname))
                    continue
                if prefix not in highest or nums > highest[prefix][0]:
                    highest[prefix] = (nums, ver)
                maxname, maxnums = maxes[prefix]
                if nums > maxnums:
                    findings += 1
                    print("%s: %s: needs %s from %s, newer than %s" % (path, name, ver, fname, maxname))
    if elves == 0:
        notes.append("no ELF members")
    for n in notes:
        print("%s: note: %s" % (path, n))
    high = ", ".join(highest[p][1] if p in highest else "%s none" % p for p in maxes)
    print("%s: %s, %d ELF members, highest needs: %s" % (path, "FAIL" if findings else "ok", elves, high))
    return findings


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fail if an ELF file in a release archive needs a symbol version newer than a --max.")
    ap.add_argument("--max", action="append", required=True, metavar="NAME_VERSION",
                    help="the newest version allowed for its prefix, such as GLIBCXX_3.4.28 (repeatable)")
    ap.add_argument("paths", nargs="+", metavar="archive or folder")
    args = ap.parse_args(argv)  # exits 2 on bad arguments
    try:
        maxes = parse_max(args.max)
    except ValueError as ex:
        print(ex, file=sys.stderr)
        return 2

    archives = check_release_paths.expand_paths(args.paths)
    if archives is None:
        return 2

    bad = 0
    for a in archives:
        try:
            bad += check_archive(a, maxes) > 0
        except (OSError, EOFError, zlib.error, zipfile.BadZipFile, tarfile.TarError) as ex:
            print("%s: cannot read: %s" % (a, ex), file=sys.stderr)
            return 2
    print("%d archives, %d with findings" % (len(archives), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
