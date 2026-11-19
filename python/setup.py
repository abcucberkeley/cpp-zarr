import os
import platform
from setuptools import setup, find_packages, Extension
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sysconfig
import pybind11
import subprocess
import shutil
import tempfile

# Determine OS and shared libraries
system = platform.system()
shared_libraries = []
if system == "Linux":
    shared_libraries.append("*.so*")
elif system == "Windows":
    shared_libraries.append("*.dll")
elif system == "Darwin":
    shared_libraries.append("*.dylib")


def prefix_map_flags():
    # The module names no folder of the machine that built it, as the library does not (CMakeLists.txt): __FILE__ in
    # pybind11's and Python's headers, and the debug information Python's own flags (-g) ask for, had put such folders
    # in the 1.7.3 wheels' modules: C:\Users\<user>\AppData\Local\Temp\build-env-<id>\Lib\site-packages\pybind11\include/...
    # and C:\Users\<user>\AppData\Local\pypa\cibuildwheel\Cache\nuget-cpython\python.<version>\tools\include/... in the
    # Windows ones (13 in the cp312 module), /tmp/build-env-<id>/... in the Linux ones. GCC takes the last map that
    # matches, so each folder comes after the folders that hold it (pybind11's is in the temp folder in cibuildwheel's
    # builds). Each is given as named and as its real path (a Windows temp folder may be named in its short 8.3 form,
    # macOS's is under /private), and on Windows with \ and with / (GCC there takes them as the same in a map; both are
    # harmless). The same maps as cpp-tiff's setup.py gives its module; .github/scripts/module_prefix_maps.py gives the
    # CI's build of the module these maps, from this function.
    paths = sysconfig.get_paths()
    folders = [(tempfile.gettempdir(), "tmp"),
               (paths["include"], "python-include"),
               (paths["platinclude"], "python-platinclude"),
               (os.path.dirname(os.path.abspath(__file__)), "."),
               (pybind11.get_include(), "pybind11-include")]
    maps = []
    for folder, name in folders:
        for form in (folder, os.path.realpath(folder)):
            form = form.rstrip("/\\")
            for f in ((form, form.replace("\\", "/")) if system == "Windows" else (form,)):
                if f and f not in [m[0] for m in maps]:
                    maps.append((f, name))
    # Shorter folders first: a folder that holds another is shorter than it
    maps.sort(key=lambda m: len(m[0]))
    return [f"-ffile-prefix-map={f}={name}" for f, name in maps]


class PrefixMapBuildExt(build_ext):
    # (on Windows the library is built with MinGW-w64 and its API passes C++ types, so the module is compiled with
    # that MinGW's gcc and g++, on PATH, as the released wheels' modules were, not with MSVC, setuptools' default
    # there, which could not link the library either; an option or config file that names a compiler still wins)
    def initialize_options(self):
        super().initialize_options()
        if system == "Windows":
            self.compiler = "mingw32"

    # The maps are GCC's and Clang's options; MSVC, which an sdist may be built with, does not take them
    def build_extensions(self):
        if self.compiler.compiler_type != "msvc":
            flags = prefix_map_flags()
            for ext in self.extensions:
                ext.extra_compile_args = list(ext.extra_compile_args or []) + flags
        super().build_extensions()


class BuildExt(PrefixMapBuildExt):
    def run(self):
        # Get the directory of the current file (setup.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define paths relative to the current directory
        cmake_source_dir = os.path.join(current_dir)
        cmake_build_dir = os.path.join(current_dir, "build")
        install_dir = os.path.join(cmake_build_dir, "install")

        # Build and install CMake-based dependency
        os.makedirs(cmake_build_dir, exist_ok=True)

        # Run cmake configuration and build commands
        cmake_cmd = [
            "cmake",
            cmake_source_dir,
            "-B", cmake_build_dir,
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            # (the library's C++ runtime kept private, as the module's is: see CMakeLists.txt)
            "-DCPPZARR_PRIVATE_RUNTIME=ON",
        ]

        # Configure and build CMake project
        try:
            subprocess.check_call(cmake_cmd)
            subprocess.check_call(["cmake", "--build", cmake_build_dir, "--parallel", "--target", "install"])
        except subprocess.CalledProcessError as e:
            print("Error during CMake build and install:", e)
            raise

        # Run the regular build_ext to compile the Python extension
        super().run()


ext_modules = [
    Extension(
        "cppzarr.cppzarr",
        include_dirs=['build/install/include',pybind11.get_include()],
        library_dirs=['build/install/lib64','build/install/lib'],
        libraries=['cppZarr'],
        sources=["cppzarr.cpp"],
        # (the static C++ runtime kept private, as in the library: see CMakeLists.txt. On macOS the module exports
        # its entry point alone: the loader merges the exported weak definitions of one name across every image a
        # process loads, so that with cpp-tiff's module loaded first, this module's std::call_once in pybind11 ran
        # cpp-tiff's copy of its code, on cpp-tiff's runtime's state, and the first write crashed. On macOS it is linked
        # without a debug map (-S), which names the build's object files by their absolute paths (the 1.7.3 wheels'
        # modules named /Users/<user>/.../python/build/temp.macosx-.../cppzarr.o, and the static C++ runtime's objects
        # in Homebrew's libstdc++.a), and is of no use once those files are gone, as cpp-tiff's module is linked)
        extra_link_args=['-static-libstdc++'] + (['-Wl,--exclude-libs,ALL'] if system == "Linux" else
                                                 ['-Wl,-exported_symbol,_PyInit_cppzarr', '-Wl,-S'] if system == "Darwin" else [])
    ),
]


# Packaging for PyPI
setup(
    name="cpp-zarr",
    version="1.8.0",
    description="Python wrappers for cpp-zarr",
    url='https://github.com/abcucberkeley/cpp-zarr',
    author='Matthew Mueller',
    author_email='matthewmueller@berkeley.edu',
    # (the license as an SPDX expression, which setuptools 77 and later write as License-Expression: given as license=,
    # setuptools 84 writes the older License field instead. And the license files, which the wheel carries in its
    # .dist-info/licenses folder with their paths: cpp-zarr's own, and those of the works built into the library and the
    # module or bundled in the wheels, which THIRD_PARTY_NOTICES.md lists. They are copies of the repository's top-level
    # files, kept here because setuptools takes no file outside the project's folder; CI checks that they match)
    license_expression="BSD-3-Clause",
    license_files=["LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/**/*"],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt} if system=="Linux" else {"build_ext": PrefixMapBuildExt},
    install_requires=['numpy>=1.20'],
    # (reads into GPU memory: nvCOMP, which the library loads when it first reads on a GPU)
    extras_require={'gpu': ['nvidia-libnvcomp-cu12; platform_system != "Darwin"']},
    package_dir={'cpp-zarr': 'cpp-zarr'},
    package_data={"": shared_libraries} if system!="Linux" else {},
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.9, <3.15',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
