# Prints, one a line, the -ffile-prefix-map flags python/setup.py gives the Python module's compile (its
# prefix_map_flags(), taken from setup.py itself, which cannot be imported as it runs setup()), for the CI's
# build of the module, and with --folders the folders those maps are for, which the module must not name.
# usage: python module_prefix_maps.py <setup.py> [--folders]
import ast, os, platform, sys, sysconfig, tempfile
import pybind11

setup_py = os.path.abspath(sys.argv[1])
tree = ast.parse(open(setup_py).read(), setup_py)
func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prefix_map_flags")
scope = {"os": os, "sysconfig": sysconfig, "tempfile": tempfile, "pybind11": pybind11,
         "system": platform.system(), "__file__": setup_py}
exec(compile(ast.Module(body=[func], type_ignores=[]), setup_py, "exec"), scope)
flags = scope["prefix_map_flags"]()
if "--folders" in sys.argv[2:]:
    # The folder of setup.py is left out: the checkout is check_build_paths.sh's
    folders = []
    for flag in flags:
        folder = flag[len("-ffile-prefix-map="):].rsplit("=", 1)
        if folder[1] != "." and folder[0] not in folders:
            folders.append(folder[0])
    print("\n".join(folders))
else:
    print("\n".join(flags))
