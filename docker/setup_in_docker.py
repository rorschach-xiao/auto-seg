import sys, os
from functools import reduce
from distutils.core import setup
from setuptools.extension import Extension
from setuptools import find_packages
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed.")
    sys.exit(1)

exclude_path = ["scripts", "experiments", "start_server.py"]

def in_excluded(path):
    for ex_path in exclude_path:
        if ex_path in path:
            return True
    return False

def scandir(directory, files=[]):
    if directory != "." and "." in directory and not directory.endswith(".py"):
        directory = directory.replace(".", os.path.sep)

    for f in os.listdir(directory):
        if directory != ".":
            path = os.path.join(directory, f)
        else:
            path = f

        if os.path.isfile(path) \
                and path.endswith(".py") \
                and not in_excluded(path):
            files.append(path.replace(os.path.sep, ".")[:-3])
        elif os.path.isdir(path) and not in_excluded(path):
            scandir(path, files)

    return files

def make_extension(ext_names):
    ext_path = ext_names.replace(".", os.path.sep) + ".py"
    return Extension(
        ext_names,
        [ext_path],
        include_dirs = ["."],   # adding the '.' to include_dirs is CRUCIAL!!
        )


packages = find_packages("auto_seg", exclude = ["scripts", "experiments"])
packages = list("auto_seg." + p for p in packages)
packages.append("auto_seg")

ext_names = reduce(lambda l1, l2: l1 + l2, [scandir(d) for d in packages])

ext_modules = [make_extension(name) for name in ext_names]

# using command "python ./setup.py build_ext --inplace"
# to compile ".py" to ".so"
directives = {"language_level": 3}

setup(name="auto_seg",
        version="0.0.1",
        description="auto_seg",
        author="AutoCV team",
        python_requires=">=3.6",
        packages = packages,
        ext_modules=cythonize(ext_modules, compiler_directives=directives),
        )
