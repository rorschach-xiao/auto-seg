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

exclude_path = ['__init__.py', 'contrib', 'deprecated']

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


packages = find_packages("autocv_classification_pytorch", exclude = ["contrib", "deprecated"])
packages = list("autocv_classification_pytorch." + p for p in packages)
packages.append("autocv_classification_pytorch")

ext_names = reduce(lambda l1, l2: l1 + l2, [scandir(d) for d in packages])

ext_modules = [make_extension(name) for name in ext_names]
# ext_modules.append('autocv.py')
# print(ext_names)

# using command "python ./setup.py build_ext --inplace"
# to compile ".py" to ".so"
directives = {"language_level": 3}
# setup(
#   name="cv_research",
#   packages = find_packages(),
#
#   ext_modules = cythonize(ext_modules, compiler_directives = directives),
#   cmdclass = {'build_ext': build_ext},
# )

setup(name="autocv_classification_pytorch",
        version="0.0.1",
        description="autocv_classification_pytorch",
        author="AutoCV team",
        install_requires=[
            "nni",
            "albumentations",
            "pycallgraph",
            "gevent",
        ],
        python_requires=">=3.6",
        packages = packages,
        ext_modules=cythonize(ext_modules, compiler_directives=directives),
        )
