import sys
from distutils.core import setup
from setuptools.extension import Extension
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed.")
    sys.exit(1)

# using command "python ./setup.py build_ext --inplace"
# to compile ".py" to ".so"
directives = {"language_level": 3, 'embedsignature': True}
setup(
    name="cv_research_loader",
    ext_modules = cythonize(Extension("run_docker", ["run_docker.py"]),
                          compiler_directives = directives),
    cmdclass = {'build_ext': build_ext},
    python_requires=">=3.6",
)
