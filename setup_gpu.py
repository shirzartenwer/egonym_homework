from setuptools import setup, Extension
import pybind11
import numpy
import sys

is_production = '--prod' in sys.argv
if '--prod' in sys.argv:
    sys.argv.remove('--prod')

if is_production:
    print("🚀 Building GPU module in PRODUCTION mode")
    compile_args = ["-O3","-std=c++11"]
    undef_macros = ["NDEBUG"]
else:
    print("🛠️ Building GPU module in DEVELOPMENT mode")
    compile_args = ["-O0", "-g", "-std=c++11", "-DDEBUG"]
    undef_macros = []

ext_modules = [
    Extension(
        "cpp_module_gpu_edge_detector",
        ["cpp_module/cpp_module_gpu.cpp"],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            "/usr/include/opencv4",
            "/usr/local/cuda/include"],
        language="c++",
        extra_compile_args=compile_args,
        libraries=["opencv_core", "opencv_imgproc", "opencv_cudaimgproc", "opencv_cudaarithm", "cudart"],
        library_dirs=["/usr/local/cuda/lib64"],
        undef_macros=undef_macros,
    ),
]

setup(
    name="cpp_module_gpu",
    version="0.1",
    ext_modules=ext_modules,
)