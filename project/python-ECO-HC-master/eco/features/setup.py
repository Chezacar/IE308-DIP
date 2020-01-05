#用于初始化有关的环境变量 由于源码中fhog调用了matlab的函数库，并且论文中也没写出fhog如何求取，所以我们参考了https://github.com/lawpdas/fhog-python/tree/master/python3/fhog的fhog提取代码
from distutils.core import setup, Extension
from numpy.distutils import misc_util

c_ext = Extension("_gradient", ["_gradient.cpp", "gradient.cpp"])
setup(
    ext_modules=[c_ext],
    include_dirs = misc_util.get_numpy_include_dirs(),
)
