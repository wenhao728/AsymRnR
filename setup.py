from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


if __name__ == '__main__':
    setup(
        name='square_dist',
        ext_modules=[
            CppExtension(
                name='square_dist', 
                sources=['square_dist.cpp']),
        ],
        cmdclass={'build_ext': BuildExtension},
        version='0.0.1',
    )
