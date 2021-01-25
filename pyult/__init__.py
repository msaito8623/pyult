"""
pyult - Preprocessing for ultrasound images in Python
=====================================================

*pyult* helps to preprocess ultrasound files exported by Articulate Assistant Advanced (AAA), an ultrasound-recording software. It can produce dataframes, images, and videos from ultrasound files.

"""

import os
import sys
import multiprocessing as mp
from pip._vendor import pkg_resources

__author__ = 'Motoki Saito'
__author_email__ = 'motoki.saito@uni-tuebingen.de'
__version__ = '0.1.1'
__license__ = 'MIT'
__description__ = 'Preprocessing ultrasound images from AAA in Python'
__classifiers__ = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering'
]


def sysinfo():
    """
    Prints the system information
    """
    pyult = pkg_resources.working_set.by_key['pyult']
    dependencies = [ i.project_name for i in pyult.requires() ]

    header = ('Pyult Information\n'
              '=================\n\n')

    general = ('General Information\n'
               '-------------------\n'
               'Python version: {}\n'
               'Pyult version: {}\n\n').format(sys.version.split()[0], __version__)

    uname = os.uname()
    osinfo = ('Operating System\n'
              '----------------\n'
              'OS: {s.sysname} {s.machine}\n'
              'Kernel: {s.release}\n'
              'CPU: {cpu_count}\n').format(s=uname, cpu_count=mp.cpu_count())
    osinfo += '\n'

    renames = {'opencv-python':'cv2', 'scikit-learn':'sklearn'}
    excludes = ['python-dateutil', 'threadpoolctl']
    dependencies = [ renames[i] if i in renames.keys() else i for i in dependencies ]
    dependencies = [ i for i in dependencies if not i in excludes ]
    deps = ('Dependencies\n'
            '------------\n')
    deps += '\n'.join( '{pkg.__name__}: {pkg.__version__}'.format(pkg=__import__(dep)) for dep in dependencies )

    print(header + general + osinfo + deps)
    return None
