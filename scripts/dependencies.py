import importlib

DEPENDENCIES = ['numpy', 'pandas', 'allel', 'h5py', 'configparser', 'scipy.optimize', 'progressbar']


def check_dependencies():
    for dep in DEPENDENCIES:
        try:
            importlib.import_module(dep)
        except ImportError:
            print('\n\n\n **** ERROR **** \n')
            print('Could not find the dependency %s, please check that you have '
                  'followed the installation instructions in the manual '
                  ' and try again.' % (dep))
            print('\n *************** \n\n\n')

            raise ImportError