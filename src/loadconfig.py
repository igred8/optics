import yaml

'''
    ivan.gadjev@raytheon.com
    author 2022-12-08

        

'''

def load_run_args_from_file( filename ):
    '''
    Load a YAML file specifying the run parameters.
    '''
    with open( filename, mode='r' ) as file:
        args = yaml.safe_load( file ) # DO NOT USE yaml.load()!!! IT IS NOT SECURE!

    print(' Successfully loaded run parameters from file:\n' + filename)

    return args
