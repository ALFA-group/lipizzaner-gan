"""
Author: Jamal Toutouh (toutouh@mit.edu) - www.jamal.es

This code is part of the research of our paper "Re-purposing Heterogeneous Generative Ensembles with Evolutionary
Computation" presented during GECCO 2020 (https://doi.org/10.1145/3377930.3390229)

create-generators-folder.py allows the creation of a folder that contains all the generators used to create the
ensemble of generators by copying them from a set of origin folders.
"""
import shutil
import sys
import argparse
import os
import glob


def main():
    parser = argparse.ArgumentParser(description='Create generators folder with consecutive names to be used to create '
                                                 'an ensemble.')

    parser.add_argument('-o', '--origin-folders', dest='origin_folders',
                        nargs='+',
                        help='List of directories that contains the generators file (.pkl)',
                        required=True)

    parser.add_argument('-p', '--generators-prefix', dest='dest_prefix',
                        help='Prefix defined for the destination generators (.pkl)',
                        required=True)

    parser.add_argument('-d', '--destination-folder', dest='dest_folder',
                        help='Prefix defined for the destination generators (.pkl)',
                        required=True)

    args = parser.parse_args()
    generator_prefix = args.dest_prefix
    destination_folder = args.dest_folder
    try:
        os.makedirs(destination_folder)
    except:
        print('Error: Destination folder {} cannot be created.'.format(args.dest_folder))
        sys.exit(-1)

    generator_count = 0
    for origin in args.origin_folders:
        for org_filename in glob.glob('{}/*.pkl'.format(origin)):
            destination_file = '{}/{}-{:03d}.pkl'.format(destination_folder,generator_prefix, generator_count)
            shutil.copy(org_filename, destination_file)
            print('Copied {} to {}'.format(org_filename, destination_file))
            generator_count += 1

    print('\n\nCopy completed: \n'
          ' - Folder created: {} \n'
          ' - Gerators copied: {} \n'
          ' - Generators prefix: {}.'.format(destination_folder, generator_count, generator_prefix))
    print('An example of the created generators is: {}/{}-{:03d}.pkl'.format(destination_folder, generator_prefix,0))



main()