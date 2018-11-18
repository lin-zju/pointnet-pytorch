import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--conf', action='store', type=str, help='configuration file',
                    default='lib/config/config.json')
parser.add_argument('--test', action='store_true',  help='test_mode')

args = parser.parse_args()

