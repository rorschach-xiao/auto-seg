import _init_paths

import sys

print(sys.path)

from model_flying import demo_server
import os
import argparse

import warnings
warnings.filterwarnings("ignore")


def start_server(port, visible_devices_list):
    demo_server.start_server(port, visible_devices_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoCV classification Module')
    parser.add_argument('--visible_devices_list', type = str, default = '0,1')
    parser.add_argument('--port', type=int, default = 5000)

    args, unknow = parser.parse_known_args()
    # rest_args = list(unknow)

    start_server(args.port, args.visible_devices_list)

