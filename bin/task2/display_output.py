import os
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..', '..'))

import research.boosting as rb
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="filename to display in /output/ folder")
    args = parser.parse_args()
    rb.grid_search_analysis(str(args.filename))
