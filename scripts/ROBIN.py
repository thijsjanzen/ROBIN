import sys
from scripts import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ROBust INference of admixture time')
    parser.add_argument('--config', default='robin.ini', help='the path to the config file')
    parser.add_argument('--seed', default = 42, help='random seed')

    args = parser.parse_args()
    # Well behaved unix programs exits with 0 on success...
    sys.exit(main.main(args.config, args.seed))
