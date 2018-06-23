import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                   const=sum, default=max,
                   help='sum the integers (default: find the max)')
parser.add_argument("-e", "--epochs", help="for how many epochs to run", type=int, )

args = parser.parse_args()
print("args.integers: ", args.integers)
print("size of args.integers: ", len(args.integers))
print("args.epochs: ", args.epochs)
print(args.accumulate(args.integers))

