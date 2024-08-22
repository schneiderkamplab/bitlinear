from argparse import ArgumentParser
from tests.Benchmark import Benchmark

    
parser = ArgumentParser()

parser.add_argument("--kernel", type=str, default='TorchLinear')
parser.add_argument("-a", action='store_true')
parser.add_argument("-p", action='store_true')
parser.add_argument("-u", action='store_true')
parser.add_argument("-t", action='store_true')
parser.add_argument("--save_dir", type=str, default='results/tmp')
args = parser.parse_args()

Benchmark(args)
