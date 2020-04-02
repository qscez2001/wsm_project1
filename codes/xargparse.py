import argparse
# python main.py --query <query>
# drill wood sharp

parser = argparse.ArgumentParser()

parser.add_argument('--query', dest="query", help="enter query", type=str)

args = parser.parse_args()

print(args.query)