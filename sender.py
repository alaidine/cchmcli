from vidstream import ScreenShareClient
import argparse

parser = argparse.ArgumentParser(
    prog='sender.py',
    description='sender part of Chrome Cast Home Made'
)

parser.add_argument('-p', '--port', type=int)
parser.add_argument('-ip', '--host', type=str)
parser.add_argument('-x', '--width', type=int)
parser.add_argument('-y', '--height', type=int)

args = parser.parse_args()

sender = ScreenShareClient(args.host, args.port, args.width, args.height)
sender.start_stream()
