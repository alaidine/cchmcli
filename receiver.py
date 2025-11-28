from vidstream import StreamingServer
import threading
import argparse

parser = argparse.ArgumentParser(
    prog='receiver.py',
    description='receiver part of Chrome Cast Home Made'
)

parser.add_argument('-p', '--port', type=int)
args = parser.parse_args()

receiver = StreamingServer("127.0.0.1", args.port)
t = threading.Thread(target=receiver.start_server)
t.start()

while input("") != "stop":
    continue

receiver.stop_server()
