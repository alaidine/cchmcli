from vidstream import StreamingServer
import threading
import argparse
import socket

def get_local_ip():
    """Get the local IP address that can be used by other machines"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

parser = argparse.ArgumentParser(
    prog='receiver.py',
    description='receiver part of Chrome Cast Home Made'
)

parser.add_argument('-p', '--port', type=int)
args = parser.parse_args()

# Get local IP to accept connections from other computers
local_ip = get_local_ip()
print(f"Starting receiver on {local_ip}:{args.port}")
print(f"Connect from other computers using: {local_ip}:{args.port}")

receiver = StreamingServer(local_ip, args.port)
t = threading.Thread(target=receiver.start_server)
t.start()

while input("") != "stop":
    continue

receiver.stop_server()
