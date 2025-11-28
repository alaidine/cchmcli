# cchm (Chrome Cast Home Made)

### Usage

Receiver:
```
# To start the receiver
$ uv run receiver.py -p <port>
stop # Type stop in the terminal to stop the receiver
```

Sender:
```
# To send screencast to the receiver
$ uv run sender.py --port <receiver_port> --host <receiver_host> --width <width> --height <height>
```
