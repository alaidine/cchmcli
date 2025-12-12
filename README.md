# CCHM - Chrome Cast Home Made

A screen casting tool written in Python. Stream your screen from one computer to another over your local network.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Windows (for full cursor capture support)

## Features

- **Screen Sharing** - Stream your entire screen to another computer
- **Cursor Capture** - Shows your actual system cursor on Windows (pointer, text cursor, resize handles, etc.) or a generic arrow on Linux/macOS
- **Camera Streaming** - Stream from your webcam
- **Video Streaming** - Stream video files
- **Simple CLI Interface** - Easy to use command-line tools
- **LAN Support** - Works across different machines on the same network
- **Cross-Platform** - Works on Windows and Linux

### Core Components (`streaming.py`)

| Class | Description |
|-------|-------------|
| `StreamingServer` | TCP server that receives and displays video streams. Handles multiple client connections with slot management. |
| `StreamingClient` | Abstract base class for all streaming clients. Handles connection, encoding, and transmission. |
| `ScreenShareClient` | Captures screen using `pyautogui`, overlays the system cursor, and streams to server. |
| `CameraClient` | Captures video from webcam using OpenCV and streams to server. |
| `VideoClient` | Streams a video file to the server with optional looping. |

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alaidine/cchmcli.git
   cd cchmcli
   ```

2. **Install dependencies using uv**
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

## Usage

### Quick Start

1. **On the receiving computer** (the display):
   ```bash
   uv run receiver.py -p 4242
   ```
   The receiver will print its IP address - note this for the sender.

2. **On the sending computer** (the one sharing screen):
   ```bash
   uv run sender.py --host <receiver_ip> --port 4242 --width 1280 --height 720
   ```

3. **To stop**: Type `stop` in the receiver terminal and press Enter.

### Receiver Options

```bash
uv run receiver.py -p <port>
```

| Option | Description |
|--------|-------------|
| `-p, --port` | Port number to listen on (required) |

### Sender Options

```bash
uv run sender.py --host <ip> --port <port> --width <w> --height <h>
```

| Option | Description |
|--------|-------------|
| `-ip, --host` | IP address of the receiver (required) |
| `-p, --port` | Port number of the receiver (required) |
| `-x, --width` | Output stream width in pixels (required) |
| `-y, --height` | Output stream height in pixels (required) |

### Example

```bash
# Machine A (receiver)
uv run receiver.py -p 5000

# Machine B (sender)
uv run sender.py --host <receiver_ip> --port 5000 --width 1920 --height 1080
```

## Contributing

Contributions are welcome! Here's how to get started:

### Development Setup

1. Fork and clone the repository
2. Install development dependencies:
   ```bash
   uv sync
   ```

### Code Overview for Contributors

**Adding a new streaming client type:**

1. Create a new class that extends `StreamingClient`
2. Override `_configure()` for setup (optional)
3. Override `_get_frame()` to return frames as numpy arrays
4. Override `_cleanup()` for resource cleanup (optional)

Example:
```python
class MyCustomClient(StreamingClient):
    def __init__(self, host, port):
        super().__init__(host, port)
    
    def _get_frame(self):
        # Return a numpy array (BGR format)
        frame = ...  # Your frame capture logic
        return frame
```

**Key implementation details:**

- Frames are JPEG-encoded before transmission for bandwidth efficiency
- Protocol uses `struct` for message size headers (big-endian unsigned long)
- Data is serialized with `pickle` for transmission
- The server uses threading for handling multiple clients

### Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions and classes
- Test your changes on a local network before submitting
- Update README if adding new features

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Ensure firewall allows the port, and both machines are on the same network |
| Black screen | Check that the sender resolution matches your display |
| Cursor not showing actual cursor | Full cursor capture is Windows-only; Linux/macOS shows a generic arrow pointer |
| High latency | Try lowering the resolution (e.g., 1280x720) |

## Dependencies

- **opencv-python** - Video capture and image processing
- **pyautogui** - Screen capture and mouse position
- **numpy** - Array operations for frame manipulation
- **pywin32** - Windows API for cursor capture (Windows only, installed automatically)
- **pillow** - Image processing support

## License

This project is open source. See repository for license details.

## Acknowledgments

- **[NeuralNine](https://www.youtube.com/@NeuralNine)** - Creator of the [vidstream](https://github.com/NeuralNine/vidstream) module, which the `streaming.py` file is based on
- Inspired by the need for a simple, lightweight alternative to commercial screen casting solutions
