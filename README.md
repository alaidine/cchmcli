# CCHM CLI

A minimal CLI tool for screencasting using aiortc WebRTC

## Features

- Screen capture and sharing using WebRTC (aiortc)
- Real-time video display with OpenCV
- CLI-only interface (no web interface)
- Cross-platform support

## Usage

### Receive (Start listening for screen shares)
```bash
uv run main.py receive --port 8080
```

### Send (Connect and share your screen)
```bash
uv run main.py send --url http://localhost:8080 --monitor 1
```

### List monitors
```bash
uv run main.py monitors
```

## Installation

```bash
uv sync
```

## Dependencies

- aiortc (WebRTC implementation)
- opencv-python (video display)
- mss (screen capture)
- aiohttp (web server)
- numpy (array processing)

## How it works

1. **Receiver**: Starts HTTP server and waits for sender connections
2. **Sender**: Captures screen using `mss` and connects to receiver via WebRTC using `aiortc`
3. **Display**: Receiver displays incoming video using OpenCV
4. **Signaling**: HTTP server handles WebRTC offer/answer exchange

## Testing

1. Start receiver (listener): `uv run main.py receive`
2. Start sender (connects to receiver): `uv run main.py send`
