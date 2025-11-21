# CCHM CLI

A minimal CLI tool for screencasting using aiortc WebRTC

## Features

- Screen capture and sharing using WebRTC (aiortc)
- Real-time video display with OpenCV
- CLI-only interface (no web interface)
- Cross-platform support

## Usage

### Send (Share your screen)
```bash
uv run main.py send --port 8080 --monitor 1
```

### Receive (View shared screen)
```bash
uv run main.py receive --url http://localhost:8080
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
- click (CLI framework)
- numpy (array processing)

## How it works

1. **Sender**: Captures screen using `mss` and streams via WebRTC using `aiortc`
2. **Receiver**: Connects to sender via WebRTC and displays video using OpenCV
3. **Signaling**: Simple HTTP server handles WebRTC offer/answer exchange

## Testing

1. Start sender: `uv run main.py send`
2. Start receiver: `uv run main.py receive`
