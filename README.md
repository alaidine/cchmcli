# CCHM CLI

A minimal CLI tool for screencasting using aiortc WebRTC

## Features

- Screen capture and sharing using WebRTC (aiortc)
- Real-time video display with OpenCV
- CLI-only interface (no web interface)
- Cross-platform support

## Usage

### Same Machine
```bash
# Start receiver
uv run main.py receive --port 8080

# Start sender (in another terminal)
uv run main.py send --url http://localhost:8080 --monitor 1
```

### Different Machines (Network)
```bash
# On receiver machine (Machine A)
uv run main.py receive --port 8080
# Note the IP address shown in the output

# On sender machine (Machine B) 
uv run main.py send --url http://192.168.1.100:8080 --monitor 1
# Replace 192.168.1.100 with the receiver's actual IP address
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

## Network Requirements

### For Different Machines
- Both machines must be on the same network (LAN/WiFi)
- Receiver machine firewall should allow incoming connections on the chosen port (default: 8080)
- If using Windows Firewall, you may need to allow the Python application through

### Port Configuration
- Default port: 8080
- Change with: `--port 9090` (use any available port)
- Make sure the port is not blocked by firewall or used by other applications

## How it works

1. **Receiver**: Starts HTTP server on all interfaces (0.0.0.0) and waits for sender connections
2. **Sender**: Captures screen using `mss` and connects to receiver via WebRTC using `aiortc`
3. **Display**: Receiver displays incoming video using OpenCV
4. **Signaling**: HTTP server handles WebRTC offer/answer exchange

## Testing

1. Start receiver (listener): `uv run main.py receive`
2. Start sender (connects to receiver): `uv run main.py send`
