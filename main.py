#!/usr/bin/env python3
"""
CCHM CLI - A minimal Chromecast-like screen sharing application using aiortc
"""

import argparse
import asyncio
import logging

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import mss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScreenCaptureTrack(VideoStreamTrack):
    """
    Video track that captures the screen using mss
    """

    def __init__(self, monitor_id: int = 1):
        super().__init__()
        self.sct = mss.mss()

        # Get monitor info
        monitors = self.sct.monitors
        if monitor_id < len(monitors):
            self.monitor = monitors[monitor_id]
        else:
            self.monitor = monitors[1]  # Primary monitor

        logger.info(
            f"Capturing monitor {monitor_id}: {self.monitor['width']}x{self.monitor['height']}"
        )

    async def recv(self):
        """Capture screen and return video frame"""
        pts, time_base = await self.next_timestamp()

        # Capture screen
        screenshot = self.sct.grab(self.monitor)
        img_array = np.array(screenshot)

        # Convert BGRA to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
        img_rgb = np.asarray(img_rgb, dtype=np.uint8)

        # Create video frame
        frame = VideoFrame.from_ndarray(img_rgb, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base

        return frame


class ScreencastSender:
    """Handles the sender side of screencasting - connects to receiver"""

    def __init__(self, receiver_url: str, monitor_id: int = 1):
        self.receiver_url = receiver_url
        self.monitor_id = monitor_id
        self.pc = None
        self.screen_track = None

    async def connect_and_send(self):
        """Connect to receiver and start sending screencast"""
        self.pc = RTCPeerConnection()

        # Add screen capture track
        if not self.screen_track:
            self.screen_track = ScreenCaptureTrack(self.monitor_id)
        self.pc.addTrack(self.screen_track)

        # Handle connection state changes
        def handle_connection_state_change():
            if self.pc:
                logger.info(f"Sender connection state: {self.pc.connectionState}")
                if self.pc.connectionState == "connected":
                    logger.info("Successfully sending screen to receiver!")
                elif self.pc.connectionState == "failed":
                    logger.error("Failed to connect to receiver!")
                elif self.pc.connectionState == "disconnected":
                    logger.info("Disconnected from receiver")
        
        self.pc.on("connectionstatechange", handle_connection_state_change)

        try:
            # Create offer and send to receiver
            import aiohttp

            # Create offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            async with aiohttp.ClientSession() as session:
                # Send offer to receiver and get answer
                async with session.post(
                    f"{self.receiver_url}/offer",
                    json={
                        "sdp": self.pc.localDescription.sdp,
                        "type": self.pc.localDescription.type,
                    },
                ) as response:
                    answer_data = await response.json()
                    answer = RTCSessionDescription(
                        sdp=answer_data["sdp"], type=answer_data["type"]
                    )
                    await self.pc.setRemoteDescription(answer)

            logger.info("Connected to receiver, sending screen...")
            logger.info("Press Ctrl+C to stop sending")

            # Keep the connection alive
            connection_time = 0
            try:
                while True:
                    await asyncio.sleep(1)
                    connection_time += 1

                    if connection_time % 30 == 0:  # Every 30 seconds
                        if self.pc:
                            logger.info(
                                f"Still sending ({connection_time}s) - Connection state: {self.pc.connectionState}"
                            )

            except KeyboardInterrupt:
                logger.info("Stopping sender...")

        finally:
            if self.pc:
                await self.pc.close()


class ScreencastReceiver:
    """Handles the receiver side of screencasting - waits for sender connections"""

    def __init__(self, port: int = 8080):
        self.port = port
        self.pcs = set()  # Track peer connections

    async def handle_offer(self, request):
        """Handle WebRTC offer from sender"""
        data = await request.json()
        
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        # Handle incoming video/audio tracks
        async def handle_received_track(track):
            logger.info(f"Received track: {track.kind}")
            if track.kind == "video":
                logger.info("Video track received, starting video handler...")
                await self._handle_video_track(track)
            else:
                logger.info(f"Received non-video track: {track.kind}")

        # Handle WebRTC connection state changes
        def handle_connection_state_change():
            logger.info(f"Receiver connection state: {pc.connectionState}")
            if pc.connectionState == "connected":
                logger.info("WebRTC connection established! Receiving video...")
            elif pc.connectionState == "failed":
                logger.error("WebRTC connection failed!")
                self.pcs.discard(pc)
            elif pc.connectionState == "closed":
                logger.info("WebRTC connection closed")
                self.pcs.discard(pc)
        
        # Register event handlers
        pc.on("track", handle_received_track)
        pc.on("connectionstatechange", handle_connection_state_change)

        # Set remote description from sender's offer
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        )



    async def start_listening(self):
        """Start the receiver server and wait for sender connections"""
        app = web.Application()
        app.router.add_post("/offer", self.handle_offer)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", self.port)
        await site.start()

        logger.info(f"Receiver listening on port {self.port}")
        logger.info(f"Senders should connect to: http://localhost:{self.port}")
        logger.info("Press Ctrl+C to stop listening")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down receiver...")
        finally:
            # Close all peer connections
            for pc in self.pcs:
                await pc.close()
            await runner.cleanup()

    async def _handle_video_track(self, track):
        """Handle incoming video track using OpenCV display instead of VLC"""
        logger.info("Received video track, starting OpenCV display...")

        try:
            import cv2
            import numpy as np

            logger.info("Starting video display using OpenCV...")
            logger.info("Controls:")
            logger.info("  - Press 'q' to quit")
            logger.info("  - Press 'f' to toggle fullscreen")
            logger.info("  - Press 'r' to reset window size")
            logger.info("  - Press 's' to toggle scaling mode")
            logger.info("  - Press '+' to increase window size")
            logger.info("  - Press '-' to decrease window size")
            logger.info("  - Close window to stop")

            frame_count = 0
            window_name = "CCHM CLI - Received Screencast"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing
            cv2.resizeWindow(window_name, 1280, 720)  # Start with reasonable size

            # Display settings
            is_fullscreen = False
            scale_to_fit = True
            original_size = None

            try:
                while True:
                    try:
                        # Receive frame from WebRTC
                        frame = await track.recv()
                        frame_count += 1

                        # Convert video frame to numpy array
                        img = frame.to_ndarray(format="bgr24")

                        # Ensure correct data type
                        img = np.asarray(img, dtype=np.uint8)

                        # Store original size on first frame
                        if original_size is None:
                            original_size = (
                                img.shape[1],
                                img.shape[0],
                            )  # (width, height)
                            logger.info(
                                f"Original video size: {original_size[0]}x{original_size[1]}"
                            )

                        # Handle scaling if enabled
                        display_img = img
                        if scale_to_fit:
                            # Get current window size
                            window_rect = cv2.getWindowImageRect(window_name)
                            if (
                                window_rect[2] > 0 and window_rect[3] > 0
                            ):  # Valid window size
                                window_width, window_height = (
                                    window_rect[2],
                                    window_rect[3],
                                )

                                # Calculate scaling to fit window while maintaining aspect ratio
                                img_height, img_width = img.shape[:2]
                                scale_x = window_width / img_width
                                scale_y = window_height / img_height
                                scale = min(
                                    scale_x, scale_y, 2.0
                                )  # Allow upscaling up to 2x for better visibility

                                if (
                                    abs(scale - 1.0) > 0.05
                                ):  # Only resize if significant difference
                                    new_width = int(img_width * scale)
                                    new_height = int(img_height * scale)

                                    # Use high-quality interpolation
                                    if scale > 1.0:
                                        # Upscaling - use cubic interpolation for smoothness
                                        interpolation = cv2.INTER_CUBIC
                                    else:
                                        # Downscaling - use area interpolation for quality
                                        interpolation = cv2.INTER_AREA

                                    display_img = cv2.resize(
                                        img,
                                        (new_width, new_height),
                                        interpolation=interpolation,
                                    )

                        # Display frame
                        cv2.imshow(window_name, display_img)

                        # Check for keyboard input
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            logger.info("User pressed 'q', stopping...")
                            break
                        elif key == ord("f"):
                            # Toggle fullscreen
                            is_fullscreen = not is_fullscreen
                            if is_fullscreen:
                                cv2.setWindowProperty(
                                    window_name,
                                    cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_FULLSCREEN,
                                )
                                logger.info("Fullscreen mode enabled")
                            else:
                                cv2.setWindowProperty(
                                    window_name,
                                    cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_NORMAL,
                                )
                                logger.info("Windowed mode enabled")
                        elif key == ord("r"):
                            # Reset window size to original video size
                            if original_size:
                                cv2.resizeWindow(
                                    window_name, original_size[0], original_size[1]
                                )
                                logger.info(
                                    f"Reset to original size: {original_size[0]}x{original_size[1]}"
                                )
                        elif key == ord("s"):
                            # Toggle scaling mode
                            scale_to_fit = not scale_to_fit
                            if scale_to_fit:
                                logger.info("Scaling enabled - video will fit window")
                            else:
                                logger.info(
                                    "Scaling disabled - video shows at original size"
                                )
                        elif key == ord("+") or key == ord("="):
                            # Increase window size
                            if original_size:
                                current_rect = cv2.getWindowImageRect(window_name)
                                new_width = int(current_rect[2] * 1.2)
                                new_height = int(current_rect[3] * 1.2)
                                cv2.resizeWindow(window_name, new_width, new_height)
                                logger.info(
                                    f"Increased window size to {new_width}x{new_height}"
                                )
                        elif key == ord("-"):
                            # Decrease window size
                            if original_size:
                                current_rect = cv2.getWindowImageRect(window_name)
                                new_width = int(current_rect[2] * 0.8)
                                new_height = int(current_rect[3] * 0.8)
                                if new_width > 100 and new_height > 100:  # Minimum size
                                    cv2.resizeWindow(window_name, new_width, new_height)
                                    logger.info(
                                        f"Decreased window size to {new_width}x{new_height}"
                                    )

                        # Check if window was closed
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            logger.info("Window closed by user, stopping...")
                            break

                        # Show periodic status
                        if frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                            window_rect = cv2.getWindowImageRect(window_name)
                            logger.info(
                                f"Frame {frame_count} | Window: {window_rect[2]}x{window_rect[3]} | Scaling: {'ON' if scale_to_fit else 'OFF'}"
                            )

                    except Exception as e:
                        logger.error(f"Error processing frame {frame_count}: {e}")
                        await asyncio.sleep(0.1)  # Brief pause before retry
                        continue

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            finally:
                cv2.destroyAllWindows()
                logger.info(
                    f"Video display stopped. Displayed {frame_count} total frames."
                )

        except Exception as e:
            logger.error(f"Error in video display: {e}")


def send(url: str, monitor: int):
    """Send/share your screen (sender mode)"""
    sender = ScreencastSender(url, monitor)
    asyncio.run(sender.connect_and_send())


def receive(port: int):
    """Receive and display screencast (receiver mode)"""
    receiver = ScreencastReceiver(port)
    asyncio.run(receiver.start_listening())


def monitors():
    """List available monitors for screen capture"""
    sct = mss.mss()
    print("Available monitors:")
    for i, monitor in enumerate(sct.monitors):
        if i == 0:
            print(f"  {i}: All monitors combined")
        else:
            print(f"  {i}: Monitor {i} - {monitor['width']}x{monitor['height']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CCHM CLI - Chromecast-like screen sharing using WebRTC"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Send command
    send_parser = subparsers.add_parser("send", help="Send/share your screen")
    send_parser.add_argument(
        "-u", "--url", default="http://localhost:8080", help="URL of the receiver to connect to"
    )
    send_parser.add_argument(
        "-m", "--monitor", type=int, default=1, help="Monitor ID to capture (1 for primary)"
    )

    # Receive command
    receive_parser = subparsers.add_parser("receive", help="Receive and display screencast")
    receive_parser.add_argument(
        "-p", "--port", type=int, default=8080, help="Port to listen on for sender connections"
    )

    # Monitors command
    subparsers.add_parser("monitors", help="List available monitors for screen capture")

    args = parser.parse_args()

    if args.command == "send":
        send(args.url, args.monitor)
    elif args.command == "receive":
        receive(args.port)
    elif args.command == "monitors":
        monitors()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
