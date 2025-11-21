#!/usr/bin/env python3
"""
CCHM CLI - A minimal Chromecast-like screen sharing application using aiortc
"""

import asyncio
import logging
from typing import Optional

import click
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
            
        logger.info(f"Capturing monitor {monitor_id}: {self.monitor['width']}x{self.monitor['height']}")

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
    """Handles the sender side of screencasting"""

    def __init__(self, port: int = 8080, monitor_id: int = 1):
        self.port = port
        self.monitor_id = monitor_id
        self.pcs = set()  # Track peer connections
        self.screen_track = None

    async def create_offer(self, request):
        """Create WebRTC offer"""
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        # Add screen capture track
        if not self.screen_track:
            self.screen_track = ScreenCaptureTrack(self.monitor_id)
        pc.addTrack(self.screen_track)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                self.pcs.discard(pc)

        # Create offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        return web.json_response(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        )

    async def handle_answer(self, request):
        """Handle WebRTC answer"""
        data = await request.json()

        # Find the corresponding peer connection (simplified - in production use proper matching)
        if self.pcs:
            pc = list(self.pcs)[-1]  # Use the most recent connection
            answer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await pc.setRemoteDescription(answer)

        return web.json_response({"status": "ok"})



    async def start_server(self):
        """Start the WebRTC signaling server"""
        app = web.Application()
        app.router.add_get("/offer", self.create_offer)
        app.router.add_post("/answer", self.handle_answer)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", self.port)
        await site.start()

        logger.info(f"WebRTC signaling server started on port {self.port}")
        logger.info(f"Receiver can connect to: http://localhost:{self.port}")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            # Close all peer connections
            for pc in self.pcs:
                await pc.close()
            await runner.cleanup()


class ScreencastReceiver:
    """Handles the receiver side of screencasting"""

    def __init__(self, sender_url: str):
        self.sender_url = sender_url
        self.pc = None

    async def connect_and_receive(self):
        """Connect to sender and start receiving screencast"""
        self.pc = RTCPeerConnection()

        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"Received track: {track.kind}")
            if track.kind == "video":
                logger.info("Video track received, starting video handler...")
                await self._handle_video_track(track)
            else:
                logger.info(f"Received non-video track: {track.kind}")

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if self.pc:
                logger.info(f"Connection state changed to: {self.pc.connectionState}")
                if self.pc.connectionState == "connected":
                    logger.info("WebRTC connection established!")
                elif self.pc.connectionState == "failed":
                    logger.error("WebRTC connection failed!")
                elif self.pc.connectionState == "disconnected":
                    logger.info("WebRTC connection disconnected")

        try:
            # Get offer from sender
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.sender_url}/offer") as response:
                    offer_data = await response.json()
                    offer = RTCSessionDescription(
                        sdp=offer_data["sdp"], type=offer_data["type"]
                    )

                await self.pc.setRemoteDescription(offer)
                answer = await self.pc.createAnswer()
                await self.pc.setLocalDescription(answer)

                # Send answer back to sender
                async with session.post(
                    f"{self.sender_url}/answer",
                    json={
                        "sdp": self.pc.localDescription.sdp,
                        "type": self.pc.localDescription.type,
                    },
                ) as response:
                    await response.json()

            logger.info("Connected to sender, waiting for video stream...")
            logger.info("Waiting for video track... (this may take a few seconds)")
            
            # Keep the connection alive and show status
            connection_time = 0
            try:
                while True:
                    await asyncio.sleep(1)
                    connection_time += 1
                    
                    if connection_time % 10 == 0:  # Every 10 seconds
                        logger.info(f"Still connected ({connection_time}s) - Connection state: {self.pc.connectionState}")
                        
                    if connection_time > 60:  # After 60 seconds
                        logger.warning("No video received after 60 seconds. Check if sender is working properly.")
                        logger.info("Try connecting with another receiver to verify sender is working")
                        
            except KeyboardInterrupt:
                logger.info("Shutting down receiver...")
        
        finally:
            if self.pc:
                await self.pc.close()

    async def _handle_video_track(self, track):
        """Handle incoming video track using OpenCV display instead of VLC"""
        logger.info(f"Received video track, starting OpenCV display...")

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
                            original_size = (img.shape[1], img.shape[0])  # (width, height)
                            logger.info(f"Original video size: {original_size[0]}x{original_size[1]}")
                        
                        # Handle scaling if enabled
                        display_img = img
                        if scale_to_fit:
                            # Get current window size
                            window_rect = cv2.getWindowImageRect(window_name)
                            if window_rect[2] > 0 and window_rect[3] > 0:  # Valid window size
                                window_width, window_height = window_rect[2], window_rect[3]
                                
                                # Calculate scaling to fit window while maintaining aspect ratio
                                img_height, img_width = img.shape[:2]
                                scale_x = window_width / img_width
                                scale_y = window_height / img_height
                                scale = min(scale_x, scale_y, 2.0)  # Allow upscaling up to 2x for better visibility
                                
                                if abs(scale - 1.0) > 0.05:  # Only resize if significant difference
                                    new_width = int(img_width * scale)
                                    new_height = int(img_height * scale)
                                    
                                    # Use high-quality interpolation
                                    if scale > 1.0:
                                        # Upscaling - use cubic interpolation for smoothness
                                        interpolation = cv2.INTER_CUBIC
                                    else:
                                        # Downscaling - use area interpolation for quality
                                        interpolation = cv2.INTER_AREA
                                    
                                    display_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
                        
                        # Display frame
                        cv2.imshow(window_name, display_img)
                        
                        # Check for keyboard input
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("User pressed 'q', stopping...")
                            break
                        elif key == ord('f'):
                            # Toggle fullscreen
                            is_fullscreen = not is_fullscreen
                            if is_fullscreen:
                                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                logger.info("Fullscreen mode enabled")
                            else:
                                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                                logger.info("Windowed mode enabled")
                        elif key == ord('r'):
                            # Reset window size to original video size
                            if original_size:
                                cv2.resizeWindow(window_name, original_size[0], original_size[1])
                                logger.info(f"Reset to original size: {original_size[0]}x{original_size[1]}")
                        elif key == ord('s'):
                            # Toggle scaling mode
                            scale_to_fit = not scale_to_fit
                            if scale_to_fit:
                                logger.info("Scaling enabled - video will fit window")
                            else:
                                logger.info("Scaling disabled - video shows at original size")
                        elif key == ord('+') or key == ord('='):
                            # Increase window size
                            if original_size:
                                current_rect = cv2.getWindowImageRect(window_name)
                                new_width = int(current_rect[2] * 1.2)
                                new_height = int(current_rect[3] * 1.2)
                                cv2.resizeWindow(window_name, new_width, new_height)
                                logger.info(f"Increased window size to {new_width}x{new_height}")
                        elif key == ord('-'):
                            # Decrease window size
                            if original_size:
                                current_rect = cv2.getWindowImageRect(window_name)
                                new_width = int(current_rect[2] * 0.8)
                                new_height = int(current_rect[3] * 0.8)
                                if new_width > 100 and new_height > 100:  # Minimum size
                                    cv2.resizeWindow(window_name, new_width, new_height)
                                    logger.info(f"Decreased window size to {new_width}x{new_height}")
                        
                        # Check if window was closed
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            logger.info("Window closed by user, stopping...")
                            break
                            
                        # Show periodic status
                        if frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                            window_rect = cv2.getWindowImageRect(window_name)
                            logger.info(f"Frame {frame_count} | Window: {window_rect[2]}x{window_rect[3]} | Scaling: {'ON' if scale_to_fit else 'OFF'}")
                            
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_count}: {e}")
                        await asyncio.sleep(0.1)  # Brief pause before retry
                        continue
                        
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            finally:
                cv2.destroyAllWindows()
                logger.info(f"Video display stopped. Displayed {frame_count} total frames.")

        except Exception as e:
            logger.error(f"Error in video display: {e}")


@click.group()
def cli():
    """CCHM CLI - Chromecast-like screen sharing using WebRTC"""
    pass


@cli.command()
@click.option("--port", "-p", default=8080, help="Port to run the sender server on")
@click.option("--monitor", "-m", default=1, help="Monitor ID to capture (1 for primary)")
def send(port: int, monitor: int):
    """Send/share your screen (sender mode)"""
    sender = ScreencastSender(port, monitor)
    asyncio.run(sender.start_server())


@cli.command()
@click.option('--url', '-u', default='http://localhost:8080', help='URL of the sender')
def receive(url: str):
    """Receive and display screencast (receiver mode)"""
    receiver = ScreencastReceiver(url)
    asyncio.run(receiver.connect_and_receive())


@cli.command()
def monitors():
    """List available monitors for screen capture"""
    sct = mss.mss()
    click.echo("Available monitors:")
    for i, monitor in enumerate(sct.monitors):
        if i == 0:
            click.echo(f"  {i}: All monitors combined")
        else:
            click.echo(f"  {i}: Monitor {i} - {monitor['width']}x{monitor['height']}")


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()
