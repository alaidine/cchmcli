import cv2
import pyautogui
import numpy as np

import socket
import pickle
import struct
import threading
import platform

# Windows-specific imports for cursor capture
if platform.system() == 'Windows':
    import win32gui
    import win32ui
    import win32con
    import win32api


class StreamingServer:
    """
    Class for the streaming server.

    Attributes
    ----------

    Private:

        __host : str
            host address of the listening server
        __port : int
            port on which the server is listening
        __slots : int
            amount of maximum avaialable slots (not ready yet)
        __used_slots : int
            amount of used slots (not ready yet)
        __quit_key : chr
            key that has to be pressed to close connection
        __running : bool
            inicates if the server is already running or not
        __block : Lock
            a basic lock used for the synchronization of threads
        __server_socket : socket
            the main server socket


    Methods
    -------

    Private:

        __init_socket : method that binds the server socket to the host and port
        __server_listening: method that listens for new connections
        __client_connection : main method for processing the client streams

    Public:

        start_server : starts the server in a new thread
        stop_server : stops the server and closes all connections
    """

    # TODO: Implement slots functionality
    def __init__(self, host, port, slots=8, quit_key='q'):
        """
        Creates a new instance of StreamingServer

        Parameters
        ----------

        host : str
            host address of the listening server
        port : int
            port on which the server is listening
        slots : int
            amount of avaialable slots (not ready yet) (default = 8)
        quit_key : chr
            key that has to be pressed to close connection (default = 'q')  
        """
        self.__host = host
        self.__port = port
        self.__slots = slots
        self.__used_slots = 0
        self.__running = False
        self.__quit_key = quit_key
        self.__block = threading.Lock()
        self.__server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__init_socket()

    def __init_socket(self):
        """
        Binds the server socket to the given host and port
        """
        self.__server_socket.bind((self.__host, self.__port))

    def start_server(self):
        """
        Starts the server if it is not running already.
        """
        if self.__running:
            print("Server is already running")
        else:
            self.__running = True
            server_thread = threading.Thread(target=self.__server_listening)
            server_thread.start()

    def __server_listening(self):
        """
        Listens for new connections.
        """
        self.__server_socket.listen()
        while self.__running:
            self.__block.acquire()
            connection, address = self.__server_socket.accept()
            if self.__used_slots >= self.__slots:
                print("Connection refused! No free slots!")
                connection.close()
                self.__block.release()
                continue
            else:
                self.__used_slots += 1
            self.__block.release()
            thread = threading.Thread(target=self.__client_connection, args=(connection, address,))
            thread.start()

    def stop_server(self):
        """
        Stops the server and closes all connections
        """
        if self.__running:
            self.__running = False
            closing_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            closing_connection.connect((self.__host, self.__port))
            closing_connection.close()
            self.__block.acquire()
            self.__server_socket.close()
            self.__block.release()
        else:
            print("Server not running!")

    def __client_connection(self, connection, address):
        """
        Handles the individual client connections and processes their stream data.
        """
        payload_size = struct.calcsize('>L')
        data = b""

        while self.__running:

            break_loop = False

            while len(data) < payload_size:
                received = connection.recv(4096)
                if received == b'':
                    connection.close()
                    self.__used_slots -= 1
                    break_loop = True
                    break
                data += received

            if break_loop:
                break

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]

            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += connection.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            cv2.imshow(str(address), frame)
            if cv2.waitKey(1) == ord(self.__quit_key):
                connection.close()
                self.__used_slots -= 1
                break


class StreamingClient:
    """
    Abstract class for the generic streaming client.

    Attributes
    ----------

    Private:

        __host : str
            host address to connect to
        __port : int
            port to connect to
        __running : bool
            inicates if the client is already streaming or not
        __encoding_parameters : list
            a list of encoding parameters for OpenCV
        __client_socket : socket
            the main client socket


    Methods
    -------

    Private:

        __client_streaming : main method for streaming the client data

    Protected:

        _configure : sets basic configurations (overridden by child classes)
        _get_frame : returns the frame to be sent to the server (overridden by child classes)
        _cleanup : cleans up all the resources and closes everything

    Public:

        start_stream : starts the client stream in a new thread
    """

    def __init__(self, host, port):
        """
        Creates a new instance of StreamingClient.

        Parameters
        ----------

        host : str
            host address to connect to
        port : int
            port to connect to
        """
        self.__host = host
        self.__port = port
        self._configure()
        self.__running = False
        self.__client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def _configure(self):
        """
        Basic configuration function.
        """
        self.__encoding_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    def _get_frame(self):
        """
        Basic function for getting the next frame.

        Returns
        -------

        frame : the next frame to be processed (default = None)
        """
        return None

    def _cleanup(self):
        """
        Cleans up resources and closes everything.
        """
        cv2.destroyAllWindows()

    def __client_streaming(self):
        """
        Main method for streaming the client data.
        """
        self.__client_socket.connect((self.__host, self.__port))
        while self.__running:
            frame = self._get_frame()
            result, frame = cv2.imencode('.jpg', frame, self.__encoding_parameters)
            data = pickle.dumps(frame, 0)
            size = len(data)

            try:
                self.__client_socket.sendall(struct.pack('>L', size) + data)
            except ConnectionResetError:
                self.__running = False
            except ConnectionAbortedError:
                self.__running = False
            except BrokenPipeError:
                self.__running = False

        self._cleanup()

    def start_stream(self):
        """
        Starts client stream if it is not already running.
        """

        if self.__running:
            print("Client is already streaming!")
        else:
            self.__running = True
            client_thread = threading.Thread(target=self.__client_streaming)
            client_thread.start()

    def stop_stream(self):
        """
        Stops client stream if running
        """
        if self.__running:
            self.__running = False
        else:
            print("Client not streaming!")


class CameraClient(StreamingClient):
    """
    Class for the camera streaming client.

    Attributes
    ----------

    Private:

        __host : str
            host address to connect to
        __port : int
            port to connect to
        __running : bool
            inicates if the client is already streaming or not
        __encoding_parameters : list
            a list of encoding parameters for OpenCV
        __client_socket : socket
            the main client socket
        __camera : VideoCapture
            the camera object
        __x_res : int
            the x resolution
        __y_res : int
            the y resolution


    Methods
    -------

    Protected:

        _configure : sets basic configurations
        _get_frame : returns the camera frame to be sent to the server
        _cleanup : cleans up all the resources and closes everything

    Public:

        start_stream : starts the camera stream in a new thread
    """

    def __init__(self, host, port, x_res=1024, y_res=576):
        """
        Creates a new instance of CameraClient.

        Parameters
        ----------

        host : str
            host address to connect to
        port : int
            port to connect to
        x_res : int
            the x resolution
        y_res : int
            the y resolution
        """
        self.__x_res = x_res
        self.__y_res = y_res
        self.__camera = cv2.VideoCapture(0)
        super(CameraClient, self).__init__(host, port)

    def _configure(self):
        """
        Sets the camera resultion and the encoding parameters.
        """
        self.__camera.set(3, self.__x_res)
        self.__camera.set(4, self.__y_res)
        super(CameraClient, self)._configure()

    def _get_frame(self):
        """
        Gets the next camera frame.

        Returns
        -------

        frame : the next camera frame to be processed
        """
        ret, frame = self.__camera.read()
        return frame

    def _cleanup(self):
        """
        Cleans up resources and closes everything.
        """
        self.__camera.release()
        cv2.destroyAllWindows()


class VideoClient(StreamingClient):
    """
    Class for the video streaming client.

    Attributes
    ----------

    Private:

        __host : str
            host address to connect to
        __port : int
            port to connect to
        __running : bool
            inicates if the client is already streaming or not
        __encoding_parameters : list
            a list of encoding parameters for OpenCV
        __client_socket : socket
            the main client socket
        __video : VideoCapture
            the video object
        __loop : bool
            boolean that decides whether the video shall loop or not


    Methods
    -------

    Protected:

        _configure : sets basic configurations
        _get_frame : returns the video frame to be sent to the server
        _cleanup : cleans up all the resources and closes everything

    Public:

        start_stream : starts the video stream in a new thread
    """

    def __init__(self, host, port, video, loop=True):
        """
        Creates a new instance of VideoClient.

        Parameters
        ----------

        host : str
            host address to connect to
        port : int
            port to connect to
        video : str
            path to the video
        loop : bool
            indicates whether the video shall loop or not
        """
        self.__video = cv2.VideoCapture(video)
        self.__loop = loop
        super(VideoClient, self).__init__(host, port)

    def _configure(self):
        """
        Set video resolution and encoding parameters.
        """
        self.__video.set(3, 1024)
        self.__video.set(4, 576)
        super(VideoClient, self)._configure()

    def _get_frame(self):
        """
        Gets the next video frame.

        Returns
        -------

        frame : the next video frame to be processed
        """
        ret, frame = self.__video.read()
        return frame

    def _cleanup(self):
        """
        Cleans up resources and closes everything.
        """
        self.__video.release()
        cv2.destroyAllWindows()


class ScreenShareClient(StreamingClient):
    """
    Class for the screen share streaming client.

    Attributes
    ----------

    Private:

        __host : str
            host address to connect to
        __port : int
            port to connect to
        __running : bool
            inicates if the client is already streaming or not
        __encoding_parameters : list
            a list of encoding parameters for OpenCV
        __client_socket : socket
            the main client socket
        __x_res : int
            the x resolution
        __y_res : int
            the y resolution


    Methods
    -------

    Protected:

        _get_frame : returns the screenshot frame to be sent to the server

    Public:

        start_stream : starts the screen sharing stream in a new thread
    """

    def __init__(self, host, port, x_res=1024, y_res=576):
        """
        Creates a new instance of ScreenShareClient.

        Parameters
        ----------

        host : str
            host address to connect to
        port : int
            port to connect to
        x_res : int
            the x resolution
        y_res : int
            the y resolution
        """
        self.__x_res = x_res
        self.__y_res = y_res
        super(ScreenShareClient, self).__init__(host, port)

    def _get_frame(self):
        """
        Gets the next screenshot.

        Returns
        -------

        frame : the next screenshot frame to be processed
        """
        screen = pyautogui.screenshot()
        frame = np.array(screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw the cursor based on platform
        if platform.system() == 'Windows':
            frame = self._draw_windows_cursor(frame)
        else:
            frame = self._draw_fallback_cursor(frame)
        
        frame = cv2.resize(frame, (self.__x_res, self.__y_res), interpolation=cv2.INTER_AREA)
        return frame

    def _draw_fallback_cursor(self, frame):
        """
        Draws a simple cursor pointer for non-Windows platforms.
        
        Parameters
        ----------
        frame : numpy array
            The screenshot frame to draw the cursor on
            
        Returns
        -------
        frame : numpy array
            The frame with the cursor drawn on it
        """
        mouse_x, mouse_y = pyautogui.position()
        
        # Draw a cursor pointer (arrow shape)
        cursor_size = 20
        
        # Arrow pointer vertices
        pts = np.array([
            [mouse_x, mouse_y],
            [mouse_x, mouse_y + cursor_size],
            [mouse_x + cursor_size // 3, mouse_y + cursor_size * 2 // 3],
            [mouse_x + cursor_size // 2, mouse_y + cursor_size],
            [mouse_x + cursor_size * 2 // 3, mouse_y + cursor_size * 2 // 3],
            [mouse_x + cursor_size, mouse_y + cursor_size // 2],
        ], np.int32)
        
        # Simple arrow (just the main pointer part)
        simple_pts = np.array([
            [mouse_x, mouse_y],
            [mouse_x, mouse_y + cursor_size],
            [mouse_x + cursor_size * 2 // 5, mouse_y + cursor_size * 3 // 5],
        ], np.int32)
        simple_pts = simple_pts.reshape((-1, 1, 2))
        
        # Draw white fill with black outline
        cv2.fillPoly(frame, [simple_pts], (255, 255, 255))
        cv2.polylines(frame, [simple_pts], True, (0, 0, 0), 1, cv2.LINE_AA)
        
        return frame

    def _draw_windows_cursor(self, frame):
        """
        Draws the actual Windows system cursor onto the frame.
        
        Parameters
        ----------
        frame : numpy array
            The screenshot frame to draw the cursor on
            
        Returns
        -------
        frame : numpy array
            The frame with the cursor drawn on it
        """
        try:
            # Get cursor info
            cursor_info = win32gui.GetCursorInfo()
            cursor_handle = cursor_info[1]
            cursor_x, cursor_y = cursor_info[2]
            
            # Get icon info to find the hotspot
            icon_info = win32gui.GetIconInfo(cursor_handle)
            hotspot_x = icon_info[1]
            hotspot_y = icon_info[2]
            
            # Clean up bitmaps from GetIconInfo
            if icon_info[3]:
                win32gui.DeleteObject(icon_info[3])
            if icon_info[4]:
                win32gui.DeleteObject(icon_info[4])
            
            # Create a device context and bitmap to draw the cursor
            cursor_size = 32
            
            # Create device contexts
            hdc = win32ui.CreateDCFromHandle(win32gui.GetDC(0))
            hdc_mem = hdc.CreateCompatibleDC()
            
            # Create bitmap
            hbmp = win32ui.CreateBitmap()
            hbmp.CreateCompatibleBitmap(hdc, cursor_size, cursor_size)
            hdc_mem.SelectObject(hbmp)
            
            # Fill with transparent color (magenta as key)
            hdc_mem.FillSolidRect((0, 0, cursor_size, cursor_size), 0xFF00FF)
            
            # Draw the cursor onto the bitmap
            win32gui.DrawIconEx(
                hdc_mem.GetSafeHdc(), 0, 0, cursor_handle,
                cursor_size, cursor_size, 0, None, win32con.DI_NORMAL
            )
            
            # Convert bitmap to numpy array
            bmp_info = hbmp.GetInfo()
            bmp_bits = hbmp.GetBitmapBits(True)
            cursor_img = np.frombuffer(bmp_bits, dtype=np.uint8)
            cursor_img = cursor_img.reshape((bmp_info['bmHeight'], bmp_info['bmWidth'], 4))
            
            # Clean up
            hdc_mem.DeleteDC()
            win32gui.ReleaseDC(0, hdc.GetSafeHdc())
            win32gui.DeleteObject(hbmp.GetHandle())
            
            # Create mask (magenta = transparent)
            mask = ~((cursor_img[:, :, 0] == 0xFF) & 
                     (cursor_img[:, :, 1] == 0x00) & 
                     (cursor_img[:, :, 2] == 0xFF))
            
            # Calculate position adjusted for hotspot
            draw_x = cursor_x - hotspot_x
            draw_y = cursor_y - hotspot_y
            
            # Overlay cursor on frame
            for y in range(cursor_size):
                for x in range(cursor_size):
                    frame_x = draw_x + x
                    frame_y = draw_y + y
                    if (0 <= frame_x < frame.shape[1] and 
                        0 <= frame_y < frame.shape[0] and mask[y, x]):
                        # BGR from cursor to RGB frame
                        frame[frame_y, frame_x] = cursor_img[y, x, :3]
                        
        except Exception as e:
            # Fallback: draw a simple cursor if Windows API fails
            mouse_x, mouse_y = pyautogui.position()
            cv2.circle(frame, (mouse_x, mouse_y), 5, (255, 255, 255), -1)
            cv2.circle(frame, (mouse_x, mouse_y), 5, (0, 0, 0), 1)
            
        return frame
