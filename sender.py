from vidstream import ScreenShareClient

sender = ScreenShareClient("127.0.0.1", 9999)
sender.start_stream()
