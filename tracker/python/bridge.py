"""Minimal msgpack-rpc client for arduino-router (/run/arduino-router.sock) -> STM32 Bridge RPCs.

Wire format (msgpack-rpc): request [0, msgid, method, params], response [1, msgid, error, result].
Thread-safe; reconnects on failure. Fallback: arduino.app_utils.Bridge if available.
"""
import itertools
import socket
import threading

import msgpack

SOCK = "/run/arduino-router.sock"


class Bridge:
    def __init__(self, path=SOCK, timeout=0.5):
        self.path, self.timeout = path, timeout
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._s = None
        self._unp = None

    def _connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.path)
        self._s, self._unp = s, msgpack.Unpacker(raw=False)

    def call(self, method, *params):
        with self._lock:
            for attempt in (0, 1):
                try:
                    if self._s is None:
                        self._connect()
                    mid = next(self._ids) & 0x7FFFFFFF
                    self._s.sendall(msgpack.packb([0, mid, method, list(params)]))
                    while True:
                        for msg in self._unp:
                            if msg[0] == 1 and msg[1] == mid:
                                if msg[2]:
                                    raise RuntimeError(f"{method}: {msg[2]}")
                                return msg[3]
                        data = self._s.recv(4096)
                        if not data:
                            raise ConnectionError("router closed")
                        self._unp.feed(data)
                except (OSError, ConnectionError, ValueError):
                    self.close()
                    if attempt:
                        raise

    def close(self):
        if self._s is not None:
            try:
                self._s.close()
            except OSError:
                pass
        self._s = None

    # --- motor API (sketch: tracker/sketch) ---
    def set_motor(self, speed):
        return self.call("set_motor", int(max(-255, min(255, speed))))

    def stop(self):
        return self.call("stop")

    def watchdog_trips(self):
        return self.call("watchdog_trips")
