"""Linux socket lifecycle regression; no model/GPU work."""
import errno
import socket
import sys
import unittest
import launch_node

LEGACY = '--legacy' in sys.argv
if LEGACY:
    sys.argv.remove('--legacy')

def check(port):
    if LEGACY:
        with socket.socket() as sock:
            sock.bind(('0.0.0.0',port))
    else:
        launch_node.check_bindable(port)

def listener():
    sock=socket.socket()
    sock.settimeout(2)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(('127.0.0.1',0))
    sock.listen(1)
    return sock

class PortGuardTests(unittest.TestCase):
    def test_reusable_time_wait_is_not_a_live_listener(self):
        server=listener();port=server.getsockname()[1]
        client=socket.create_connection(('127.0.0.1',port),timeout=2)
        peer,_=server.accept();peer.settimeout(2)
        peer.shutdown(socket.SHUT_WR)
        self.assertEqual(client.recv(1),b'')
        peer.close();client.close();server.close()
        with socket.socket() as plain:
            with self.assertRaises(OSError) as caught:
                plain.bind(('0.0.0.0',port))
            self.assertEqual(caught.exception.errno,errno.EADDRINUSE)
        check(port)

    def test_live_listener_still_rejected(self):
        with listener() as server:
            with self.assertRaises(OSError) as caught:
                check(server.getsockname()[1])
            self.assertEqual(caught.exception.errno,errno.EADDRINUSE)

if __name__=='__main__':
    unittest.main()
