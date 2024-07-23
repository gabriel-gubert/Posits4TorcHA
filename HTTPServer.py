import sys

import numpy as np

from urllib.parse import urlparse, parse_qs

from http.server import HTTPServer, BaseHTTPRequestHandler

from Posits4TorcHA import Accel

import argparse
from argparse import ArgumentParser

class HTTPRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.responses = {
                404: ('Not Found', ''), 
                400: ('Bad Request', ''),
                500: ('Internal Server Error', '')
                }

        self.error_message_format = '{}: {}\n\n{}'

    def do_GET(self):
        try:
            url = urlparse(self.path)
            path = url.path
            query = parse_qs(url.query)

            if path == '/load':
                if any([v not in query for v in ['Part', 'R', 'C', 'N', 'Es', 'QSize', 'Depth', 'num_threads']]):
                    self.send_error(404)

                    return
                
                self.load(
                        str(query['Part'][0]),
                        int(query['R'][0]),
                        int(query['C'][0]),
                        int(query['N'][0]),
                        int(query['Es'][0]),
                        int(query['QSize'][0]),
                        int(query['Depth'][0]),
                        int(query['num_threads'][0]),
                        force = bool(int(query['force'][0])) if 'force' in query else False
                        )

                self.send_response(200)
                self.end_headers()

                return

            self.send_error(404)
        except Exception as E:
            try:
                self.send_error(500, explain = str(E))
            except:
                pass

    def do_POST(self):
        try:
            url = urlparse(self.path)
            path = url.path
            # query = url.query

            if path == '/gemm':
                self.GEMM()

                return

            self.send_error(404)
        except Exception as E: 
            try:
                print(E)
                self.send_error(500, explain = str(E))
            except:
                pass

    def load(self, Part, R, C, N, Es, QSize, Depth, num_threads, force = False):
        default_accel.load(Part, R, C, N, Es, QSize, Depth, num_threads, force = force)

    def GEMM(self):
        Content_Type = self.headers.get('Content-Type')
        Content_Length = self.headers.get('Content-Length')
        Ar, Ac = self.headers.get('Ar'), self.headers.get('Ac')
        Br, Bc = self.headers.get('Br'), self.headers.get('Bc')

        if any([h == None for h in [Content_Type, Content_Length, Ar, Ac, Br, Bc]]):
            self.send_error(400)

            return

        Content_Type = str(Content_Type)
        Content_Length = int(Content_Length)
        Ar, Ac = int(Ar), int(Ac)
        Br, Bc = int(Br), int(Bc)

        N = 2 ** int(np.ceil(np.log2(default_accel.N)))

        if Content_Type != 'application/octet-stream':
            self.send_error(400)

            return

        if Content_Length != (Ar * Ac + Br * Bc) * N // 8:
            self.send_error(400)

            return

        buffer = self.rfile.read(Content_Length)

        C = np.frombuffer(buffer, np.__getattribute__(f'uint{N}'))
        C = np.ndarray.copy(C)

        A = C[0:Ar * Ac].reshape(Ar, Ac)
        B = C[Ar * Ac:].reshape(Br, Bc) 

        Y = default_accel.GEMM(A, B)

        self.send_response(200)

        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', Y.size * N // 8)
        self.send_header('Yr', Y.shape[0])
        self.send_header('Yc', Y.shape[1])

        self.end_headers()

        self.wfile.write(Y.tobytes())

def main(host = '127.0.0.1', port = 8080):
    HTTPServer((host, port), HTTPRequestHandler).serve_forever()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--host', type = str, default = '127.0.0.1')
    parser.add_argument('--port', type = int, default = 8080)
    parser.add_argument('--verbose', action = 'store_true')
    parser.add_argument('--report_timings', action = 'store_true')

    args = parser.parse_args()

    default_accel = Accel(verbose = args.verbose, report_timings = args.report_timings)

    main(args.host, args.port)
