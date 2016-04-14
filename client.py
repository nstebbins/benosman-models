#!/usr/bin/python

import socket

host = "0.0.0.0"
port = 8000

tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.connect((host, port))
tcpsock.send("inverting_memory;input 2000 input 2900 input 5000".encode())
tcpsock.close()
