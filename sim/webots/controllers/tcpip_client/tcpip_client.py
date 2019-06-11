import os
os.environ["PYTHONUNBUFFERED"] = "1"

from controller import Supervisor
import socket, pickle, selectors
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def start_connection(host, port, sel):
    addr = (host,port)
    print("[client]: starting connection to", addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = {"selector": sel,"socket" : sock, "address" : addr}
    sel.register(sock, events, data=message)

def write_sim_data(sock, sim_data):
    print("[client]: write sim data")
    sock.send(sim_data)
    
def read_instructions(sock):
    data = []
    while True:
        try:
            packet = sock.recv(4096)
        except BlockingIOError:
            # Resource temporarily unavailable (errno EWOULDBLOCK)
            print("[client]: socket blocked")
            break
        if not packet: break
        data.append(packet)
    if len(data) > 0:
        #TODO process data
        pass
    return data


sel = selectors.DefaultSelector()
host, port = 'localhost', 10000
sim_data = pickle.dumps(dict(reward= np.random.randint(1,10), done=True))
start_connection(host, port, sel)

try:
    while True:
        events = sel.select(timeout=1)
        for key, mask in events:
            sock = key.data["socket"]
            sel = key.data["selector"]

            if mask & selectors.EVENT_WRITE:
                write_sim_data(sock, sim_data)
                sel.modify(sock, selectors.EVENT_READ, data=key.data)

            if mask & selectors.EVENT_READ:
                instructions = read_instructions(sock)

                #TODO with instructions (must break loop here
                #if len(instructions) > 0:
                #    start_connection(host,port,sel)
                if instructions and instructions != bytes([222]):
                    print("[client]: instruction received:", instructions)
                    print("[client]: write received instructions to txtfile ...")
                    f = open("instructions.txt", "w+")
                    f.write(str(instructions[0]))
                    f.close()

                    # reload
                    print("[client]: reload world")
                    sel.modify(sock, selectors.EVENT_WRITE, data=key.data)
                    sv = Supervisor()
                    sv.worldReload()
                else:
                    print("[client] quit simulation ")
                    sv = Supervisor()
                    sv.simulationQuit(status=100)


        # Check for a socket being monitored to continue.
        if not sel.get_map():
            break
except KeyboardInterrupt:
    print("caught keyboard interrupt, exiting")
finally:
    sel.close()

