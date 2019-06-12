import os
os.environ["PYTHONUNBUFFERED"] = "1"

from controller import Supervisor
from controller import Robot
import socket, pickle, selectors
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
import logging
logging.basicConfig(format='%(asctime)s %(message)s',filename='client.log',level=logging.DEBUG)

def start_connection(host, port, sel):
    addr = (host,port)
    logging.info("[client]: starting connection to"+ str(addr))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(sock, events)

def write_sim_data(sock, sim_data):
    sim_data = pickle.dumps(sim_data)
    logging.info("[client]: write sim data")
    sock.send(sim_data)
    
def read_instructions(sock):
    data = []
    while True:
        try:
            packet = sock.recv(4096)
        except BlockingIOError:
            # Resource temporarily unavailable (errno EWOULDBLOCK)
            logging.info("[client]: socket blocked")
            break
        if not packet: break
        data.append(packet)
    if len(data) > 0:
        # concat all strings in data list
        data = b''.join(data)
        recv_data = pickle.loads(data)
        return recv_data

def run_simulation(steps):
    sim_data = []
    robot = sv.getFromDef("robot")
    field = robot.getField("translation")
    
    for s in range(steps):
        sv.step(32)
        sim_data.append(field.getSFVec3f())

    return sim_data


def message_server(sel, sim_data):
    try:
        while True:
            logging.info("[client] waiting for connection")
            events = sel.select(timeout=None)
            for key, mask in events:
                sock = key.fileobj

                if mask & selectors.EVENT_WRITE:
                    write_sim_data(sock, sim_data)
                    sel.modify(sock, selectors.EVENT_READ, data=key.data)

                if mask & selectors.EVENT_READ:
                    instructions = read_instructions(sock)

                    if instructions and instructions != bytes([222]):
                        logging.info("[client]: instruction received:" + str(instructions))
                        logging.info("[client]: write received instructions to txtfile ...")
                        f = open("instructions.txt", "w+")
                        f.write(str(instructions))
                        f.close()

                        # reload
                        logging.info("[client]: reload world")
                        sv.worldReload()
                        return "reloaded"
                    else:
                        logging.info("[client] quit simulation ")
                        sv.simulationQuit(status=100)
                        return "quit"

            # Check for a socket being monitored to continue.
            if not sel.get_map():
                break
    except KeyboardInterrupt:
        logging.info("caught keyboard interrupt, exiting")
    finally:
        logging.info("[client]: close selector")
        sel.close()
    return "interrupted"


logging.info("[client]: simulation is started")

sv = Supervisor()
sel = selectors.DefaultSelector()
host, port = 'localhost', 10002
sim_data = run_simulation(steps = 10)
start_connection(host, port, sel)
done = message_server(sel, sim_data)
logging.info("[client] " +str(done))