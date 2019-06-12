import socket, pickle
import selectors
import subprocess
import time
import tensorflow as tf
from rl_policy import actor_critic
from advantage_estimation import Buffer
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s %(message)s',filename='server.log',level=logging.DEBUG)

def accept_wrapper(sock, sel):
    conn, addr = sock.accept()  # Should be ready to read
    logging.info("[server]: accepted connection from" + str(addr))
    conn.setblocking(False)

    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events)

def read_episode_data(sock):
    data = []
    while True:
        try:
            packet = sock.recv(4096)
        except BlockingIOError:
            # Resource temporarily unavailable (errno EWOULDBLOCK)
            logging.info("[server]: blocked")
            break
        if not packet: break
        data.append(packet)
    if len(data) > 0:
        #concat all strings in data list
        data = b''.join(data)
        recv_data = pickle.loads(data)
    return recv_data

def write_instruction(sock, instruction):
    logging.info("[server]: write instructions")
    instruction = pickle.dumps(instruction)
    totalsent = 0
    while totalsent < len(instruction):
        sent = sock.send(instruction[totalsent:])
        totalsent += sent
        if sent == 0:
            logging.info("[server]: socket connection broken")

def close(selector, sock):
    try:
        selector.unregister(sock)
    except Exception as e:
        logging.info(
            f"error: selector.unregister() exception for",
                f"{sock}: {repr(e)}",
        )

    try:
        sock.close()
    except OSError as e:
        logging.info(
            f"error: socket.close() exception for",
            f"{sock}: {repr(e)}",
        )
    finally:
        # Delete reference to socket object for garbage collection
        sock = None

def setup_selector():
    sel = selectors.DefaultSelector()
    host, port = 'localhost', 10002
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.bind((host, port))
    lsock.listen()
    logging.info("[server]: listening on " + str(host) + str(port))
    lsock.setblocking(False)
    sel.register(lsock, selectors.EVENT_READ)
    return sel

def create_webots_instances(n_instances = 2):
    children = []
    for i in range(n_instances):
        children.append(
            subprocess.Popen(["webots --minimize --stdout --stderr --batch ../../worlds/speed_test.wbt"], shell=True))
    return children


def run_job(n_instances = 2, n_iterations = 3, n_steps=10):
    sel = setup_selector()
    instances = create_webots_instances(n_instances)
    epoch_data = []
    try:
        processes_done = 0
        while processes_done < n_instances:
            events = sel.select(timeout=None)
            for key, mask in events:
                sock = key.fileobj
                if key.data is None:
                    accept_wrapper(sock, sel)
                else:

                    if mask & selectors.EVENT_READ:
                        ep_data = read_episode_data(sock)
                        epoch_data += ep_data
                        sel.modify(sock, selectors.EVENT_WRITE, data=key.data)

                    if mask & selectors.EVENT_WRITE:
                        #pass instructions for next episode
                        if iteration >= n_iterations:
                            cont = False
                            processes_done += 1
                        else:
                            #count next iteration
                            iteration = iteration + 1
                            cont = True
                        write_instruction(sock, instruction= cont)
                        close(sel, sock)
    except KeyboardInterrupt:
        logging.info("[server]: caught keyboard interrupt, exiting")
    finally:
        logging.info("[server]: closing selector")
        sel.close()

    return epoch_data



"""
 #should not be necessary
    #TODO check if removing changes anything
    for i in instances:
        i.kill()

load all episode timesteps into buffer and 
compute advantages + reward to go
"""
#buffer.process_epoch(ep_data=ep_data)