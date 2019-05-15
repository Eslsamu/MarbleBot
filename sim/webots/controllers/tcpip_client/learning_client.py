import socket, pickle, selectors
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')



class Message():

    def __init__(self, selector, sock, addr, request):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self.request = request

    def process_events(self, mask):
        print("process_events")
        if mask & selectors.EVENT_READ:
            self.read()
        if mask & selectors.EVENT_WRITE:
            self.write()

    def read(self):
        data = []
        while True:
            packet = self.sock.recv(4096)
            if not packet: break
            data.append(packet)
        if len(data) > 0:
            self.process_response(data)

    def write(self):
        self.sock.send(self.request)
        # Set selector to listen for read events, we're done writing.
        events = selectors.EVENT_READ
        self.selector.modify(self.sock, events, data=self)

    def process_response(self, data):
        #TODO:
        #load pickled data
        #either use weights and restart or just restart simulation
        print("data",data)
        pass



def start_connection(host, port, episode_info):
    addr = (host,port)
    print("starting connection to", addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = {"selector": sel,"socket" : sock, "address" : addr, "sim_data" : sim_data}
    sel.register(sock, events, data=message)


sel = selectors.DefaultSelector()
host, port = 'localhost', 50000
sim_data = pickle.dumps(dict(reward= np.ones(1), done=False))
start_connection(host, port, sim_data)

try:
    while True:
        events = sel.select(timeout=1)
        for key, mask in events:
            message = key.data
            message.process_events(mask)

        # Check for a socket being monitored to continue.
        if not sel.get_map():
            break
except KeyboardInterrupt:
    print("caught keyboard interrupt, exiting")
finally:
    sel.close()