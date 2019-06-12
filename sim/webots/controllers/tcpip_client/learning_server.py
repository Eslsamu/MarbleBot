import socket, pickle
import selectors
import subprocess
import time
import logging
logging.basicConfig(format='%(asctime)s %(message)s',filename='server.log',level=logging.DEBUG)

def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    logging.info("[server]: accepted connection from" + str(addr))
    conn.setblocking(False)
    accepted = True
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=accepted)

def read_sim_data(sock):
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

def close(selector, sock, addr=1):
    logging.info("[server]: closing connection to" + str(addr))
    try:
        selector.unregister(sock)
    except Exception as e:
        logging.info(
            f"error: selector.unregister() exception for",
                f"{addr}: {repr(e)}",
        )

    try:
        sock.close()
    except OSError as e:
        logging.info(
            f"error: socket.close() exception for",
            f"{addr}: {repr(e)}",
        )
    finally:
        # Delete reference to socket object for garbage collection
        sock = None

sel = selectors.DefaultSelector()
host, port = 'localhost', 10002
lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
lsock.bind((host, port))
lsock.listen()
logging.info("[server]: listening on " + str(host) + str(port))
lsock.setblocking(False)
sel.register(lsock, selectors.EVENT_READ, data=None)

#inside experience loop
episode_info = []
n_iterations = 3
iteration = 0
epoch = 0
n_epochs = 2

#create webots instances
n_processes = 2
children = []
for i in range(n_processes):
    children.append(subprocess.Popen(["webots --minimize --stdout --stderr --batch ../../worlds/speed_test.wbt"], shell=True))



#experience buffer
ep_buf = []


#timer
t = time.time()
try:
    processes_done = 0
    while processes_done < n_processes:
        logging.info("[server]: epoch" + str(epoch))
        events = sel.select(timeout=None)
        logging.info("[server]:"+ str(len(events))+ "events")
        for key, mask in events:
            sock = key.fileobj
            if key.data is None:
                accept_wrapper(sock)
            else:
                if mask & selectors.EVENT_READ:
                    sim_data = read_sim_data(sock)

                    # load episode data into buffer
                    ep_buf.append(sim_data)

                    sel.modify(sock, selectors.EVENT_WRITE, data=key.data)

                if iteration >= n_iterations:
                    iteration = 0
                    epoch = epoch + 1

                if mask & selectors.EVENT_WRITE:
                    #pass instructions for next simulations
                    if epoch >= n_epochs:
                        logging.info('[server]: send exit code:' + str(222))
                        instruction = bytes([222])

                        processes_done += 1
                        logging.info("[server] processes done:" + str(processes_done))
                    else:
                        #count next iteration
                        iteration = iteration + 1
                        logging.info("[server]: iteration:" + str(iteration))

                        # reload world
                        logging.info("[server]: send instruction:" + str(iteration))
                        instruction = [epoch, iteration]
                        write_instruction(sock, instruction)

                    close(sel, sock)

except KeyboardInterrupt:
    logging.info("[server]: caught keyboard interrupt, exiting")
finally:
    logging.info("[server]: closing selector")
    sel.close()


logging.info("[server]: buffer:" + str(len(ep_buf)) + str(ep_buf))
logging.info("[server]: time:" + str(time.time() - t))


for c in children:
    c.kill()