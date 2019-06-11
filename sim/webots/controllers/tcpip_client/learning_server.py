import socket, pickle
import selectors
import subprocess
import time

def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    print("[server]: accepted connection from", addr)
    conn.setblocking(False)

    message = {"selector": sel, "connection": conn, "address" : addr}
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=message)

def read_sim_data(sock):
    data = []
    while True:
        try:
            packet = sock.recv(4096)
        except BlockingIOError:
            # Resource temporarily unavailable (errno EWOULDBLOCK)
            print("[server]: blocked")
            break
        if not packet: break
        data.append(packet)
    if len(data) > 0:
        #concat all strings in data list
        data = b''.join(data)
        recv_data = pickle.loads(data)
    return recv_data

def write_instruction(sock, instruction):
    print("[server]: write instructions")
    sock.send(instruction)

def close(selector, sock, adrr):
    print("[server]: closing connection to", addr)
    try:
        selector.unregister(sock)
    except Exception as e:
        print(
            f"error: selector.unregister() exception for",
                f"{addr}: {repr(e)}",
        )

    try:
        sock.close()
    except OSError as e:
        print(
            f"error: socket.close() exception for",
            f"{addr}: {repr(e)}",
        )
    finally:
        # Delete reference to socket object for garbage collection
        sock = None

sel = selectors.DefaultSelector()
host, port = 'localhost', 10000
lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
lsock.bind((host, port))
lsock.listen()
print("[server]: listening on", (host, port))
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
    children.append(subprocess.Popen(["webots --minimize --stdout --batch"], shell=True))



#experience buffer
ep_buf = []



#timer
t = time.time()
try:
    processes_done = 0
    while processes_done < n_processes:
        print("[server]: epoch", epoch)
        events = sel.select(timeout=None)
        print("[server]:", len(events), "events")
        for key, mask in events:
            if key.data is None:
                accept_wrapper(key.fileobj)
                #should also pass weights at this point
            else:
                sock = key.data["connection"]
                sel = key.data["selector"]
                addr = key.data["address"]

                print("[server]: process events", addr)
                if mask & selectors.EVENT_READ:
                    sim_data = read_sim_data(sock)

                    print("[server]: data received from", addr)
                    # load episode data into buffer
                    ep_buf.append(sim_data)

                    sel.modify(sock, selectors.EVENT_WRITE, data=key.data)

                if iteration >= n_iterations:
                    iteration = 0
                    epoch = epoch + 1

                if mask & selectors.EVENT_WRITE:
                    #pass instructions for next simulations
                    if epoch >= n_epochs:
                        print(["[server]: send exit code:", 222])
                        instruction = bytes([222])

                        processes_done += 1
                        print("[server] processes done:", processes_done)
                    else:
                        #count next iteration
                        iteration = iteration + 1
                        print("[server]: iteration:", iteration)

                        # reload world
                        print("[server]: send instruction:", iteration)
                        instruction = bytes([iteration])
                        write_instruction(sock, instruction)

                    close(sel, sock, addr)

except KeyboardInterrupt:
    print("[server]: caught keyboard interrupt, exiting")
finally:
    print("[server]: closing selector")
    sel.close()


print("[server]: buffer:",len(ep_buf), ep_buf)
print("[server]: time:", time.time() - t)


for c in children:
    c.kill()