import socket, pickle
import selectors
import subprocess

def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    print("accepted connection from", addr)
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
            print("blocked")
            break
        if not packet: break
        data.append(packet)
    if len(data) > 0:
        #concat all strings in data list
        data = b''.join(data)
        recv_data = pickle.loads(data)
    return recv_data

def write_instruction(sock, instruction):
    print("write instructions")
    sock.send(instruction)

def close(selector, sock, adrr):
    print("closing connection to", addr)
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
print("listening on", (host, port))
lsock.setblocking(False)
sel.register(lsock, selectors.EVENT_READ, data=None)

#inside experience loop
episode_info = []
iteration = 0
epoch = 0

#create webots instances
child1=subprocess.Popen(["webots --minimize --stdout --batch"], shell=True)
child2=subprocess.Popen(["webots --minimize --stdout --batch"], shell=True)



#experience buffer
ep_buf = []


try:
    while epoch < 2:
        print("epoch", epoch)
        events = sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                accept_wrapper(key.fileobj)
                #should also pass weights at this point
            else:
                sock = key.data["connection"]
                sel = key.data["selector"]
                addr = key.data["address"]

                print("process events", addr)
                if mask & selectors.EVENT_READ:
                    sim_data = read_sim_data(sock)

                    print("data received from", addr)
                    # load episode data into buffer
                    ep_buf.append(sim_data)

                    #count finished episode
                    iteration = iteration + 1
                    print("it:", iteration)
                    sel.modify(sock, selectors.EVENT_WRITE, data=key.data)


                if mask & selectors.EVENT_WRITE:
                    #pass instructions for next simulations
                    if epoch >= 2:
                        instruction = bytes([222])
                    elif iteration > 3:
                        #reload world
                        instruction = bytes([iteration])
                        write_instruction(sock, instruction)

                        iteration = 0
                        epoch = epoch + 1

                    else:
                        #client will reload itself
                        pass
                    close(sel, sock, addr)

except KeyboardInterrupt:
    print("caught keyboard interrupt, exiting")
finally:
    sel.close()


print("buffer:",len(ep_buf), ep_buf)