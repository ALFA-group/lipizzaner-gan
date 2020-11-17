import logging
# import sys
import time 
from threading import Thread

from distribution.node_client import NodeClient
# import LipizzanerMaster, GENERATOR_PREFIX

_logger = logging.getLogger(__name__)

HEARTBEAT_FREQUENCY_SEC = 3


class Heartbeat(Thread):
    def __init__(self, event, kill_clients_on_disconnect, master):
        Thread.__init__(self)
        self.kill_clients_on_disconnect = kill_clients_on_disconnect
        self.stopped = event
        self.master = master # reference to LipizzanerMaster object
        self.success = None
        self.node_client = NodeClient(None)

        _logger.info('HEARTBEAT passed master ref to heartbeat {}'.format(master))

    def run(self):
        # TESTING CODE ONLY
        start_time = 0
        while not self.stopped.wait(HEARTBEAT_FREQUENCY_SEC):
            client_statuses = self.node_client.get_client_statuses()
            # TODO request checkpoint from client using statuses and call master checkpoint func to save new one in output/master
            dead_clients = [c for c in client_statuses if not c['alive'] or not c['busy']]
            alive_clients = [c for c in client_statuses if c['alive'] and c['busy']]

            # TESTING CODE ONLY - here we can pretend that a client has died and try to start up a new cell 

            if dead_clients:
                printable_names = '.'.join([c['address'] + ":" + str(c['port']) for c in dead_clients])
                if self.kill_clients_on_disconnect:
                    _logger.critical('Heartbeat: One or more clients ({}) are not alive anymore; '
                                    'exiting others as well.'.format(printable_names))

                    self.node_client.stop_running_experiments(dead_clients)
                    self.success = False
                    return
                else:
                    _logger.info("Heartbeat: Dead clients {} but will attempt to reconnect to them".format(printable_names)) 
                    # otherwise assume dead and start a new client
                    sleep_time = 5
                    for i in range(3):
                        # could also iterate through clients
                        new_client_statuses = self.node_client.get_client_statuses()
                        new_dead_clients = [c for c in new_client_statuses if not c['alive'] or not c['busy']]
                        
                        resurrected_clients = [c for c in dead_clients if c not in new_dead_clients] 
                        still_dead_clients = [c for c in dead_clients if c in new_dead_clients] 

                        if new_dead_clients == []:
                            _logger.info("Heartbeat: Dead clients {} came back online".format(resurrected_clients))
                            break 
                        else:
                            for dead_client in still_dead_clients:
                                _logger.info("Heartbeat: dead after attempt {} to reconnect {}".format(str(i+1), still_dead_clients))
                            
                        time.sleep(sleep_time)
                        sleep_time += 2
                    
                    # still dead after 3 attempts to reconnect, create new client
                    if still_dead_clients != []:
                        _logger.info('Heartbeat: STILL DEAD after 3 attempts to reconnect. Should create new client')
                        self.master.restart_client(still_dead_clients[0]['port'])

            elif all(c['finished'] for c in alive_clients):
                _logger.info('Heartbeat: All clients finished their experiments.')
                self.success = True
                return
