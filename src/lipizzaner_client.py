from distribution.client_api import ClientAPI
from distribution.client_environment import ClientEnvironment
from helpers.network_helpers import is_port_open

DEFAULT_CLIENT_PORT = 5000
MAX_CLIENT_PORT = 5500


class LipizzanerClient:
    def run(self):
        port = DEFAULT_CLIENT_PORT
        while not is_port_open(port):
            port += 1
            if port == MAX_CLIENT_PORT:
                raise IOError('No free port between {} and {} available.'.format(DEFAULT_CLIENT_PORT, MAX_CLIENT_PORT))
        ClientEnvironment.port = port
        ClientAPI().listen(port)
