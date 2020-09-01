import tempfile

import codecs
import torch


class StateEncoder:
    """
    Class to en- and decode the state dictionaries of PyTorch optimizers to base64 strings.
    TODO: Refactor to use BytesIO when PyTorch 0.4.0 is released, as torch.save() currently does not support it.
    """

    @staticmethod
    def encode(state_dict):
        with tempfile.TemporaryFile() as tmp:
            torch.save(state_dict, tmp)
            tmp.flush()
            tmp.seek(0)
            return codecs.encode(tmp.read(), "base64").decode()

    @staticmethod
    def decode(data):
        with tempfile.TemporaryFile() as tmp:
            tmp.write(codecs.decode(data.encode(), "base64"))
            tmp.flush()
            tmp.seek(0)
            return torch.load(tmp)
