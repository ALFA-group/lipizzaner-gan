"""
Helper to wait for two events simultanously.
Taken from https://stackoverflow.com/a/12320352
"""
import threading


def or_set(self):
    self._set()
    self.changed()


def or_clear(self):
    self._clear()
    self.changed()


def orify(e, changed_callback):
    e._set = e.set
    e._clear = e.clear
    e.changed = changed_callback
    e.set = lambda: or_set(e)
    e.clear = lambda: or_clear(e)


def or_event(*events):
    event = threading.Event()

    def changed():
        bools = [e.is_set() for e in events]
        if any(bools):
            event.set()
        else:
            event.clear()

    for e in events:
        orify(e, changed)
    changed()
    return event
