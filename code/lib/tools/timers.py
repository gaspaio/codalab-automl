""" Timer registry."""

from timeit import default_timer


class TimerRegistry(object):

    """ Manage an ensemble of timers. """

    def __init__(self):
        self.timers_ = {}

    def start(self, *args):
        for name in args:
            if name not in self.timers_:
                self.timers_[name] = Timer(name)
            self.timers_[name].start()

    def stop(self, *args):
        for name in args:
            if name not in self.timers_:
                raise ValueError("Unknown timer '{}'".format(name))
            self.timers_[name].stop()

    def get(self, name):
        if name not in self.timers_:
            raise ValueError("Unknown timer '{}'".format(name))

        return self.timers_[name].get()

    def getTimer(self, name):
        if name not in self.timers_:
            raise ValueError("Unknown timer '{}'".format(name))

        return self.timers_[name]

    def getTimers(self):
        return self.timers_


class Timer(object):

    """ Create a simple timer. """

    STATE_NOT_STARTED = "not_started"
    STATE_RUNNING = "running"
    STATE_STOPPED = "stopped"

    def __init__(self, name='default'):
        self.timer = default_timer
        self.name = name
        self.status = self.STATE_NOT_STARTED
        self.end_ = self.start_ = None
        self.elapsed_ = 0

    def start(self):
        self.start_ = self.timer()
        self.end_ = None
        self.elapsed_ = 0
        self.status = self.STATE_RUNNING

    def stop(self):
        self.end_ = self.timer()
        self.elapsed_ = self.end_ - self.start_

    def get(self):
        if self.STATE_RUNNING == self.status:
            self.elapsed_ = self.timer() - self.start_
        return self.elapsed_
