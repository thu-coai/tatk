#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import multiprocessing as mp
import threading as th


class LockBase(object):
    def __init__(self):
        self.locks = []

    def enter(self):
        for lock in self.locks:
            lock.acquire()

    def leave(self):
        for lock in self.locks[::-1]:  # Release in inverse order of acquisition
            lock.release()

    def __enter__(self):
        # print("enter")
        self.enter()

    def __exit__(self, *exc):
        self.leave()
        # print("leave")


class GlobalLock(LockBase):
    def __init__(self, name: str = 'lock', is_rlock=False):
        super().__init__()
        cls = type(self)
        mp_name = 'mp_' + name
        th_name = 'th_' + name
        cls.create_mp_lock(mp_name, is_rlock)
        cls.create_th_lock(th_name, is_rlock)
        self.locks = [lk for lk in [getattr(cls, mp_name), getattr(cls, th_name)] if lk is not None]

    @classmethod
    def create_mp_lock(cls, name, is_rlock):
        if not hasattr(cls, name):
            try:
                if is_rlock:
                    setattr(cls, name, mp.RLock())  # multiprocessing lock
                else:
                    setattr(cls, name, mp.Lock())  # multiprocessing lock
            except ImportError:  # pragma: no cover
                setattr(cls, name, None)
            except OSError:  # pragma: no cover
                setattr(cls, name, None)

    @classmethod
    def create_th_lock(cls, name, is_rlock):
        if not hasattr(cls, name):
            try:
                if is_rlock:
                    setattr(cls, name, th.RLock())  # thread lock
                else:
                    setattr(cls, name, th.Lock())  # thread lock
            except OSError:  # pragma: no cover
                setattr(cls, name, None)


class GlobalSemaphore(LockBase):
    def __init__(self, value: int = 1, name: str = 'sem'):
        super().__init__()
        cls = type(self)
        mp_name = 'mp_' + name
        th_name = 'th_' + name
        cls.create_mp_sem(mp_name, value)
        cls.create_th_sem(th_name, value)
        self.locks = [lk for lk in [getattr(cls, mp_name), getattr(cls, th_name)] if lk is not None]

    @classmethod
    def create_mp_sem(cls, name, value):
        if not hasattr(cls, name):
            try:
                setattr(cls, name, mp.Semaphore(value))  # multiprocessing lock
            except ImportError:  # pragma: no cover
                setattr(cls, name, None)
            except OSError:  # pragma: no cover
                setattr(cls, name, None)

    @classmethod
    def create_th_sem(cls, name, value):
        if not hasattr(cls, name):
            try:
                setattr(cls, name, th.Semaphore(value))  # thread lock
            except OSError:  # pragma: no cover
                setattr(cls, name, None)


class MyLock(LockBase):
    def __init__(self, is_rlock=False):
        super().__init__()
        mp_lock = self.create_mp_lock(is_rlock)
        th_lock = self.create_th_lock(is_rlock)
        self.locks = [lk for lk in [mp_lock, th_lock] if lk is not None]

    @staticmethod
    def create_mp_lock(is_rlock):
        try:
            if is_rlock:
                mp_lock = mp.RLock()  # multiprocessing lock
            else:
                mp_lock = mp.Lock()  # multiprocessing lock
        except ImportError:  # pragma: no cover
            mp_lock = None
        except OSError:  # pragma: no cover
            mp_lock = None
        return mp_lock

    @staticmethod
    def create_th_lock(is_rlock):
        try:
            if is_rlock:
                th_lock = th.RLock()  # thread lock
            else:
                th_lock = th.Lock()  # thread lock
        except OSError:  # pragma: no cover
            th_lock = None
        return th_lock


class MySemaphore(LockBase):
    def __init__(self, value: int = 1):
        super().__init__()
        mp_sem = self.create_mp_sem(value)
        th_sem = self.create_th_sem(value)
        self.locks = [lk for lk in [mp_sem, th_sem] if lk is not None]

    @staticmethod
    def create_mp_sem(value):
        try:
            mp_lock = mp.Semaphore(value)  # multiprocessing lock
        except ImportError:  # pragma: no cover
            mp_lock = None
        except OSError:  # pragma: no cover
            mp_lock = None
        return mp_lock

    @staticmethod
    def create_th_sem(value):
        try:
            th_lock = th.Semaphore(value)  # thread lock
        except OSError:  # pragma: no cover
            th_lock = None
        return th_lock


class DeployError(Exception):
    def __init__(self, msg: str, module: str = 'system', model: str = ''):
        super().__init__(self)
        self.msg = msg
        self.module = module
        self.model = model

    def __str__(self):
        text = ''
        if self.module:
            text += '[%s]' % self.module
        if self.model:
            text += '<%s>' % self.model
        return text + ' ' + self.msg


def test():
    lock = GlobalLock(is_rlock=False)
    with lock:
        print("running")
        with lock:
            print("running")



if __name__ == '__main__':
    test()
