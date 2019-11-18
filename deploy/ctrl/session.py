#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import copy
import datetime
import uuid
from deploy.utils import MyLock


class ExpireDict(object):
    def __init__(self, max_items=None, expire_sec=None):
        self.max_items = max_items
        self.expire_sec = expire_sec
        self.values = dict({})
        self.lock = MyLock()

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        if key not in self.values.keys():
            raise KeyError
        self.lock.enter()  # in
        val = self.__getitem(key)
        self.lock.leave()  # out
        return val

    def __setitem__(self, key, value):
        self.lock.enter()  # in
        self.__setitem(key, value)
        self.lock.leave()  # out

    def __delitem__(self, key):
        self.lock.enter()  # in
        self.__delitem(key)
        self.lock.leave()  # out

    def __iter__(self):
        return iter(self.values)

    def __str__(self):
        return self.values.__str__()

    def pop_expire(self):
        self.lock.enter()  # in
        ret = {}
        for key in self.__get_expire():
            ret[key] = self.__getitem(key)
            self.__delitem(key)
        self.lock.leave()  # out
        return ret

    def keys(self):
        return self.values.keys()

    def get(self, key, default_value=None):
        try:
            value = self[key]
        except KeyError:
            value = default_value
        return value

    def __getitem(self, key):
        self.values[key][0] = self.__time_stamp()
        return copy.deepcopy(self.values[key][1])

    def __setitem(self, key, value):
        self.values[key] = [self.__time_stamp(), value]

    def __delitem(self, key):
        if key in self.values.keys():
            del self.values[key]

    def __get_expire(self):
        expire_keys = set({})
        if self.max_items is not None:
            if len(self.values) > self.max_items:
                items_sorted = sorted(self.values.items(), key=lambda x: x[1][0])
                expire_keys = expire_keys.union(set([k for k, _ in items_sorted[:self.max_items // 2]]))
            pass

        if self.expire_sec is not None:
            now = self.__time_stamp()
            for key, value in self.values.items():
                if now - value[0] > self.expire_sec:
                    expire_keys.add(key)
            pass

        return expire_keys

    @staticmethod
    def __time_stamp():
        return datetime.datetime.now().timestamp()


class SessionCtrl(object):

    def __init__(self, max_items=None, expire_sec=None):
        self.sessions = ExpireDict(max_items, expire_sec)

    def get_session(self, token):
        return self.sessions[token]

    def has_token(self, token) -> bool:
        return token in self.sessions.keys()

    def new_token(self):
        token = str(uuid.uuid4())
        while token in self.sessions.keys():
            token = str(uuid.uuid4())
        return token


def test():
    '''
    aaa = ExpireDict(max_items=2)
    aaa['a'] = 1
    aaa['a'] = 2
    aaa['b'] = 'sss'
    aaa['c'] = 'sss11'
    print(aaa)
    pp = aaa.pop_expire()
    print(pp)
    print(aaa)
    pp = aaa.pop_expire()
    print(pp)
    print(aaa)
    '''
    aaa = uuid.uuid4()
    print(str(aaa))


if __name__ == '__main__':
    test()
