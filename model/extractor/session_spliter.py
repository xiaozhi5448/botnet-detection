from collections import deque
from typing import *

import numpy as np

from extractor.common import PacketInfo, TCP_FLAGS, Peer, Session, get_packet_info
from extractor.session_extractor import get_features_from_session

window_size = 10
window_step = 5
session_feature_count = 20


class SessionContainer:
    def __init__(self, window_size: int = 3):
        """
        :param window_size: 每次选择临近的若干个session进行流量判定
        """
        self.sessions: Dict[str, Deque[Session]] = dict()
        self.active_sessions: Dict[str, Session] = Dict()
        self.pkts: Deque[PacketInfo] = deque()
        self.window_size = window_size

    def update_pkts(self, pkts: List[PacketInfo]):
        self.pkts.extend(pkts)
        while self.pkts:
            pkt = self.pkts.popleft()
            session_key = Session.generate_key(pkt, False)
            reverse_session_key = Session.generate_key(pkt, True)
            if session_key in self.active_sessions:
                cur_session = self.active_sessions[session_key]
            elif reverse_session_key in self.active_sessions:
                cur_session = self.active_sessions[reverse_session_key]
            else:
                cur_session = Session()
                self.active_sessions[session_key] = cur_session
            cur_session.add_pkt(pkt)
            if cur_session.is_close():
                addr = cur_session.src.addr
                if addr not in self.sessions:
                    self.sessions[addr] = deque()
                    self.sessions[addr].appendleft(cur_session)
                else:
                    self.sessions[addr].appendleft(cur_session)
                if len(self.sessions[addr]) > self.window_size:
                    self.sessions[addr].popleft()
                if session_key in self.active_sessions and self.active_sessions[session_key] == cur_session:
                    del self.active_sessions[session_key]
                elif reverse_session_key in self.active_sessions and self.active_sessions[
                    reverse_session_key] == cur_session:
                    del self.active_sessions[reverse_session_key]

    def get_features(self):
        """
        获取所有的 主机-特征(基于window-size的大小进行了上下文拼接) 字典
        :return:
        """
        session_dict = dict()
        for session_key in self.active_sessions:
            host = session_key.split(':')[0]
            if host not in session_dict:
                session_dict[host] = deque()
            session_dict[host].append(self.active_sessions[session_key])
        for host in self.sessions:
            if host not in session_dict:
                session_dict[host] = deque()
            session_dict[host].extend(self.sessions[host])
        features_dict = dict()
        for host in session_dict:
            features = [get_features_from_session(session.get_pkts()) + [0] for session in
                        session_dict[host][0: self.window_size]]
            while len(features) < self.window_size:
                features.append(np.zeros(session_feature_count + 1))
            features = np.array(features).ravel()
            features_dict[host] = features
        return features_dict
