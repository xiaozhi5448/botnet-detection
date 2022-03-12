import fnmatch
import glob

from numpy import ndarray
from scapy.all import *
import os
import uuid
from extractor.common import PacketInfo, tcp_flags
from collections import Counter, namedtuple
from typing import *
import numpy as np
import logging
import argparse

logging.basicConfig(level=logging.INFO)

FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20
ECE = 0x40
CWR = 0x80


def iopr(pkts: List[PacketInfo]) -> float:
    src = pkts[0].src
    out_pkt_count = len([pkt for pkt in pkts if pkt.src == src])
    in_pkt_count = len([pkt for pkt in pkts if pkt.dst == src])
    if in_pkt_count != 0:
        return out_pkt_count / in_pkt_count
    else:
        return 100


def fps(pkts: List[PacketInfo]) -> int:
    if not pkts:
        return 0
    for pkt in pkts[2:]:
        if pkt.flags & tcp_flags['ACK'] and not pkt.flags & tcp_flags['SYN'] and pkt.size > 60:
            return pkt.size


def dpl(pkts: List[PacketInfo]) -> float:
    if not pkts:
        return 0
    counter = Counter([pkt.payload_size for pkt in pkts])
    freq = counter.most_common(1)[0][1]
    return freq / len(pkts)


def psp(pkts):
    if not pkts:
        return 0
    return len([pkt for pkt in pkts if 63 < pkt.size < 400]) / len(pkts)


def apl(pkts: List[PacketInfo]):
    if not pkts:
        return 0
    return sum((pkt.payload_size for pkt in pkts)) / len(pkts)


def abps(pkts: List[PacketInfo]):
    if not pkts:
        return 0
    duration = (pkts[-1].time - pkts[0].time).total_seconds()
    if duration == 0:
        return len(pkts)
    return sum([pkt.size for pkt in pkts]) / duration


def pps(pkts: List[PacketInfo]):
    if not pkts:
        return 0
    duration = (pkts[-1].time - pkts[0].time).total_seconds()
    if duration == 0:
        return len(pkts)
    return len(pkts) / duration


feature_keys = [
    'src', 'sport', 'dst', 'dport', 'start_time', 'end_time', 'px', 'nnp', 'psp', 'iopr', 'duration', 'fps', 'tbt',
    'apl', 'dpl', 'pv', 'abps', 'pps', 'ipc', 'opc'
]
headers = ','.join(feature_keys) + ',bpr'
features = {
    'src': lambda pkts: pkts[0].src,
    'sport': lambda pkts: pkts[0].sport,
    'dst': lambda pkts: pkts[0].dst,
    'dport': lambda pkts: pkts[0].dport,
    'start_time': lambda pkts: float(pkts[0].time),
    'end_time': lambda pkts: float(pkts[-1].time),
    'px': lambda pkts: len(pkts),  # total number of packets exchanged
    # number of null packets
    'nnp': lambda pkts: len([pkt for pkt in pkts if pkt.flags == 0]),
    'psp': psp,  # percentage of small package
    'iopr': iopr,  # incoming packets over the outgoing packets
    'duration': lambda pkts: float((pkts[-1].time - pkts[0].time).real),
    'fps': fps,  # first packet size
    'tbt': lambda pkts: sum([len(pkt) for pkt in pkts]),  # total bytes
    'apl': apl,  # average packet payload length
    'dpl': dpl,
    'pv': lambda pkts: np.array([len(pkt) for pkt in pkts]).std(),  # 数据包长度的标准差
    'abps': abps,  # 平均每秒发送的字节数目,
    'pps': pps,  # 平均每秒发送的数据包数目
    # 发送的数据包数量
    'ipc': lambda pkts: len([pkt for pkt in pkts if pkt.src == pkts[0].src]),
    # 接受的数据包数量
    'opc': lambda pkts: len([pkt for pkt in pkts if pkt.src == pkts[0].dst]),
}


def get_features_from_session(pkts: List[PacketInfo]) -> ndarray:
    return np.array(list(features[name](pkts) for name in features))
