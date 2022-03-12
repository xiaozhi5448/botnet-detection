import logging
from typing import *
from enum import Enum
import tldextract
from scapy.all import *
from scapy.layers.dns import DNSQR, DNSRR, DNS
from scapy.layers.inet import IP, TCP, UDP
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)

session_timeout = 300


class TCP_FLAGS(Enum):
    FIN = 0X01
    SYN = 0x02
    RST = 0x04
    PSH = 0x08
    ACK = 0x10
    URG = 0x20
    ECE = 0x40
    CWR = 0x80


tcp_flags = {
    'FIN': 0x01,
    'SYN': 0x02,
    'RST': 0x04,
    'PSH': 0x08,
    'ACK': 0x10,
    'URG': 0x20,
    'ECE': 0x40,
    'CWR': 0x80
}


class PacketInfo:
    def __init__(self):
        self.src = ''
        self.sport = 0
        self.dst = ''
        self.dport = 0
        self.time: datetime = 0
        self.size = 0
        self.payload_size = 0
        self.transport_layer = 'tcp'
        self.flags = 0
        self.dns_info = None

    def __str__(self) -> str:
        return '{}:{}->{}:{} {} {}'.format(
            self.src, self.sport, self.dst, self.dport, self.transport_layer, self.size
        )

    def __repr__(self) -> str:
        return self.__str__()


class SessionState(Enum):
    SYN_SENT = 0
    SYN_RECV = 1
    ESTABLISHED = 2
    FIN_WAIT_1 = 3
    CLOSE_WAIT = 4
    FIN_WAIT_2 = 5
    LAST_ACK = 6
    CLOSED = 7


class Peer:
    def __init__(self, addr: str, port: int):
        self.addr = addr
        self.port = port

    def __eq__(self, p):
        if type(p) is not Peer:
            return False
        return self.addr == p.addr and self.port == p.port


class Session:

    @classmethod
    def generate_key(cls, pkt: PacketInfo, reverse=False):
        if reverse:
            return '{}:{}->{}:{}'.format(pkt.dst, pkt.dport, pkt.src, pkt.sport)
        else:
            return '{}:{}->{}:{}'.format(pkt.src, pkt.sport, pkt.dst, pkt.dport)


    def __init__(self):
        self.pkts: Deque[PacketInfo] = deque()
        self.src: Peer = None
        self.dst: Peer = None
        self.start_time: datetime = None
        self.end_time: datetime = None
        self.duration: float = 0
        self.update_time: datetime = None
        self.src_state = SessionState.CLOSED
        self.dst_state = SessionState.CLOSED

    def first_packet(self, pktinfo: PacketInfo):
        self.src = Peer(pktinfo.src, pktinfo.sport)
        self.dst = Peer(pktinfo.dst, pktinfo.dport)
        self.update_time = pktinfo.time
        self.start_time = pktinfo.time
        # normal create connection
        if pktinfo.flags & TCP_FLAGS.SYN.value and not pktinfo.flags & TCP_FLAGS.ACK.value:
            self.src_state = SessionState.SYN_SENT
            self.dst_state = SessionState.SYN_RECV
        else:
            # 异常情况,session尚未创建时收到包,直接创建该session
            self.src_state = SessionState.ESTABLISHED
            self.dst_state = SessionState.ESTABLISHED

    def is_include(self, pkt: PacketInfo) -> bool:
        if (pkt.time - self.update_time).total_seconds() > session_timeout:
            return False
        return True

    def get_pkts(self) -> Deque[PacketInfo]:
        return self.pkts

    def add_pkt(self, pkt: PacketInfo):
        self.pkts.append(pkt)
        if len(self.pkts) == 0:
            self.first_packet(pkt)
        else:
            self.update_state(pkt)

    def get_dest(self, pkt: PacketInfo):
        if pkt.dst == self.src.addr and pkt.dport == self.src.port:
            return 'src'
        elif pkt.dst == self.dst.addr and pkt.dport == self.dst.port:
            return 'dst'
        return 'unknown'

    def is_close(self):
        return self.src_state == SessionState.CLOSED and self.dst_state == SessionState.CLOSED

    def update_state(self, pkt: PacketInfo):
        dest = self.get_dest(pkt)
        # close connection
        if pkt.flags & TCP_FLAGS.RST.value:
            self.src_state = SessionState.CLOSED
            self.dst_state = SessionState.CLOSED
        # 三次握手
        elif dest == 'dst' and pkt.flags & TCP_FLAGS.SYN.value and pkt.flags & TCP_FLAGS.ACK.value:
            self.src_state = SessionState.ESTABLISHED
        elif dest == 'src' and pkt.flags & TCP_FLAGS.ACK.value and self.dst_state == SessionState.SYN_RECV:
            self.dst_state = SessionState.ESTABLISHED
        # 四次
        elif pkt.flags & TCP_FLAGS.FIN.value:
            if dest == 'src':
                if self.src_state == SessionState.ESTABLISHED:
                    self.src_state = SessionState.FIN_WAIT_1
                elif self.src_state == SessionState.CLOSE_WAIT:
                    self.src_state = SessionState.LAST_ACK
            else:
                if self.dst_state == SessionState.ESTABLISHED:
                    self.dst_state = SessionState.FIN_WAIT_1
                elif self.dst_state == SessionState.CLOSE_WAIT:
                    self.dst_state = SessionState.LAST_ACK
        elif pkt.flags & TCP_FLAGS.ACK.value:

            if self.dst_state == SessionState.FIN_WAIT_1 and dest == 'dst':
                self.src_state = SessionState.CLOSE_WAIT
                self.dst_state = SessionState.FIN_WAIT_2
            elif self.src_state == SessionState.FIN_WAIT_1 and dest == 'src':
                self.dst_state = SessionState.CLOSE_WAIT
                self.src_state = SessionState.FIN_WAIT_2

            if self.dst_state == SessionState.LAST_ACK and dest == 'dst':
                self.dst_state = SessionState.CLOSED
                self.src_state = SessionState.CLOSED
            elif self.src_state == SessionState.LAST_ACK and dest == 'src':
                self.dst_state = SessionState.CLOSED
                self.src_state = SessionState.CLOSED



def get_packet_info(pkt: Packet):
    if TCP not in pkt and UDP not in pkt:
        return None
    pkt_info = PacketInfo()
    pkt_info.src = pkt[IP].src
    pkt_info.dst = pkt[IP].dst
    pkt_info.time = datetime.fromtimestamp(float(pkt.time))
    pkt_info.size = len(pkt)
    layer = TCP if TCP in pkt else UDP
    pkt_info.sport = pkt[layer].sport
    pkt_info.dport = pkt[layer].dport
    pkt_info.payload_size = len(pkt[layer].payload)
    if TCP in pkt:
        pkt_info.transport_layer = 'tcp'
        pkt_info.flags = pkt[TCP].flags
        pkt_info.dns_info = None
    else:
        pkt_info.transport_layer = 'udp'
        pkt_info.dns_info = {
            'type': 'qr',
            'domain': '',
            'ip_addrs': []
        }
        if DNSQR in pkt:
            pkt_info.dns_info['type'] = 'qr'
            pkt_info.dns_info['domain'] = pkt[DNSQR].qname.decode('utf-8')
        elif DNSRR in pkt:
            pkt_info.dns_info['type'] = 'rr'
            pkt_info.dns_info['domain'] = pkt[DNSRR].rrname.decode('utf-8')
            for i in range(pkt[DNS].ancount):
                ip_addr = pkt[DNS].an[i].rdata
                pkt_info.dns_info['ip_addrs'].append(ip_addr)
        else:
            pkt_info.dns_info = dict()
    return pkt_info
