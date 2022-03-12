import logging
from typing import *
from collections import defaultdict
import tldextract
from scapy.all import *
import numpy as np

logging.basicConfig(level=logging.INFO)
from extractor.common import PacketInfo, get_packet_info, tcp_flags

logging.basicConfig(level=logging.INFO)

window_size = 10
window_step = 5


def tbt(pkts: Deque[PacketInfo]):
    res = 0
    for pkt in pkts:
        res += pkt.payload_size
    return res


def udpr(pkts: Deque[PacketInfo]):
    destinations = set()
    sources = set()
    for pkt in pkts:
        sources.add(pkt.src)
        destinations.add('{}:{}'.format(pkt.dst, pkt.dport))
    return len(destinations) / len(sources)


def uspr(pkts: Deque[PacketInfo]):
    sources = set()
    ip_addrs = set()
    for pkt in pkts:
        ip_addrs.add(pkt.src)
        sources.add('{}:{}'.format(pkt.src, pkt.sport))
    return len(sources) / len(ip_addrs)


def dpl(pkts: List[PacketInfo]):
    """
    最大的数据包的数量占比
    :param pkts:
    :return:
    """
    c = Counter([pkt.payload_size for pkt in pkts])
    most_common_count = c.most_common(1)[0][1]
    return most_common_count / len(pkts)


def dpl2(pkts: List[PacketInfo]):
    """
    不同的数据包大小的个数
    :param pkts:
    :return:
    """
    c = Counter([pkt.payload_size for pkt in pkts])
    return len(c) / len(pkts)


def dns_rate(pkts: List[PacketInfo]):
    """
    dns 查询的比例
    :param pkts:
    :return:
    """
    count = len([pkt for pkt in pkts if pkt.dns_info])
    return count / len(pkts)


def dnc(pkts: List[PacketInfo]):
    """查询到的不同域名个数"""
    dns_pkts = [pkt for pkt in pkts if pkt.dns_info]
    domains = []
    for pkt in dns_pkts:
        domains.append(pkt.dns_info['domain'])
    return len(domains)


def fdnc(pkts: List[PacketInfo]):
    # 失败的dns查询比率
    # 找到应答报文
    dns_rr_pkts = [pkt for pkt in pkts if pkt.dns_info and pkt.dns_info['type'] == 'rr']
    return len([pkt for pkt in dns_rr_pkts if len(pkt.dns_info['ip_addrs']) == 0]) / len(dns_rr_pkts)


def dor(pkts: List[PacketInfo]):
    """查询最频繁的域名占所有dns查询的比率"""
    sub_domains = defaultdict(int)
    for pkt in pkts:
        if not pkt.dns_info or pkt.dns_info['type'] == 'rr':
            continue
        d = tldextract.extract(pkt.dns_info['domain'])
        name = d.domain + '.' + d.suffix
        sub_domains[name] += 1
    count = Counter(sub_domains).most_common(1)[0][1]
    return count / sum(sub_domains.values())


feature_names = ['tbt', 'apl', 'udpr', 'uspr', 'pps', 'bps', 'mp', 'dpl', 'pv', 'px', 'nnp', 'nsp', 'mpl', 'ttf', 'fin',
                 'rst', 'psh', 'ack', 'syn', 'dns_rate', 'dnc', 'fdnc']

features = {
    'tbt': tbt,  # total number of bytes
    'apl': lambda pkts: tbt(pkts) / len(pkts),  # average payload length
    'udpr': udpr,
    'uspr': uspr,
    'pps': lambda pkts: len(pkts) / window_size,  # average pkts per second
    'bps': lambda pkts: tbt(pkts) / window_size,  # average bits per second
    # the number of maximum packet
    'mp': lambda pkts: Counter([pkt.payload_size for pkt in pkts]).most_common(1)[0][1] / len(pkts),
    'dpl': dpl,  # packets with same length over the total number of packets
    'dpl2': dpl2,
    # 数据包长度的标准差,
    'pv': lambda pkts: np.array([pkt.size for pkt in pkts]).std(),
    'px': lambda pkts: len(pkts),  # '交换的数据包数量'
    # null pkts
    'nnp': lambda pkts: len([pkt for pkt in pkts if pkt.transport_layer == 'tcp' and pkt.flags == 0]),
    'nsp': lambda pkts: len([pkt for pkt in pkts if pkt.payload_size < 340]) / len(pkts),
    # maximum length of packet
    'mpl': lambda pkts: max([pkt.size for pkt in pkts]),
    'ttf': lambda pkts: len([pkt for pkt in pkts if
                             pkt.transport_layer == 'tcp' and pkt.flags & tcp_flags['SYN'] and not pkt.flags &
                                                                                                   tcp_flags['ACK']]),
    # connection count
    # fin flags
    'fin': lambda pkts: len(
        [pkt for pkt in pkts if pkt.transport_layer == 'tcp' and pkt.flags & tcp_flags['FIN']]) / len(pkts),
    # RST flags
    'rst': lambda pkts: len(
        [pkt for pkt in pkts if pkt.transport_layer == 'tcp' and pkt.flags & tcp_flags['RST']]) / len(pkts),
    'psh': lambda pkts: len(
        [pkt for pkt in pkts if pkt.transport_layer == 'tcp' and pkt.flags & tcp_flags['PSH']]) / len(pkts),
    'ack': lambda pkts: len(
        [pkt for pkt in pkts if pkt.transport_layer == 'tcp' and pkt.flags & tcp_flags['ACK']]) / len(pkts),
    'syn': lambda pkts: len(
        [pkt for pkt in pkts if pkt.transport_layer == 'tcp' and pkt.flags & tcp_flags['SYN']]) / len(pkts),
    'dns_rate': dns_rate,
    'dnc': dnc,
    'fdnc': fdnc,
    'dor': dor
}


def extract_window_features(pkts: List[PacketInfo]) -> List[float]:
    res = []
    for name in feature_names:
        res.append(features[name](pkts))
    return res
