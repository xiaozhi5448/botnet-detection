from datetime import datetime
from typing import *
from extractor.common import *
from collections import Counter
from extractor.session_spliter import SessionContainer
from extractor.window_extactor import extract_window_features

class WindowEvaluator:
    def __init__(self, clf):
        self.clf = clf
        self.update_time = None

    def evaluate(self, pkts: List[Packet]):
        pkt_info_list = [get_packet_info(pkt) for pkt in pkts]
        host_pkts = defaultdict(list)
        for pktinfo in pkt_info_list:
            if pktinfo.src in host_pkts:
                host_pkts[pktinfo.src].append(pktinfo)
            elif pktinfo.dst in host_pkts:
                host_pkts[pktinfo.dst].append(pktinfo)
            else:
                host_pkts[pktinfo.src].append(pktinfo)
        features = [extract_window_features(items) for items in host_pkts.values()]
        y_pred = self.clf.predict(features)
        res = defaultdict(int)
        for i, host in enumerate(host_pkts.keys()):
            hosts = defaultdict(int)
            for pkt in host_pkts[host]:
                hosts[pkt.src] += 1
                hosts[pkt.dst] += 1
            most_freq_host = Counter(hosts).most_common(1)[0][0]
            res[most_freq_host] = y_pred[i]
        return res


