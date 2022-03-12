from datetime import datetime
from typing import *
from extractor.common import *
from extractor.session_spliter import SessionContainer


class SessionEvaluator:
    def __init__(self, clf):
        self.clf = clf
        self.update_time = None
        self.session_ctx = SessionContainer()

    def evaluate(self, pkts: List[Packet]):
        pkt_info_list = [get_packet_info(pkt) for pkt in pkts]
        self.update_time = datetime.now()
        self.session_ctx.update_pkts(pkt_info_list)
        features = self.session_ctx.get_features()
        res = dict()
        y_pred = self.clf.predict(list(features.values()))
        for i, host in enumerate(features.keys()):
            res[host] = y_pred[i]
        return res
