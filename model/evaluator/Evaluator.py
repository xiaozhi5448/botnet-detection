from scapy.all import *
from extractor.common import get_packet_info, PacketInfo, Peer, Session
from extractor.session_extractor import get_features_from_session
from extractor.window_extactor import extract_window_features
from extractor.session_spliter import SessionContainer

class Evaluator:
    def __init__(self, clf):
        self.classfier = clf

    def evaluate(self, pkts: List[Packet]):
        info_list = [get_packet_info(pkt) for pkt in pkts]

