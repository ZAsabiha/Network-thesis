import csv
import os
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub

class TrafficCollector(app_manager.RyuApp):
    OF_VERSION = [ofproto_v1_3.OFP_VERSION]  # <-- Fixed!

    def __init__(self, *args, **kwargs):
        super(TrafficCollector, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor_loop)
        
        # Dictionary to store previous stats to calculate delta changes (for throughput)
        # Structure: {(datapath_id, port_no): (prev_tx_bytes, prev_rx_bytes, prev_timestamp)}
        self.prev_stats = {}
        
        # Define CSV file path
        self.csv_filename = "traffic_dataset.csv"
        self._initialize_csv()

    def _initialize_csv(self):
        """Creates the CSV file with headers if it doesn't exist yet."""
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'switch_id', 'port_no', 
                    'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps',
                    'tx_errors', 'rx_errors', 'is_congested'
                ])
            self.logger.info(f"Initialized empty dataset file: {self.csv_filename}")

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """Tracks switches as they connect or disconnect from Ryu."""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info(f"Switch joined the control plane: Datapath ID {datapath.id}")
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info(f"Switch left the control plane: Datapath ID {datapath.id}")
                del self.datapaths[datapath.id]

    def _monitor_loop(self):
        """Triggers an OpenFlow stats request rule every 5 seconds."""
        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
            hub.sleep(5)  # Polls every 5 seconds to build precise delta metrics

    def _request_stats(self, datapath):
        self.logger.debug(f"Sending stats request to switch {datapath.id}")
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Request port counters from the switch
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        """Triggers automatically when Mininet switches reply with stats."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        import time
        current_time = int(time.time())

        for stat in body:
            port_no = stat.port_no
            
            # Ignore internal/local reserved management ports
            if port_no > 50 or port_no == 0:
                continue
                
            tx_bytes = stat.tx_bytes
            rx_bytes = stat.rx_bytes
            tx_errors = stat.tx_errors
            rx_errors = stat.rx_errors
            
            # --- TELEMETRY CALCULATIONS (FEATURE ENGINEERING) ---
            tx_kbps = 0.0
            rx_kbps = 0.0
            stat_key = (dpid, port_no)
            
            if stat_key in self.prev_stats:
                prev_tx, prev_rx, prev_time = self.prev_stats[stat_key]
                time_delta = current_time - prev_time
                
                if time_delta > 0:
                    # Formula: (Delta Bytes * 8 bits/byte) / 1000 = Kbps / time_delta
                    tx_kbps = ((tx_bytes - prev_tx) * 8) / (1000 * time_delta)
                    rx_kbps = ((rx_bytes - prev_rx) * 8) / (1000 * time_delta)
            
            # Save current state as historical point for the next loop run
            self.prev_stats[stat_key] = (tx_bytes, rx_bytes, current_time)
            
            # --- TARGET LABEL GENERATION FOR PHASE 2 ---
            # Ground truth metric label: If a link handles extreme throughput or starts dropping
            # packets via tx_errors, we flag it as congested (1) for our ML training set.
            is_congested = 1 if (tx_kbps > 8000 or tx_errors > 0) else 0
            
            # --- EXPORT AUTOMATICALLY TO DATASET ---
            with open(self.csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    current_time, dpid, port_no, 
                    tx_bytes, rx_bytes, round(tx_kbps, 2), round(rx_kbps, 2),
                    tx_errors, rx_errors, is_congested
                ])
                
            print(f"[DATA EXPORTED] Switch {dpid} Port {port_no} -> TX Speed: {round(tx_kbps, 2)} Kbps | Congested Status: {is_congested}")