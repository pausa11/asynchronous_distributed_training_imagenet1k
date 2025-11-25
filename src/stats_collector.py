import os
import json
import time
import threading
import psutil
import csv
from datetime import datetime

class StatsCollector:
    def __init__(self, system_name, base_dir="stats"):
        self.system_name = system_name
        # Create a unique directory for each run using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = os.path.join(base_dir, system_name, timestamp)
        self.running = False
        self.monitor_thread = None
        
        # Create directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize CSV files
        self.system_stats_file = os.path.join(self.base_dir, "system_stats.csv")
        self._init_csv(self.system_stats_file, ["timestamp", "cpu_percent", "memory_percent", "net_sent_mb", "net_recv_mb"])
        
        self.training_metrics_file = os.path.join(self.base_dir, "training_metrics.csv")
        self._init_csv(self.training_metrics_file, ["timestamp", "epoch", "batch", "loss", "accuracy"])

    def _init_csv(self, filepath, headers):
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_parameters(self, params):
        """Log configuration parameters to a JSON file."""
        param_file = os.path.join(self.base_dir, "parameters.json")
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=4)

    def log_training_metrics(self, metrics):
        """Log training metrics (epoch, batch, loss, etc.)."""
        with open(self.training_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                metrics.get("epoch", ""),
                metrics.get("batch", ""),
                metrics.get("loss", ""),
                metrics.get("accuracy", "")
            ])

    def _monitor_loop(self, interval):
        # Get initial network counters
        net_io_start = psutil.net_io_counters()
        
        while self.running:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                # Network I/O since last check (approximate rate if interval is consistent)
                net_io_now = psutil.net_io_counters()
                sent_mb = (net_io_now.bytes_sent - net_io_start.bytes_sent) / (1024 * 1024)
                recv_mb = (net_io_now.bytes_recv - net_io_start.bytes_recv) / (1024 * 1024)
                
                # Update start for next delta
                # Note: For cumulative stats, we might want to keep start fixed. 
                # But for "rate", we reset. Let's log cumulative for now as it's easier to diff later, 
                # OR log current snapshot. 
                # Let's log snapshot of usage.
                
                with open(self.system_stats_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        cpu_percent,
                        memory.percent,
                        sent_mb, 
                        recv_mb
                    ])
                
                time.sleep(interval)
            except Exception as e:
                print(f"Error in stats monitor: {e}")

    def start_monitoring(self, interval=1):
        if self.running:
            return
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
