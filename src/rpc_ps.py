import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import threading
import os
import socket
import time
import warnings
from .model import get_model
from .utils import get_device
from .stats_collector import StatsCollector

# Suppress PyTorch distributed backend deprecation warning
warnings.filterwarnings("ignore", message=".*Backend.*ProcessGroup.*deprecated.*")

class ParameterServer:
    def __init__(self, num_classes=200, checkpoint_dir="checkpoints"):
        self.lock = threading.Lock()
        # Force CPU for RPC compatibility - PS must use CPU for tensor serialization
        self.device = torch.device("cpu")
        self.model = get_model(num_classes=num_classes).to(self.device)
        print(f"Parameter Server using device: {self.device} (forced CPU for RPC compatibility)")
        # Use Adam optimizer to match train_simple.py
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_acc = 0.0
        
        # Try to load checkpoint
        self.load_checkpoint()
        
        # Track number of gradient updates received
        self.update_count = 0
        
        # CRITICAL: Set model to eval mode
        # The PS never does forward passes, so BatchNorm statistics are never updated.
        # Keeping the model in eval() mode prevents BatchNorm from trying to use
        # uninitialized running_mean/running_var, which causes validation to explode.
        self.model.eval()
        
        # Initialize Stats Collector
        hostname = socket.gethostname()
        self.stats = StatsCollector(f"{hostname}_ps")
        self.stats.log_parameters({
            "role": "parameter_server",
            "model": "resnet18",
            "optimizer": "Adam",
            "lr": 0.001,
            "device": str(self.device),
            "checkpoint_dir": checkpoint_dir
        })
        self.stats.start_monitoring()
        
        # Start background saver for 'last' checkpoint
        self.saver_thread = threading.Thread(target=self._periodic_saver, daemon=True)
        self.saver_thread.start()
        
        print(f"Parameter Server initialized on {get_device()}")

    def load_checkpoint(self):
        """Loads the best or last checkpoint if available."""
        last_ckpt = os.path.join(self.checkpoint_dir, "last.pth")
        best_ckpt = os.path.join(self.checkpoint_dir, "best.pth")
        
        ckpt_path = None
        if os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
        elif os.path.exists(best_ckpt):
            ckpt_path = best_ckpt
            
        if ckpt_path:
            print(f"Loading checkpoint from {ckpt_path}...")
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.best_acc = checkpoint.get('best_acc', 0.0)
                print(f"Checkpoint loaded. Best accuracy so far: {self.best_acc:.2f}%")
                # Ensure model stays in eval mode after loading checkpoint
                self.model.eval()
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

    def save_checkpoint(self, filename="last.pth"):
        """Saves the current model state."""
        save_path = os.path.join(self.checkpoint_dir, filename)
        # We should probably lock while saving to get a consistent state
        with self.lock:
            state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_acc': self.best_acc,
                'timestamp': time.time()
            }
        
        # Save to a temp file then rename to avoid corruption
        tmp_path = save_path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, save_path)
        print(f"Checkpoint saved to {save_path}")

    def _periodic_saver(self, interval=600):
        """Periodically saves the 'last' checkpoint."""
        while True:
            time.sleep(interval)
            self.save_checkpoint("last.pth")

    def get_weights(self):
        """
        Returns the current model weights to the worker.
        """
        # No lock needed for reading, or maybe a reader lock if strict consistency required.
        # For async training, reading stale weights is acceptable.
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def remote_update_parameters(self, gradients):
        """
        Updates the model parameters with the received gradients.
        Uses a lock to ensure atomic updates.
        """
        with self.lock:
            # Zero grads first (critical!)
            self.optimizer.zero_grad()
            
            # Apply gradients (REPLACE, not accumulate)
            for name, param in self.model.named_parameters():
                if name in gradients:
                    # Move grad to device and REPLACE (not +=)
                    param.grad = gradients[name].to(param.device)
            
            # Step optimizer
            self.optimizer.step()
            
            # Track updates
            self.update_count += 1
            
            # Log every 100 updates
            if self.update_count % 100 == 0:
                print(f"[PS] Processed {self.update_count} gradient updates")
                self.stats.log_training_metrics({
                    "update_count": self.update_count,
                    "type": "gradient_update"
                })
            
        return True

    def report_metric(self, accuracy):
        """
        Updates best accuracy and saves best checkpoint if improved.
        """
        with self.lock:
            if accuracy > self.best_acc:
                print(f"New best accuracy: {accuracy:.2f}% (was {self.best_acc:.2f}%)")
                self.best_acc = accuracy
                # Save best checkpoint immediately (or trigger it)
                # We can do it in a separate thread to not block RPC, but for now let's do it here
                # actually, saving might take time, better to do it async or just quick copy?
                # For simplicity, let's save directly but maybe we should optimize later.
                threading.Thread(target=self.save_checkpoint, args=("best.pth",)).start()
        return True

    def update_batchnorm_stats(self, state_dict):
        """
        Updates the BatchNorm running statistics from a worker's state_dict.
        This is critical because the PS never does forward passes, so its BN stats
        never update naturally. We must sync them from workers.
        """
        with self.lock:
            current_state = self.model.state_dict()
            for name, param in state_dict.items():
                # Only update running_mean and running_var
                if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                    if name in current_state:
                        # Update the buffer on the correct device
                        current_state[name].copy_(param.to(self.device))
            
            # Load the updated state back into the model
            self.model.load_state_dict(current_state)
            print("[PS] Updated BatchNorm statistics from worker")
        return True

# Global PS instance
global_ps = None

def get_global_weights():
    global global_ps
    if global_ps is None:
        print("ERROR: global_ps is None")
        return {}
    return global_ps.get_weights()

def update_global_parameters(grads):
    global global_ps
    if global_ps is None:
        print("ERROR: global_ps is None")
        return False
    return global_ps.remote_update_parameters(grads)

def update_global_batchnorm_stats(state_dict):
    global global_ps
    if global_ps is None:
        print("ERROR: global_ps is None")
        return False
    return global_ps.update_batchnorm_stats(state_dict)

def report_global_metric(accuracy):
    global global_ps
    if global_ps is None:
        print("ERROR: global_ps is None")
        return False
    return global_ps.report_metric(accuracy)

def run_ps(rank, world_size, master_addr, master_port, checkpoint_dir="checkpoints"):
    global global_ps
    global_ps = ParameterServer(checkpoint_dir=checkpoint_dir)
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Fix for macOS network interface detection
    # Force Gloo to use the correct network interface
    if master_addr != "localhost" and master_addr != "127.0.0.1":
        # Try to detect the network interface automatically
        import socket
        import subprocess
        try:
            # Get the network interface name for the IP
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            current_iface = None
            for line in lines:
                if line and not line.startswith('\t') and not line.startswith(' '):
                    # This is an interface name line
                    current_iface = line.split(':')[0]
                elif 'inet ' in line and master_addr in line:
                    # Found the interface with our IP
                    if current_iface:
                        os.environ['GLOO_SOCKET_IFNAME'] = current_iface
                        print(f"Setting GLOO_SOCKET_IFNAME={current_iface}")
                        break
        except Exception as e:
            print(f"Warning: Could not auto-detect network interface: {e}")
            # Fallback: try common macOS interface names
            os.environ['GLOO_SOCKET_IFNAME'] = 'en0'
            print("Using fallback interface: en0")
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=300  # Increased from 60s to 300s for better stability
    )
    
    rpc.init_rpc(
        f"ps",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )
    
    print("Parameter Server is running...")
    print("Press Ctrl+C to stop the server")
    
    # Keep the PS alive - wait indefinitely until interrupted
    try:
        import signal
        import sys
        
        def signal_handler(sig, frame):
            print("\nShutting down Parameter Server...")
            if global_ps:
                global_ps.save_checkpoint("last.pth")
                global_ps.stats.stop_monitoring()
            rpc.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Block forever until signal received
        signal.pause()
    except KeyboardInterrupt:
        print("\nShutting down Parameter Server...")
        if global_ps:
            global_ps.save_checkpoint("last.pth")
            global_ps.stats.stop_monitoring()
        rpc.shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=3)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29500")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    run_ps(args.rank, args.world_size, args.master_addr, args.master_port, args.checkpoint_dir)
