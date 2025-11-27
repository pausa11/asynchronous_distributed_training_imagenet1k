import torch
import torch.distributed.rpc as rpc
import os
import socket
import time
import warnings
from .model import get_model
from .utils import get_device
from .dataset import get_imagenet_dataset
from .stats_collector import StatsCollector
from functools import wraps

# Suppress PyTorch distributed backend deprecation warning
warnings.filterwarnings("ignore", message=".*Backend.*ProcessGroup.*deprecated.*")

def rpc_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    """Decorator to retry RPC calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    last_exception = e
                    error_msg = str(e)
                    # Check if it's a connection error
                    if "EOF" in error_msg or "EINVAL" in error_msg or "connection" in error_msg.lower():
                        if attempt < max_retries - 1:
                            print(f"RPC call failed (attempt {attempt + 1}/{max_retries}): {error_msg}. Retrying in {delay}s...")
                            time.sleep(delay)
                            delay *= backoff_factor
                        else:
                            print(f"RPC call failed after {max_retries} attempts: {error_msg}")
                            raise
                    else:
                        # Not a connection error, raise immediately
                        raise
            
            # Should not reach here, but just in case
            raise last_exception
        return wrapper
    return decorator

class Worker:
    def __init__(self, ps_rref, rank, world_size, dataset_url, val_dataset_url=None, use_webdataset=False):
        self.ps_rref = ps_rref
        self.rank = rank
        self.device = get_device()
        self.model = get_model(num_classes=200).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize Stats Collector
        hostname = socket.gethostname()
        self.stats = StatsCollector(f"{hostname}_worker{rank}")
        self.stats.log_parameters({
            "role": "worker",
            "rank": rank,
            "world_size": world_size,
            "dataset_url": dataset_url,
            "val_dataset_url": val_dataset_url,
            "batch_size": 64,
            "device": str(self.device),
            "use_webdataset": use_webdataset
        })
        self.stats.start_monitoring()
        
        # Load training dataset
        if use_webdataset:
            print(f"📦 [Worker {rank}] Using WebDataset from: {dataset_url}")
            from .dataset import get_imagenet_dataset
            self.loader = get_imagenet_dataset(dataset_url, batch_size=64, num_workers=2, train=True)
        else:
            print(f"📁 [Worker {rank}] Using ImageFolder from: {dataset_url}")
            from .dataset import get_imagenet_folder_dataset
            self.loader = get_imagenet_folder_dataset(dataset_url, batch_size=64, num_workers=2, train=True)
        
        # Load validation dataset if provided
        self.val_loader = None
        if val_dataset_url:
            print(f"🔍 [Worker {rank}] Loading validation data from: {val_dataset_url}")
            if use_webdataset:
                from .dataset import get_imagenet_dataset
                self.val_loader = get_imagenet_dataset(val_dataset_url, batch_size=64, num_workers=0, train=False)
            else:
                from .dataset import get_imagenet_folder_dataset
                self.val_loader = get_imagenet_folder_dataset(val_dataset_url, batch_size=64, num_workers=0, train=False)
            print(f"✓ [Worker {rank}] Validation loader created successfully")
        
    def train(self, epochs=1):
        """Train for multiple epochs with validation."""
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            self.train_epoch(epoch)
            
            # Validate after each epoch if validation set is available
            if self.val_loader is not None:
                val_acc = self.validate_epoch(epoch)
                # Report validation accuracy to PS for checkpointing
                rpc.rpc_sync("ps", report_global_metric, args=(val_acc,))
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} Complete")
            print(f"{'='*60}\n")
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 1. Pull latest weights from PS (CPU) with retry logic
            try:
                weights_cpu = self._get_weights_with_retry()
                # Move weights to worker's device (MPS) for computation
                # Move weights to worker's device (MPS) for computation
                weights = {k: v.to(self.device) for k, v in weights_cpu.items()}
                
                # CRITICAL: Do NOT load BatchNorm statistics from PS during training!
                # Since PS never updates BN stats (no forward pass), loading them would
                # reset the worker's local running_mean/var to initial values every batch.
                # We want the worker to accumulate BN stats locally, then sync to PS at end of epoch.
                training_weights = {
                    k: v for k, v in weights.items() 
                    if "running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k
                }
                
                self.model.load_state_dict(training_weights, strict=False)
            except Exception as e:
                print(f"Failed to get weights from PS: {e}. Skipping batch {batch_idx}.")
                continue
            
            # 2. Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 3. Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # 4. Push gradients to PS with retry logic
            # Move gradients to CPU for RPC serialization
            try:
                grads = {k: v.grad.cpu() for k, v in self.model.named_parameters() if v.grad is not None}
                self._update_parameters_with_retry(grads)
            except Exception as e:
                print(f"Failed to push gradients to PS: {e}. Continuing with next batch.")
            
            # Statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Print progress
            if batch_idx % 50 == 0:
                batch_loss = loss.item()
                batch_acc = 100. * correct / total
                print(f'Epoch {epoch} [{batch_idx}/{len(self.loader)}] '
                      f'Loss: {batch_loss:.4f} Acc: {batch_acc:.2f}%')
            
            # Log to stats every 100 batches to reduce I/O overhead
            if batch_idx % 100 == 0:
                self.stats.log_training_metrics({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss": loss.item(),
                    "accuracy": 100. * correct / total
                })
        
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(self.loader)
        avg_acc = 100. * correct / total
        
        print(f'\nEpoch {epoch} Training Summary:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Average Accuracy: {avg_acc:.2f}%')
        print(f'  Time: {epoch_time:.2f}s')
        
        # Sync BatchNorm statistics to PS
        # This is critical because the PS never does forward passes
        print(f"🔄 [Worker {self.rank}] Syncing BatchNorm statistics to PS...")
        try:
            # Create a state dict with only BN stats
            bn_stats = {
                k: v.cpu() for k, v in self.model.state_dict().items() 
                if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k
            }
            rpc.rpc_sync("ps", update_global_batchnorm_stats, args=(bn_stats,))
            print(f"✓ [Worker {self.rank}] BatchNorm statistics synced successfully")
        except Exception as e:
            print(f"❌ [Worker {self.rank}] Failed to sync BatchNorm stats: {e}")
    
    @rpc_retry(max_retries=5, initial_delay=1.0, backoff_factor=2.0)
    def _get_weights_with_retry(self):
        """Get weights from PS with retry logic."""
        return rpc.rpc_sync("ps", get_global_weights, timeout=120)
    
    @rpc_retry(max_retries=5, initial_delay=1.0, backoff_factor=2.0)
    def _update_parameters_with_retry(self, grads):
        """Update parameters on PS with retry logic."""
        return rpc.rpc_sync("ps", update_global_parameters, args=(grads,), timeout=120)
    
    def validate_epoch(self, epoch):
        """Validate the model on the validation set."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        # Load weights ONCE at the beginning of validation (not per batch!)
        # This ensures consistent validation metrics, matching train_simple.py behavior
        try:
            weights_cpu = self._get_weights_with_retry()
            weights = {k: v.to(self.device) for k, v in weights_cpu.items()}
            self.model.load_state_dict(weights)
        except Exception as e:
            print(f"Failed to get weights from PS for validation: {e}")
            print(f"Validation will be skipped for epoch {epoch}")
            return 0.0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Statistics
                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Print progress
                if batch_idx % 50 == 0:
                    batch_loss = loss.item()
                    batch_acc = 100. * correct / total
                    print(f'Validation [{batch_idx}/{len(self.val_loader)}] '
                          f'Loss: {batch_loss:.4f} Acc: {batch_acc:.2f}%')
        
        val_time = time.time() - start_time
        avg_loss = running_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
        print(f'\nEpoch {epoch} Validation Summary:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Average Accuracy: {avg_acc:.2f}%')
        print(f'  Time: {val_time:.2f}s')
        
        # Log validation metrics
        self.stats.log_training_metrics({
            "epoch": epoch,
            "batch": "validation",
            "loss": avg_loss,
            "accuracy": avg_acc
        })
        
        return avg_acc

# Helper functions to be called by RPC
from .rpc_ps import get_global_weights, update_global_parameters, report_global_metric, update_global_batchnorm_stats

def run_worker(rank, world_size, master_addr, master_port, dataset_url, val_dataset_url, epochs, use_webdataset=False):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Fix for macOS network interface detection
    if master_addr != "localhost" and master_addr != "127.0.0.1":
        # Try common interface names
        import subprocess
        try:
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            # Look for active interfaces with inet addresses
            for i, line in enumerate(lines):
                if line and not line.startswith('\t') and not line.startswith(' '):
                    iface = line.split(':')[0]
                    # Check next few lines for inet
                    for j in range(i+1, min(i+10, len(lines))):
                        if 'inet ' in lines[j] and '127.0.0.1' not in lines[j]:
                            os.environ['GLOO_SOCKET_IFNAME'] = iface
                            print(f"Setting GLOO_SOCKET_IFNAME={iface}")
                            break
                    if 'GLOO_SOCKET_IFNAME' in os.environ:
                        break
        except Exception as e:
            print(f"Warning: Could not auto-detect network interface: {e}")
            os.environ['GLOO_SOCKET_IFNAME'] = 'en0'
            print("Using fallback interface: en0")
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=300  # Increased from 60s to 300s for better stability
    )
    
    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )
    
    print(f"Worker {rank} started...")
    
    # Start training
    worker = Worker(None, rank, world_size, dataset_url, val_dataset_url, use_webdataset)
    worker.train(epochs)

    
    worker.stats.stop_monitoring()
    rpc.shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=3)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29500")
    parser.add_argument("--dataset_url", type=str, required=True)
    parser.add_argument("--val_dataset_url", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    run_worker(args.rank, args.world_size, args.master_addr, args.master_port, args.dataset_url, args.val_dataset_url, args.epochs)