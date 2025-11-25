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
    def __init__(self, ps_rref, rank, world_size, dataset_url, val_dataset_url=None):
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
            "device": str(self.device)
        })
        self.stats.start_monitoring()
        
        # Load training dataset
        # Note: In a real distributed setting, we'd shard the dataset based on rank
        self.loader = get_imagenet_dataset(dataset_url, batch_size=64, num_workers=2, train=True)
        
        # Load validation dataset if provided
        # Use num_workers=0 for validation to avoid "fewer shards than workers" error
        self.val_loader = None
        if val_dataset_url:
            self.val_loader = get_imagenet_dataset(val_dataset_url, batch_size=64, num_workers=0, train=False)
        
    def train(self, epochs=1):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            
            # Validate after each epoch if validation set is available
            if self.val_loader is not None:
                val_acc = self.validate_epoch(epoch)
                # Report validation accuracy to PS for checkpointing
                rpc.rpc_sync("ps", report_global_metric, args=(val_acc,))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for i, (data, target) in enumerate(self.loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 1. Pull latest weights from PS (CPU) with retry logic
            try:
                weights_cpu = self._get_weights_with_retry()
                # Move weights to worker's device (MPS) for computation
                weights = {k: v.to(self.device) for k, v in weights_cpu.items()}
                self.model.load_state_dict(weights)
            except Exception as e:
                print(f"Failed to get weights from PS: {e}. Skipping batch {i}.")
                continue
            
            # 2. Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Track loss only (accuracy calculated in validation)
            total_loss += loss.item()
            num_batches += 1
            
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
            
            # Log only loss (reduced overhead)
            if i % 10 == 0:
                print(f"Rank {self.rank}, Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
            
            # Log to stats every 100 batches to reduce I/O overhead
            if i % 100 == 0:
                self.stats.log_training_metrics({
                    "epoch": epoch,
                    "batch": i,
                    "loss": loss.item(),
                    "accuracy": None  # Only calculated in validation
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Rank {self.rank}, Epoch {epoch} Training Summary: Average Loss: {avg_loss:.4f}")
    
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
        total_loss = 0
        correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for i, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Pull latest weights from PS (CPU) with retry logic
                try:
                    weights_cpu = self._get_weights_with_retry()
                    # Move weights to worker's device (MPS) for computation
                    weights = {k: v.to(self.device) for k, v in weights_cpu.items()}
                    self.model.load_state_dict(weights)
                except Exception as e:
                    print(f"Failed to get weights from PS during validation: {e}. Skipping batch {i}.")
                    continue
                
                # Forward pass only
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct_batch = pred.eq(target.view_as(pred)).sum().item()
                correct += correct_batch
                total_samples += len(target)
                total_loss += loss.item()
                num_batches += 1
                
                if i % 10 == 0:
                    accuracy = 100. * correct_batch / len(target)
                    print(f"Rank {self.rank}, Epoch {epoch}, Validation Batch {i}, Loss: {loss.item():.4f}, Acc: {accuracy:.2f}%")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = 100. * correct / total_samples if total_samples > 0 else 0
        print(f"Rank {self.rank}, Epoch {epoch} Validation Summary: Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.2f}%")
        
        # Log validation metrics
        self.stats.log_training_metrics({
            "epoch": epoch,
            "batch": "validation",
            "loss": avg_loss,
            "accuracy": avg_acc
        })
        
        return avg_acc

# Helper functions to be called by RPC
from .rpc_ps import get_global_weights, update_global_parameters, report_global_metric

def run_worker(rank, world_size, master_addr, master_port, dataset_url, val_dataset_url, epochs):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
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
    worker = Worker(None, rank, world_size, dataset_url, val_dataset_url)
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