"""
Quick script to verify the label mapping issue.
"""

# Read the class names from wnids.txt
wnids_path = "/Users/danieltorosoto/universidad/arq-cliente-servidor/asynchronous_distributed_training_imagenet1k/data/tiny-imagenet-200/wnids.txt"

with open(wnids_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"Total classes: {len(class_names)}")
print(f"First 10 classes: {class_names[:10]}")

# Show the problem with hash
print("\n" + "="*80)
print("PROBLEM: Using hash() for label mapping")
print("="*80)

# Test hash mapping (current buggy approach)
test_classes = class_names[:5]
print("\nCurrent approach (WRONG):")
for cls in test_classes:
    label_id = int(hash(cls) % 200)
    print(f"  {cls} -> {label_id}")

print("\nThe problem:")
print("  1. hash() is NOT deterministic across Python sessions")
print("  2. hash() doesn't preserve the order")
print("  3. Multiple classes can map to the same ID")

# Show correct mapping
print("\n" + "="*80)
print("SOLUTION: Use proper class-to-index mapping")
print("="*80)

class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

print("\nCorrect approach:")
for cls in test_classes:
    label_id = class_to_idx[cls]
    print(f"  {cls} -> {label_id}")

print("\nThis ensures:")
print("  1. Consistent mapping across all runs")
print("  2. Each class has a unique ID from 0 to 199")
print("  3. Training and validation use the same mapping")
