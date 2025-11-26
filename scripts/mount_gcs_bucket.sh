#!/bin/bash
# Script to mount GCS bucket using gcsfuse for distributed training
# This allows accessing the bucket as a local filesystem

set -e  # Exit on error

BUCKET_NAME="caso-estudio-2"
MOUNT_POINT="$HOME/gcs_data"

echo "🔧 GCS Bucket Mounting Script"
echo "=============================="
echo ""

# Check if gcsfuse is installed
if ! command -v gcsfuse &> /dev/null; then
    echo "❌ gcsfuse is not installed!"
    echo ""
    echo "To install gcsfuse on macOS:"
    echo "  brew install gcsfuse"
    echo ""
    echo "To install gcsfuse on Linux:"
    echo "  export GCSFUSE_REPO=gcsfuse-\$(lsb_release -c -s)"
    echo "  echo \"deb https://packages.cloud.google.com/apt \$GCSFUSE_REPO main\" | sudo tee /etc/apt/sources.list.d/gcsfuse.list"
    echo "  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install gcsfuse"
    exit 1
fi

echo "✓ gcsfuse is installed"
echo ""

# Create mount point if it doesn't exist
if [ ! -d "$MOUNT_POINT" ]; then
    echo "📁 Creating mount point: $MOUNT_POINT"
    mkdir -p "$MOUNT_POINT"
fi

# Check if already mounted
if mountpoint -q "$MOUNT_POINT" 2>/dev/null || mount | grep -q "$MOUNT_POINT"; then
    echo "⚠️  Bucket is already mounted at $MOUNT_POINT"
    echo ""
    echo "To unmount, run:"
    echo "  umount $MOUNT_POINT"
    echo "  # or on macOS:"
    echo "  fusermount -u $MOUNT_POINT"
    exit 0
fi

# Mount the bucket
echo "🚀 Mounting bucket gs://$BUCKET_NAME to $MOUNT_POINT..."
echo ""

gcsfuse \
    --implicit-dirs \
    --stat-cache-ttl 10s \
    --type-cache-ttl 10s \
    "$BUCKET_NAME" "$MOUNT_POINT"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully mounted gs://$BUCKET_NAME to $MOUNT_POINT"
    echo ""
    echo "Dataset path: $MOUNT_POINT/tiny-imagenet-200"
    echo ""
    echo "To unmount later, run:"
    echo "  umount $MOUNT_POINT"
    echo "  # or on macOS:"
    echo "  fusermount -u $MOUNT_POINT"
else
    echo ""
    echo "❌ Failed to mount bucket"
    echo ""
    echo "Make sure you have:"
    echo "  1. Authenticated with gcloud: gcloud auth application-default login"
    echo "  2. Proper permissions to access the bucket"
    exit 1
fi
