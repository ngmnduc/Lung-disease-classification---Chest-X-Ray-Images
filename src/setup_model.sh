#!/bin/bash

# Script tự động download ResNet50 weights
# Chạy script này TRƯỚC KHI chạy features.py lần đầu tiên

echo "=========================================="
echo "Setup ResNet50 Weights"
echo "=========================================="

# Tạo thư mục nếu chưa có
mkdir -p ~/.keras/models

# Kiểm tra file đã tồn tại chưa
if [ -f ~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 ]; then
    echo "✓ ResNet50 weights already exists!"
    echo "  Location: ~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    ls -lh ~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    exit 0
fi

echo ""
echo "Downloading ResNet50 weights (~94MB)..."
echo "This may take 1-2 minutes depending on your internet speed."
echo ""

# Download với curl (skip SSL verification)
curl -k -L -o ~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
  "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Kiểm tra download thành công
if [ -f ~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Download successful!"
    echo "=========================================="
    echo "File size:"
    ls -lh ~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    echo ""
    echo "You can now run: python3 -m src.features"
else
    echo ""
    echo "❌ Download failed!"
    echo "Please download manually from:"
    echo "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    echo ""
    echo "And save to: ~/.keras/models/"
    exit 1
fi
