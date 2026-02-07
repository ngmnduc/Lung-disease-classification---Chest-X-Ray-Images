@echo off
REM Script tự động download ResNet50 weights cho Windows
REM Chạy script này TRƯỚC KHI chạy features.py lần đầu tiên

echo ==========================================
echo Setup ResNet50 Weights
echo ==========================================

REM Tạo thư mục
if not exist "%USERPROFILE%\.keras\models" (
    mkdir "%USERPROFILE%\.keras\models"
)

REM Kiểm tra file đã tồn tại
if exist "%USERPROFILE%\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5" (
    echo [OK] ResNet50 weights already exists!
    echo   Location: %USERPROFILE%\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    dir "%USERPROFILE%\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    exit /b 0
)

echo.
echo Downloading ResNet50 weights (~94MB)...
echo This may take 1-2 minutes depending on your internet speed.
echo.

REM Download bằng PowerShell
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' -OutFile '%USERPROFILE%\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'}"

REM Kiểm tra download thành công
if exist "%USERPROFILE%\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5" (
    echo.
    echo ==========================================
    echo [OK] Download successful!
    echo ==========================================
    dir "%USERPROFILE%\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    echo.
    echo You can now run: python -m src.features
) else (
    echo.
    echo [ERROR] Download failed!
    echo Please download manually from:
    echo https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    echo.
    echo And save to: %USERPROFILE%\.keras\models\
    exit /b 1
)
