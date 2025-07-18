# Simple Makefile to build PyTorch + CUDA extension with setup.py

# Output shared object name
TARGET = my_cuda_conv

# Extension suffix (e.g., .cpython-311-x86_64-linux-gnu.so)
EXT = $(shell python3-config --extension-suffix)

# Default build rule
all:
	python3 setup.py build_ext --inplace

# Clean rule to remove build artifacts
clean:
	rm -rf build my_cuda_conv*.so *.egg-info __pycache__

