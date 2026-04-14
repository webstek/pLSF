# Makefile for pLSF – Parallelized Lower-Star Filtration
# 
# Build system: CUDA + host C++
# Compilers: nvcc (CUDA), g++ (host C++)
#
# Usage:
#   make                  # build all targets
#   make release          # build optimized binary
#   make debug            # build with debugging symbols
#   make clean            # remove build artifacts
#   make run              # build and run with default settings
# 

# Configuration 
NVCC                := nvcc
CXX                 := g++
CXXFLAGS            := -std=c++17 -Wall -Wextra
NVCCFLAGS           := -std=c++17
INCLUDE_DIRS        := -I./src -I./lib/phat/include
LDFLAGS             :=

# Build modes
DEBUG_FLAGS         := -g -O0
DEBUG_NVFLAGS       := -G -g -O0
RELEASE_FLAGS       := -O3 -DNDEBUG
RELEASE_NVFLAGS     := -O3 -DNDEBUG

# Default: Release
BUILD_MODE          ?= release
ifeq ($(BUILD_MODE), debug)
    CXXFLAGS        += $(DEBUG_FLAGS)
    NVCCFLAGS       += $(DEBUG_NVFLAGS)
else
    CXXFLAGS        += $(RELEASE_FLAGS)
    NVCCFLAGS       += $(RELEASE_NVFLAGS)
endif

# CUDA architecture (default: generate for common architectures)
# Override with: make CUDA_ARCH="sm_80"
CUDA_ARCH           ?= native
NVCCFLAGS           += -arch=$(CUDA_ARCH)

# OpenMP for boundary matrix computation
CXXFLAGS            += -fopenmp
LDFLAGS             += -Xcompiler -fopenmp -lstdc++fs

# File organization 
SRC_DIR             := src
CUDA_DIR            := src/cuda
BUILD_DIR           := build
BIN_DIR             := bin

# Source files
MAIN_SRC            := $(SRC_DIR)/main.cpp
CUDA_SRCS           := $(wildcard $(CUDA_DIR)/*.cu)

# Object files
MAIN_OBJ            := $(BUILD_DIR)/main.o
CUDA_OBJS           := $(patsubst $(CUDA_DIR)/%.cu, $(BUILD_DIR)/cuda/%.o, $(CUDA_SRCS))
OBJECTS             := $(MAIN_OBJ) $(CUDA_OBJS)

# Output binary
EXECUTABLE          := $(BIN_DIR)/plsf

# Targets 
.PHONY: all release debug clean run info help

# Default target
all: $(EXECUTABLE)

# Build modes
release: BUILD_MODE := release
release: all

debug: BUILD_MODE := debug
debug: all

# Link: use nvcc to link CUDA objects with host objects
$(EXECUTABLE): $(OBJECTS) | $(BIN_DIR)
	@echo "[LD] $@"
	@$(NVCC) $(NVCCFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

# Compile main.cpp (host code) — use nvcc to handle CUDA includes
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp | $(BUILD_DIR)
	@echo "[NVCC] $<"
	@$(NVCC) $(NVCCFLAGS) $(INCLUDE_DIRS) -Xcompiler "$(CXXFLAGS)" -c $< -o $@

# Compile CUDA kernel files
$(BUILD_DIR)/cuda/%.o: $(CUDA_DIR)/%.cu | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	@echo "[NVCC] $<"
	@$(NVCC) $(NVCCFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Create directories
$(BUILD_DIR) $(BIN_DIR):
	@mkdir -p $@

# Clean build artifacts
clean:
	@echo "[RM] Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)

# Build and run
run: $(EXECUTABLE)
	@echo "[RUN] $(EXECUTABLE) -h"
	@$(EXECUTABLE) -h

# Show configuration
info:
	@echo "pLSF Build Configuration"
	@echo "========================"
	@echo "NVCC              : $(NVCC)"
	@echo "Host CXX          : $(CXX)"
	@echo "C++ Standard      : C++17"
	@echo "Build Mode        : $(BUILD_MODE)"
	@echo "CUDA Architecture : $(CUDA_ARCH)"
	@echo "Build Directory   : $(BUILD_DIR)"
	@echo "Binary            : $(EXECUTABLE)"
	@echo ""
	@echo "Source Files:"
	@echo "  Main     : $(MAIN_SRC)"
	@echo "  CUDA     : $(CUDA_SRCS)"

# Show help
help:
	@echo "pLSF Makefile Targets"
	@echo "====================="
	@echo ""
	@echo "make [release]      Build optimized binary (default)"
	@echo "make debug          Build with debugging symbols"
	@echo "make clean          Remove all build artifacts"
	@echo "make run            Build and display help"
	@echo "make info           Show build configuration"
	@echo "make help           Show this message"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ARCH=sm_80   Target a specific GPU architecture"
	@echo ""
	@echo "Environment Variables:"
	@echo "  BUILD_MODE=debug    Compile with -g -O0"
	@echo "  CUDA_ARCH=sm_80     Target a specific NVIDIA architecture"
	@echo "  make                         # Release build"
	@echo "  make debug                   # Debug build"
	@echo "  make CUDA_ARCH=sm_90         # Build for Hopper"
	@echo "  make clean && make release   # Clean rebuild"
