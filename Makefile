# Makefile for pLSF – Parallelized Lower-Star Filtration
# ─────────────────────────────────────────────────────────────────────────────
# Build system: AdaptiveCpp (hip SYCL)
# Compiler: acpp (AdaptiveCpp C++ compiler)
#
# Usage:
#   make                  # build all targets
#   make release          # build optimized binary
#   make debug            # build with debugging symbols
#   make clean            # remove build artifacts
#   make run              # build and run with default settings
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuration ──────────────────────────────────────────────────────────────
CXX                 := acpp
CXXFLAGS            := -std=c++17 -Wall -Wextra
INCLUDE_DIRS        := -I./src
LDFLAGS             :=

# Build modes
DEBUG_FLAGS         := -g -O0 -fno-omit-frame-pointer
RELEASE_FLAGS       := -O3 -DNDEBUG

# Default: Release
BUILD_MODE          ?= release
ifeq ($(BUILD_MODE), debug)
    CXXFLAGS        += $(DEBUG_FLAGS)
else
    CXXFLAGS        += $(RELEASE_FLAGS)
endif

# Targets (optional; AdaptiveCpp will auto-detect or use "generic")
# For specific hardware, add: --hipsycl-targets=...
# Examples:
#   --hipsycl-targets=generic                   # CPU (auto-detect)
#   --hipsycl-targets=cuda                      # NVIDIA GPU
#   --hipsycl-targets=hip                       # AMD GPU
#   --hipsycl-targets=omp                       # OpenMP
SYCL_TARGETS        ?= generic

# ── File organization ─────────────────────────────────────────────────────────
SRC_DIR             := src
BUILD_DIR           := build
BIN_DIR             := bin

# Source files
# io.cpp and lsf.cpp are header-only template stubs; only main.cpp is compiled
MAIN_SRC            := $(SRC_DIR)/main.cpp

SOURCES             := $(MAIN_SRC)

# Object files
OBJECTS             := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

# Output binary
EXECUTABLE          := $(BIN_DIR)/plsf

# ── Targets ────────────────────────────────────────────────────────────────────
.PHONY: all release debug clean run info

# Default target
all: $(EXECUTABLE)

# Build modes
release: BUILD_MODE := release
release: all

debug: BUILD_MODE := debug
debug: all

# Build executable
$(EXECUTABLE): $(OBJECTS) | $(BIN_DIR)
	@echo "[LD] $@"
	@$(CXX) $(CXXFLAGS) --hipsycl-targets=$(SYCL_TARGETS) $(OBJECTS) $(LDFLAGS) -o $@

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	@echo "[CXX] $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) --hipsycl-targets=$(SYCL_TARGETS) -c $< -o $@

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
	@echo "Compiler          : $(CXX)"
	@echo "C++ Standard      : C++17"
	@echo "Build Mode        : $(BUILD_MODE)"
	@echo "SYCL Targets      : $(SYCL_TARGETS)"
	@echo "Build Directory   : $(BUILD_DIR)"
	@echo "Binary            : $(EXECUTABLE)"
	@echo ""
	@echo "Source Files:"
	@echo "  Main     : $(MAIN_SRC)"

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
	@echo "Environment Variables:"
	@echo "  BUILD_MODE=debug    Compile with -g -O0"
@echo "  SYCL_TARGETS=...    AdaptiveCpp backend: generic|omp|cuda|hip"
	@echo "  make                         # Release build, auto-detect GPU"
	@echo "  make debug                   # Debug build"
	@echo "  make SYCL_TARGETS=omp        # CPU via OpenMP"
	@echo "  make clean && make release   # Clean rebuild"
