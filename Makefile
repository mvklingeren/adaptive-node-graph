# Makefile for compiling the CUDA Shakespeare LLM project

# Compiler
NVCC = /usr/local/cuda/bin/nvcc

# Executable name
TARGET = run_model

# Source files
CPP_SRC = main.cpp
CU_SRC = generated-shakespear-llm-kernel.cu

# Object files
CPP_OBJ = $(CPP_SRC:.cpp=.o)
CU_OBJ = $(CU_SRC:.cu=.o)

# Compiler flags
# -O3 for optimization
# -arch=sm_89 is for RTX 4090. Use sm_75 for Tesla T4, sm_80 for A100, etc.
# -std=c++17 to enable modern C++ features used in main.cpp
# WORKSPACE_SIZE_MB can be overridden: make WORKSPACE_SIZE_MB=2048
WORKSPACE_SIZE_MB ?= 1024
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_89 -DWORKSPACE_SIZE_MB=$(WORKSPACE_SIZE_MB)

# Default target
all: $(TARGET)

# Linking the final executable
$(TARGET): $(CPP_OBJ) $(CU_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(CPP_OBJ) $(CU_OBJ)
	@echo "Linking complete. Executable created: $(TARGET)"

# Compiling C++ source
$(CPP_OBJ): $(CPP_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $(CPP_SRC) -o $(CPP_OBJ)

# Compiling CUDA source
$(CU_OBJ): $(CU_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $(CU_SRC) -o $(CU_OBJ)

# Clean up build files
clean:
	rm -f $(TARGET) $(CPP_OBJ) $(CU_OBJ)
	@echo "Cleanup complete."

.PHONY: all clean
