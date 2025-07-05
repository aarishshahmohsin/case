# Makefile for SCIP C API Wide-Reach Classification Solver

# SCIP installation path
SCIP_PATH = /opt/homebrew/Cellar/scip/9.2.2 # for arm macs

# Compiler settings
CC = gcc
CXX = g++

# Compiler flags
CFLAGS = -O3 -Wall -Wextra -std=c99
CXXFLAGS = -O3 -Wall -Wextra -std=c++11

# Include directories
INCLUDES = -I$(SCIP_PATH)/include

# Library directories
LIBDIRS = -L$(SCIP_PATH)/lib

# Libraries to link
LIBS = -lscip -lsoplex  -ltbb -lgmp -lreadline -lz -lm

# Source files
SOURCES = scip_solver.c
OBJECTS = $(SOURCES:.c=.o)

# Target executable and shared library
TARGET = scip_solver
SHARED_LIB = scip_solver.so

# Default target - build both executable and shared library
all: $(TARGET) $(SHARED_LIB)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LIBDIRS) $(LIBS) -o $(TARGET)

# Build the shared library for Python wrapper
$(SHARED_LIB): $(OBJECTS)
	$(CC) -shared -fPIC $(OBJECTS) $(LIBDIRS) $(LIBS) -o $(SHARED_LIB)

# Build object files with position-independent code for shared library
%.o: %.c
	$(CC) $(CFLAGS) -fPIC $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET) $(SHARED_LIB) *.lp

# Install dependencies (macOS with Homebrew)
install-deps:
	@echo "Installing SCIP dependencies via Homebrew..."
	brew install scip

# Check SCIP installation
check-scip:
	@echo "Checking SCIP installation..."
	@if [ -d "$(SCIP_PATH)" ]; then \
		echo "SCIP found at $(SCIP_PATH)"; \
		echo "Include directory: $(SCIP_PATH)/include"; \
		echo "Library directory: $(SCIP_PATH)/lib"; \
		ls -la $(SCIP_PATH)/lib/libscip* 2>/dev/null || echo "Warning: libscip not found in lib directory"; \
	else \
		echo "SCIP not found at $(SCIP_PATH)"; \
		echo "Please check your SCIP installation path"; \
		exit 1; \
	fi

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Run with valgrind for memory checking
valgrind: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build the executable (default)"
	@echo "  clean       - Remove build files"
	@echo "  debug       - Build with debug symbols"
	@echo "  run         - Build and run the program"
	@echo "  valgrind    - Run with valgrind memory checker"
	@echo "  check-scip  - Check SCIP installation"
	@echo "  install-deps - Install SCIP via Homebrew"
	@echo "  help        - Show this help message"

# Run Python test with shared library
test-python: $(SHARED_LIB)
	python3 scip_c_wrapper.py

# Phony targets
.PHONY: all clean debug run valgrind check-scip install-deps help test-python

# Alternative library configurations for different systems
# Uncomment the appropriate section for your system

# For Linux systems (uncomment if needed):
# LIBS = -lscip -lsoplex -ltbb -lgmp -lzimpl -lreadline -lncurses -lz -lm -ldl

# For systems without some optional libraries (minimal build):
# LIBS = -lscip -lsoplex -lgmp -lz -lm

# Static linking (uncomment if needed):
# LIBS = -static -lscip -lsoplex -ltbb -lgmp -lzimpl -lreadline -lz -lm