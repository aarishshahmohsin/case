# Makefile for SCIP C API Wide-Reach Classification Solver on Linux

# SCIP installation path (customized to your system)
SCIP_PATH = /home/aarish/scip_install_try/scipoptsuite-9.2.2/scip
# SCIP_PATH = /home/aarish/scip_cplex/scipoptsuite-9.2.2/scip

# Compiler settings
CC = gcc
CXX = g++

# Compiler flags
CFLAGS = -O3 -Wall -Wextra -std=c99
CXXFLAGS = -O3 -Wall -Wextra -std=c++11

# Include directories
INCLUDES = -I$(SCIP_PATH)/src -I$(SCIP_PATH)/obj/shared/O.linux.x86_64.gnu.opt/include

# Library directories
LIBDIRS = -L$(SCIP_PATH)/lib/shared

# Libraries to link (no Zimpl here)
LIBS = -lscip -ltbb -lgmp -lreadline -lncurses -lz -lm -ldl

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

# Build object files with position-independent code
%.o: %.c
	$(CC) $(CFLAGS) -fPIC $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET) $(SHARED_LIB) *.lp

# Check SCIP installation
check-scip:
	@echo "Checking SCIP installation..."
	@if [ -d "$(SCIP_PATH)" ]; then \
		echo "SCIP found at $(SCIP_PATH)"; \
		echo "Include directory: $(SCIP_PATH)/src"; \
		echo "Library directory: $(SCIP_PATH)/lib/shared"; \
		ls -la $(SCIP_PATH)/lib/shared/libscip* 2>/dev/null || echo "Warning: libscip not found"; \
	else \
		echo "SCIP not found at $(SCIP_PATH)"; \
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
	@echo "  all         - Build the executable and shared lib"
	@echo "  clean       - Remove build files"
	@echo "  debug       - Build with debug symbols"
	@echo "  run         - Build and run the program"
	@echo "  valgrind    - Run with valgrind memory checker"
	@echo "  check-scip  - Check SCIP installation"
	@echo "  help        - Show this help message"

# Run Python test with shared library
test-python: $(SHARED_LIB)
	python3 scip_c_wrapper.py

# Phony targets
.PHONY: all clean debug run valgrind check-scip help test-python
