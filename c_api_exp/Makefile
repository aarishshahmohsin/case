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

# Check if heuristic files exist and conditionally include them
HEUR_EXISTS := $(shell test -f heur_directionalrounding.c && test -f heur_directionalrounding.h && echo yes)

ifeq ($(HEUR_EXISTS),yes)
    SOURCES = scip_solver.c heur_directionalrounding.c
    HEADERS = heur_directionalrounding.h
else
    SOURCES = scip_solver.c
    HEADERS = 
endif

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

# Conditional rules based on whether heuristic files exist
ifeq ($(HEUR_EXISTS),yes)
# Specific rule for the main solver (depends on heuristic header)
scip_solver.o: scip_solver.c heur_directionalrounding.h
	$(CC) $(CFLAGS) -fPIC $(INCLUDES) -c scip_solver.c -o scip_solver.o

# Specific rule for the custom heuristic
heur_directionalrounding.o: heur_directionalrounding.c heur_directionalrounding.h
	$(CC) $(CFLAGS) -fPIC $(INCLUDES) -c heur_directionalrounding.c -o heur_directionalrounding.o
else
# Rule for main solver without heuristic dependency
scip_solver.o: scip_solver.c
	$(CC) $(CFLAGS) -fPIC $(INCLUDES) -c scip_solver.c -o scip_solver.o
endif

# Create template files for custom heuristic if they don't exist
create-heuristic:
	@if [ ! -f heur_directionalrounding.h ]; then \
		echo "Creating heur_directionalrounding.h..."; \
		echo '#ifndef __SCIP_HEUR_DIRECTIONALROUNDING_H__' > heur_directionalrounding.h; \
		echo '#define __SCIP_HEUR_DIRECTIONALROUNDING_H__' >> heur_directionalrounding.h; \
		echo '' >> heur_directionalrounding.h; \
		echo '#include "scip/scip.h"' >> heur_directionalrounding.h; \
		echo '' >> heur_directionalrounding.h; \
		echo '#ifdef __cplusplus' >> heur_directionalrounding.h; \
		echo 'extern "C" {' >> heur_directionalrounding.h; \
		echo '#endif' >> heur_directionalrounding.h; \
		echo '' >> heur_directionalrounding.h; \
		echo 'SCIP_RETCODE SCIPincludeHeurDirectionalrounding(SCIP* scip);' >> heur_directionalrounding.h; \
		echo '' >> heur_directionalrounding.h; \
		echo '#ifdef __cplusplus' >> heur_directionalrounding.h; \
		echo '}' >> heur_directionalrounding.h; \
		echo '#endif' >> heur_directionalrounding.h; \
		echo '' >> heur_directionalrounding.h; \
		echo '#endif' >> heur_directionalrounding.h; \
	fi
	@if [ ! -f heur_directionalrounding.c ]; then \
		echo "Creating heur_directionalrounding.c..."; \
		echo '#include "scip/scip.h"' > heur_directionalrounding.c; \
		echo '#include "heur_directionalrounding.h"' >> heur_directionalrounding.c; \
		echo '' >> heur_directionalrounding.c; \
		echo '#define HEUR_NAME             "directionalrounding"' >> heur_directionalrounding.c; \
		echo '#define HEUR_DESC             "directional rounding heuristic"' >> heur_directionalrounding.c; \
		echo '#define HEUR_DISPCHAR         '\''D'\''' >> heur_directionalrounding.c; \
		echo '#define HEUR_PRIORITY         1000' >> heur_directionalrounding.c; \
		echo '#define HEUR_FREQ             1' >> heur_directionalrounding.c; \
		echo '#define HEUR_FREQOFS          0' >> heur_directionalrounding.c; \
		echo '#define HEUR_MAXDEPTH         -1' >> heur_directionalrounding.c; \
		echo '#define HEUR_TIMING           SCIP_HEURTIMING_AFTERLPPLUNGE' >> heur_directionalrounding.c; \
		echo '#define HEUR_USESSUBSCIP      FALSE' >> heur_directionalrounding.c; \
		echo '' >> heur_directionalrounding.c; \
		echo 'static SCIP_DECL_HEUREXEC(heurExecDirectionalrounding)' >> heur_directionalrounding.c; \
		echo '{' >> heur_directionalrounding.c; \
		echo '   *result = SCIP_DIDNOTRUN;' >> heur_directionalrounding.c; \
		echo '   return SCIP_OKAY;' >> heur_directionalrounding.c; \
		echo '}' >> heur_directionalrounding.c; \
		echo '' >> heur_directionalrounding.c; \
		echo 'SCIP_RETCODE SCIPincludeHeurDirectionalrounding(SCIP* scip)' >> heur_directionalrounding.c; \
		echo '{' >> heur_directionalrounding.c; \
		echo '   SCIP_HEUR* heur = NULL;' >> heur_directionalrounding.c; \
		echo '' >> heur_directionalrounding.c; \
		echo '   SCIP_CALL( SCIPincludeHeurBasic(scip, &heur,' >> heur_directionalrounding.c; \
		echo '         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,' >> heur_directionalrounding.c; \
		echo '         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP,' >> heur_directionalrounding.c; \
		echo '         heurExecDirectionalrounding, NULL) );' >> heur_directionalrounding.c; \
		echo '' >> heur_directionalrounding.c; \
		echo '   assert(heur != NULL);' >> heur_directionalrounding.c; \
		echo '' >> heur_directionalrounding.c; \
		echo '   return SCIP_OKAY;' >> heur_directionalrounding.c; \
		echo '}' >> heur_directionalrounding.c; \
	fi

# Build without custom heuristic (fallback option)
build-without-heuristic:
	$(MAKE) SOURCES=scip_solver.c all

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET) $(SHARED_LIB) *.lp

# Clean everything including generated heuristic files
clean-all: clean
	rm -f heur_directionalrounding.h heur_directionalrounding.c

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

# Check if heuristic files exist
check-heuristic:
	@echo "Checking custom heuristic files..."
	@if [ -f "heur_directionalrounding.h" ] && [ -f "heur_directionalrounding.c" ]; then \
		echo "Custom heuristic files found."; \
	else \
		echo "Custom heuristic files missing. Run 'make create-heuristic' first."; \
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
	@echo "  all                    - Build the executable and shared lib"
	@echo "  create-heuristic       - Create template heuristic files"
	@echo "  build-without-heuristic- Build without custom heuristic"
	@echo "  check-heuristic        - Check if heuristic files exist"
	@echo "  clean                  - Remove build files"
	@echo "  clean-all              - Remove build files and heuristic templates"
	@echo "  debug                  - Build with debug symbols"
	@echo "  run                    - Build and run the program"
	@echo "  valgrind               - Run with valgrind memory checker"
	@echo "  check-scip             - Check SCIP installation"
	@echo "  help                   - Show this help message"

# Run Python test with shared library
test-python: $(SHARED_LIB)
	python3 scip_c_wrapper.py

# Phony targets
.PHONY: all clean clean-all debug run valgrind check-scip help test-python create-heuristic build-without-heuristic check-heuristic