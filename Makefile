# Makefile for C++ project

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -pthread

# Source files
SRCS = annealer.cpp parser.cpp
HEADERS = parser.hh

# Executable name
TARGET = annealer

# Default target
all: $(TARGET)

# Rule to create the executable
$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean up generated files
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean run
