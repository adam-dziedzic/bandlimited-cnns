# cxx
CXX := g++
CXXFLAGS :=
CXXFLAGS += -g
#CXXFLAGS += -O3
CXXFLAGS += -std=c++11
#CXXFLAGS += -fopenmp

# cc
CC := gcc
CFLAGS :=
CFLAGS += -g

# nvcc
NVCC      := nvcc
NVCCFLAGS :=
NVCCFLAGS += -G
NVCCFLAGS += -gencode arch=compute_20,code=sm_20 \
             -gencode arch=compute_20,code=sm_21 \
             -gencode arch=compute_30,code=sm_30 \
             -gencode arch=compute_35,code=sm_35 \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_50,code=compute_50

# ld
LDFLAGS  :=
LDLIBS   :=

# dep
C_HDRS   := $(shell find . -name "*.h")
C_SRCS   := $(shell find . -name "*.c")
CXX_HDRS := $(shell find . -name "*.hpp")
CXX_SRCS := $(shell find . -name "*.cpp")
CU_SRCS  := $(shell find . -name "*.cu")
DEP      := $(C_HDRS) $(C_SRCS) $(CXX_HDRS) $(CXX_SRCS) $(CU_SRCS)

# build
BUILD := build
OBJS  := complex_mul_profile
OBJS  += $(patsubst %.c,$(BUILD)/%.c.o,$(C_SRCS))
OBJS  += $(patsubst %.cpp,$(BUILD)/%.cpp.o,$(CXX_SRCS))
OBJS  += $(patsubst %.cu,$(BUILD)/%.cu.o,$(CU_SRCS))
BIN   := # TODO

#########
# rules #
#########

.PHONY: pre test* clean

all: $(BIN)

pre:
	mkdir -p $(BUILD)

$(BUILD)/%.c.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

$(BUILD)/%.cpp.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(BUILD)/%.cu.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

%.c: %.h

%.cpp: %.hpp

%.cu: %.hpp

$(BIN): pre $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(BUILD)/$@ $(OBJS) $(LDFLAGS) $(LDLIBS)

test: all

clean:
	rm -rf $(BUILD)
