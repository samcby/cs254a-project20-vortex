ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk
PROJECT := cost_pipeline
SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)
SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp $(SRC_DIR)/relu.cpp
VX_CFLAGS += -DDXA_DOUBLE_BUFFER
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
KERNEL_LIB := vortex2
include ../common.mk
