TARGET = kmeans

# Where is the Altera SDK for OpenCL software?
# ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
# $(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation)
# endif
# ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
# $(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation.)
# endif

SRCS = main.cpp kmeans.cpp fpga_kmeans.cpp
SRCS_FILES = $(foreach F, $(SRCS), $(F))
OBJS=$(SRCS:.c=.o)
COMMON_FILES = ./common/src/AOCL_Utils.cpp
CXX_FLAGS = -lm -O3

# arm cross compiler
# CROSS-COMPILE = arm-linux-gnueabihf-

# OpenCL compile and link flags.
# AOCL_COMPILE_CONFIG=$(shell aocl compile-config --arm) -I./common/inc 
# AOCL_LINK_CONFIG=$(shell aocl link-config --arm) 


all:
	# $(CROSS-COMPILE)g++ $(SRCS_FILES) $(COMMON_FILES) -g -o $(TARGET)  $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) 
	clang -framework OpenCL -DAPPLE -O3 -Wno-deprecated-declarations $(EXTRA_CFLAGS) $(SRCS_FILES) -o $(TARGET)  $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG)

# kmeans.aocx:
	# aoc kmeans.cl -o kmeans.aocx --board de1soc_sharedonly

clean:
	@rm -f *.o $(TARGET) data means
