# Device can be p100, v100, bdw, skx, tx2
DEVICES = $(DEVICE)
# Nvidia TeslaP100 on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring p100,$(DEVICES)))
CXXFLAGS=-O3 -I/apps/t3/sles12sp2/cuda/8.0.61/include -ta=nvidia:cc60 -Minfo -std=c++11 -DOWN_INDEX_SEQUENCE -DNO_ASSERT_IN_CONSTEXPR -DENABLE_OPENACC
CXX=pgc++
LDFLAGS = -Mcudalib=cufft -ta=nvidia:cc60 -acc
TARGET = vlp4d.p100_acc
endif

# Intel Broadwell on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring bdw,$(DEVICES)))
CXXFLAGS=-O3 -std=c++11 -qopenmp -xCORE-AVX2 -DOWN_INDEX_SEQUENCE -DNO_ASSERT_IN_CONSTEXPR
CXX=icpc
LDFLAGS = -lm -lfftw3_threads -lfftw3 -qopenmp
TARGET = vlp4d.bdw_omp
endif

# Nvidia TeslaV100 on Summit (OLCF, US)
ifneq (,$(findstring v100,$(DEVICES)))
CXXFLAGS=-O3 -ta=nvidia:cc70 -Minfo -std=c++11 -DOWN_INDEX_SEQUENCE -DNO_ASSERT_IN_CONSTEXPR -DENABLE_OPENACC
CXX=pgc++
LDFLAGS = -Mcudalib=cufft -ta=nvidia:cc70 -acc
TARGET = vlp4d.v100_acc
endif

# Intel Skylake on JFRS-1 (IFERC-CSC, Japan)
ifneq (,$(findstring skx,$(DEVICES)))
CXXFLAGS=-O3 -std=c++11 -qopenmp -xCORE-AVX512 -DOWN_INDEX_SEQUENCE -DNO_ASSERT_IN_CONSTEXPR
CXX=CC
LDFLAGS = -lfftw3_threads -lfftw3 -qopenmp
TARGET = vlp4d.skx_omp
endif

# Marvell Thunder X2 on CEA Computing Complex (CEA, France)
ifneq (,$(findstring tx2,$(DEVICES)))
CXXFLAGS=-O3 -std=c++11 -fopenmp -armpl -mcpu=native -DOWN_INDEX_SEQUENCE -DNO_ASSERT_IN_CONSTEXPR 
CXX=armclang++
LDFLAGS=-fopenmp -armpl -mcpu=native
# Execution files
TARGET = vlp4d.tx2_omp
endif

# Linker
LINKER = $(CXX)

# Source files
SRCS = $(wildcard ./*.cpp)

# Object files
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

# Targets
all: $(TARGET)

$(TARGET) : $(OBJS)
	$(LINKER) -o $@ $^ $(LDFLAGS)
	mv $(TARGET) ../wk

# Compiler .cpp files to create object files
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.SUFFIXES: .c .cpp .cu .o
.PHONY: all clean
.PHONY: clean

clean:
	rm -f ../wk/vlp4d.*acc ../wk/vlp4d.*omp *.o
