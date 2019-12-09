# Device can be p100, v100, bdw, skx, tx2
KOKKOS_DEVICES = $(DEVICE)
SRC = $(wildcard ./*.cpp)
vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

# Nvidia TeslaP100 on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring p100,$(KOKKOS_DEVICES)))
KOKKOS_PATH = /home/1/17IKA143/work2019/lib/kokkos
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -O3
ifneq (,$(findstring 3D,$(RANGE_POLICY)))
  # In order to set tile sizes at compile time
  CXXFLAGS += -DPOLICY_3D
  CXXFLAGS += -DTILE_X0=32 -DTILE_Y0=4 -DTILE_Z0=2
  CXXFLAGS += -DTILE_X1=32 -DTILE_Y1=4 -DTILE_Z1=2
  CXXFLAGS += -DTILE_X2=32 -DTILE_Y2=4 -DTILE_Z2=2
  CXXFLAGS += -DTILE_X3=32 -DTILE_Y3=4 -DTILE_Z3=2
endif
LINK = ${CXX}
LDFLAGS = -lcufft
EXE = vlp4d.p100_kokkos
KOKKOS_DEVICES = "Cuda,OpenMP"
KOKKOS_ARCH = "BDW,Pascal60"
KOKKOS_CUDA_OPTIONS += "enable_lambda"
endif

# Intel Broadwell on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring bdw,$(KOKKOS_DEVICES)))
KOKKOS_PATH = /home/1/17IKA143/work2019/lib/kokkos
CXX = icpc
CXXFLAGS = -O3
ifneq (,$(findstring 3D,$(RANGE_POLICY)))
  # In order to set tile sizes at compile time
  CXXFLAGS += -DPOLICY_3D
  CXXFLAGS += -DTILE_X0=16 -DTILE_Y0=1  -DTILE_Z0=1
  CXXFLAGS += -DTILE_X1=1  -DTILE_Y1=32 -DTILE_Z1=4
  CXXFLAGS += -DTILE_X2=1  -DTILE_Y2=1  -DTILE_Z2=8
  CXXFLAGS += -DTILE_X3=1  -DTILE_Y3=1  -DTILE_Z3=16
endif
LINK = ${CXX}
LDFLAGS = -lfftw3_threads -lfftw3
EXE = vlp4d.bdw_kokkos
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "BDW"
endif

# Nvidia TeslaV100 on Summit (OLCF, US)
ifneq (,$(findstring v100,$(KOKKOS_DEVICES)))
KOKKOS_PATH = /ccs/home/yasahi/lib/kokkos
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -O3
ifneq (,$(findstring 3D,$(RANGE_POLICY)))
  # In order to set tile sizes at compile time
  CXXFLAGS += -DPOLICY_3D
  CXXFLAGS += -DTILE_X0=32 -DTILE_Y0=4 -DTILE_Z0=2
  CXXFLAGS += -DTILE_X1=32 -DTILE_Y1=4 -DTILE_Z1=2
  CXXFLAGS += -DTILE_X2=32 -DTILE_Y2=4 -DTILE_Z2=2
  CXXFLAGS += -DTILE_X3=32 -DTILE_Y3=4 -DTILE_Z3=2
endif
LINK = ${CXX}
LDFLAGS = -lcufft
EXE = vlp4d.v100_kokkos
KOKKOS_DEVICES = "Cuda,OpenMP"
KOKKOS_ARCH = "Power9,Volta70"
KOKKOS_CUDA_OPTIONS += "enable_lambda"
endif

# Intel Skylake on JFRS-1 (IFERC-CSC, Japan)
ifneq (,$(findstring skx,$(KOKKOS_DEVICES)))
KOKKOS_PATH = /home/asahiy/lib/kokkos
CXX = CC
CXXFLAGS = -O3
ifneq (,$(findstring 3D,$(RANGE_POLICY)))
  # In order to set tile sizes at compile time
  CXXFLAGS += -DPOLICY_3D
  CXXFLAGS += -DTILE_X0=16 -DTILE_Y0=1  -DTILE_Z0=1
  CXXFLAGS += -DTILE_X1=1  -DTILE_Y1=32 -DTILE_Z1=4
  CXXFLAGS += -DTILE_X2=1  -DTILE_Y2=1  -DTILE_Z2=8
  CXXFLAGS += -DTILE_X3=1  -DTILE_Y3=1  -DTILE_Z3=16
endif
LINK = ${CXX}
LDFLAGS = -lfftw3_threads -lfftw3
EXE = vlp4d.skx_kokkos
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "SKX"
endif

ifneq (,$(findstring tx2,$(KOKKOS_DEVICES)))
# Marvell Thunder X2 on CEA Computing Complex (CEA, France)
KOKKOS_PATH = /ccc/home/cont002/qst/asahiy/lib/src/kokkos
CXX=armclang++
CXXFLAGS=-O3 -armpl
ifneq (,$(findstring 3D,$(RANGE_POLICY)))
  # In order to set tile sizes at compile time
  CXXFLAGS += -DTILE_X0=16 -DTILE_Y0=1  -DTILE_Z0=1
  CXXFLAGS += -DTILE_X1=1  -DTILE_Y1=32 -DTILE_Z1=4
  CXXFLAGS += -DTILE_X2=1  -DTILE_Y2=1  -DTILE_Z2=8
  CXXFLAGS += -DTILE_X3=1  -DTILE_Y3=1  -DTILE_Z3=16
endif
LINK = ${CXX}
LDFLAGS=-armpl
# Execution files
EXE = vlp4d.tx2_kokkos
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "ARMv8-TX2"
endif

KOKKOS_SRC_PATH = ${KOKKOS_PATH}
DEPFLAGS = -M

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

test: $(EXE)
	./$(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE) $(LDFLAGS)
	mv $(EXE) ../wk

clean: kokkos-clean 
	rm -f *.o vlp4d.*kokkos ../wk/vlp4d.*kokkos

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)
