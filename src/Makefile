#BLOCK=32
#TIME=0
#BITS=10
#EXECS=10
#SEGS=1048576
#SEGS=32768

all: cpu mergeseg radixseg fixcub fixpass fixpassdiff fixthrust fixmergemgpu fixbitonic fixmerge fixoddeven fixquick bitonicseg 
#bitonicsort mergesort oddevensort quicksort

generate: gen_equal.cpp gen_diff.cpp
	g++ -O3 gen_equal.cpp -o equal.exe -DEXP_BITS_SIZE=$(BITS) -DRAND
	g++ -O3 gen_diff.cpp -o diff.exe -DEXP_BITS_SIZE=$(BITS)
	g++ -O3 parser.cpp -o parser.exe -std=c++11 -DSEGMENTS=$(SEGS)

gensortasc: gen_equal.cpp gen_diff.cpp
	g++ -O3 gen_equal.cpp -o equal.exe -DEXP_BITS_SIZE=$(BITS) -DSORTASC
	g++ -O3 gen_diff.cpp -o diff.exe -DEXP_BITS_SIZE=$(BITS)

gensortdesc: gen_equal.cpp gen_diff.cpp
	g++ -O3 gen_equal.cpp -o equal.exe -DEXP_BITS_SIZE=$(BITS) -DSORTDESC
	g++ -O3 gen_diff.cpp -o diff.exe -DEXP_BITS_SIZE=$(BITS)

genrandminmax: gen_equal.cpp gen_diff.cpp
	g++ -O3 gen_equal.cpp -o equal.exe -DEXP_BITS_SIZE=$(BITS) -DRANDMINMAX
	g++ -O3 gen_diff.cpp -o diff.exe -DEXP_BITS_SIZE=$(BITS)

cpu:	
	g++ -O3 cpu.cpp -o cpu.exe -std=c++11 -DELAPSED_TIME=$(TIME)

fixmergemgpu: fixmergemgpu.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 fixmergemgpu.cu -o fixmergemgpu.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixcub: fixcub.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 fixcub.cu -o fixcub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)
 
fixthrust: fixthrust.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 fixthrust.cu -o fixthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixthrustkernel: fixthrustkernel.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 fixthrustkernel.cu -o fixthrustkernel.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

bitonicseg: bitonicseg/main.cpp
	make clean -C bitonicseg
	make -C bitonicseg

