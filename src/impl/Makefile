libs=-I"../../lib"
params=-std=c++11 --expt-extended-lambda -lcuda
gen=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52


all: cpu fixcub fixthrust bitonicseg

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
	nvcc $(gen) fixmergemgpu.cu -o fixmergemgpu.exe $(libs) $(params) -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixcub: fixcub.cu
	nvcc $(gen) fixcub.cu -o fixcub.exe $(libs) $(params) -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)
 
fixthrust: fixthrust.cu
	nvcc $(gen) fixthrust.cu -o fixthrust.exe $(params) -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

bitonic: bitonicseg/main.cpp
#	make clean -C bitonicseg
	make -C bitonicseg
