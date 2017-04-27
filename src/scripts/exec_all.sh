in=$1 # input files dir
out=$2 # time files dir

./scripts/exec.sh fixmergemgpu.exe $in > $out/fixmergemgpu.time
#/scripts/exec.sh fixpassdiff.exe $in > $out/fixpassdiff.time
./scripts/exec.sh fixpass.exe $in > $out/fixpass.time
./scripts/exec.sh bitonicseg/bitonicseg.exe $in > $out/bitonicseg.time
./scripts/exec.sh mergeseg.exe $in > $out/mergeseg.time
./scripts/exec.sh radixseg.exe $in > $out/radixseg.time
./scripts/exec.sh fixcub.exe $in > $out/fixcub.time
./scripts/exec.sh fixthrust.exe $in > $out/fixthrust.time
#./scripts/exec.sh nthrust.exe $in > $out/nthrust.time


