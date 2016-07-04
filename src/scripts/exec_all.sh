in=$1 # input files dir
out=$2 # time files dir

./scripts/exec.sh fixmergemgpu.exe $in > $out/fixmergemgpu.time
<<<<<<< HEAD
#/scripts/exec.sh fixpassdiff.exe $in > $out/fixpassdiff.time
=======
./scripts/exec.sh fixpassdiff.exe $in > $out/fixpassdiff.time
>>>>>>> b2c76e6e4a9415dd0d267c8abba0246c08332396
./scripts/exec.sh fixpass.exe $in > $out/fixpass.time
./scripts/exec.sh fixsort/bitonicseg/bitonicseg.exe $in > $out/bitonicseg.time
./scripts/exec.sh mergeseg.exe $in > $out/mergeseg.time
./scripts/exec.sh radixseg.exe $in > $out/radixseg.time
./scripts/exec.sh fixcub.exe $in > $out/fixcub.time
./scripts/exec.sh fixthrust.exe $in > $out/fixthrust.time
<<<<<<< HEAD
#./scripts/exec.sh nthrust.exe $in > $out/nthrust.time
=======
>>>>>>> b2c76e6e4a9415dd0d267c8abba0246c08332396
./scripts/exec.sh fixsort/bitonic/bitonicsort.exe $in > $out/fixbitonic.time
./scripts/exec.sh fixsort/mergesort/mergesort.exe $in > $out/fixmerge.time
./scripts/exec.sh fixsort/oddevensort/oddevensort.exe $in > $out/fixoddeven.time
./scripts/exec.sh fixsort/quicksort/quicksort.exe $in > $out/fixquick.time


