#!/bin/bash
dir=$1
c=32768
while [ $c -le 67108864 ]
do
	#echo $c 
	d=1
	while [[ $d -lt $c && $d -le 1048576 ]] 
	do
		./diff.exe $d $c > $dir/$d"_"$c.in
		((d=$d*2))
		sleep 1.0
	done
	((c=$c*2))
done
