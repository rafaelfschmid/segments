#!/bin/bash
dir=$1
c=32768
while [ $c -le 67108864 ]
do
	#echo $c 
	d=1
	while [ $d -le  32768 ] 
	do
		i=1
		while [ $i -le  10 ] 
		do
			./diff.exe $d $c > $dir/$d"_"$c"_"$i".in"
			sleep 1
			((i=$i+1))
		done
		((d=$d*2))
	done
	((c=$c*2))
done
