#!/bin/bash
dir=$1 # output files dir
c=32768
<<<<<<< HEAD
while [ $c -le 67108864 ]
do
	#echo $c 
	d=1
	while [[ $d -lt $c && $d -le 128 ]]
=======
while [ $c -le 134217728 ]
do
	#echo $c 
	d=1
	while [[ $d -lt $c && $d -le 1048576 ]]
>>>>>>> b2c76e6e4a9415dd0d267c8abba0246c08332396
	do
		((x=$c/$d))
		./equal.exe $d $x > $dir/$d"_"$c.in
		#echo $d"_"$c".in"
		((d=$d*2))
		sleep 1.0
	done
	((c=$c*2))
done
