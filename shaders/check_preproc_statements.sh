#!/bin/bash
for F in $(find .)
do
	if [ -d $F ]
	then
		continue
	fi
	N1=$(cat "$F" | sed 's/[[:blank:]]//g' | sed '/^[^#]/d' | grep -e "#if" | wc -l)
	N2=$(cat "$F" | sed 's/[[:blank:]]//g' | sed '/^[^#]/d' | grep -e "#endif" | wc -l)
	if [ "$N1" != "$N2" ]
	then
		echo $F $N1 $N2
	fi
done
