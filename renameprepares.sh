for i in 8 7 6 5 4 3 2 1
do
	echo "asd"
	for F in $(find . -name "prepare$i*")
	do
		FIND="prepare$i"
		REPLACE="prepare$(expr $i + 1)"
		cat $F | sed "s/$FIND/$REPLACE/" > $(echo $F | sed "s/$FIND/$REPLACE/")
	done
done
