#!/bin/bash
mv blockstatelist.txt lastblockstatelist.txt
for F in blockstates/*
do
	modeltype=$(cat $F | sed '2!d' | sed 's/[^a-z]//g')
	blockname=$(echo $F | sed 's/^.*\///' | sed 's/\..*$//')
	if [ "${modeltype}" == "variants" ]
	then
		localstates=$(cat $F | tr '{' '\n' | grep -e '=' | sed 's/[ ":\{\[]//g' | sed 's/,/:/g')
		hasstates="no"
		for state in ${localstates}
		do
			hasstates="yes"
			echo "${blockname}:${state}" >> blockstatelist.txt
		done
		if [ ${hasstates} == "no" ]
		then
			echo "${blockname}" >> blockstatelist.txt
		fi
	else
		wrotesomething="no"
		localstates=$(cat $F | sed 's/^ *//' | grep -e '"[a-z_0-9]*": "[a-z_0-9]*"' | sed 's/": "/=/' | tr ',' '\n' | tr -d '"')
		echo $blockname $localstates
	    localbasestates=$(echo ${localstates} | tr ' ' '\n' | sed 's/=.*$//' | sort -u)
		for basestate in ${localbasestates}
		do
			missingtypes=""
			basestatevarname=${basestate}
			declare ${basestatevarname}=""
			for state in ${localstates}
			do
				otherbasestate=$(echo ${state} | sed 's/=.*//')
				if [ "${basestate}" == "${otherbasestate}" ]
				then
					statetype="$(echo $state | sed 's/.*=//' | tr '|' ' ')"
					for singlestatetype in $statetype
					do
						if [ "${singlestatetype}" == "true" ]
						then
							missingtypes="${missingtypes} false"
						elif [ "${singlestatetype}" == "false" ]
						then
							missingtypes="${missingtypes} true"
						elif [ "${singlestatetype}" -eq "${singlestatetype}" ] 2>/dev/null
						then
							missingtypes="${missingtypes} 0"
						else
							missingtypes="${missingtypes} none"
						fi
					done
				    eval "${basestatevarname}"='"${statetype} ${!basestatevarname}"'
				fi
			done
			eval "${basestatevarname}"='"${!basestatevarname} ${missingtypes}"'
			eval "${basestatevarname}"='$(echo ${!basestatevarname} | tr " " "\n" | sort -u | tr "\n" " ")'
		done
		printf "%s              \r" ${blockname}

		n=0
		for basestate in ${localbasestates}
		do
			declare basestate$n=$basestate
			n=$(( n+1 ))
		done
		while (( $n < 7 ))
		do
			declare basestate$n="none"
			n=$(( n+1 ))
		done
		none="NONE"
		for basestate0type in ${!basestate0}
		do
			for basestate1type in ${!basestate1}
			do
				for basestate2type in ${!basestate2}
				do
					for basestate3type in ${!basestate3}
					do
						for basestate4type in ${!basestate4}
						do
							for basestate5type in ${!basestate5}
							do
								for basestate6type in ${!basestate6}
								do
									echo "${blockname}:${basestate0}=${basestate0type}:${basestate1}=${basestate1type}:${basestate2}=${basestate2type}:${basestate3}=${basestate3type}:${basestate4}=${basestate4type}:${basestate5}=${basestate5type}:${basestate6}=${basestate6type}" | sed 's/:[a-z0-9_]*=NONE//g' >> blockstatelist.txt
									wrotesomething="yes"
								done
							done	
						done
					done	
				done	
			done
		done
		if [ $wrotesomething == "no" ]
		then
			echo "failed to write ${blockname}!" 1>&2
			echo ${blockname} >> blockstatelist.txt
		fi
	fi
done
