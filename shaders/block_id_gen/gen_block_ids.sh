#!/bin/bash
cd $(echo "$0" | sed 's/[^/]*$//')
./blockmapping.sh old_block.properties blockstatelist.txt
linecount=$(cat id_map.txt | tail -n 1 | sed 's/\(.*\):.*$/(\1 + 511)\/512/' | bc)
printf "P3
512 ${linecount}
255
0 0 0 " > blockIdMap.ppm
mapped_pairs=$(cat id_map.txt | sed 's/^.*:\(.*\)$/\1%256!\1\/256!0/' | tr '!' '\n' | bc)
n=0
for pair in ${mapped_pairs}
do
    printf "%s " ${pair} >> blockIdMap.ppm
    n=$(( $n+1 ))
    if [ $n -eq 1536 ]
    then
        printf "
" >> blockIdMap.ppm
        n=0
    fi
done
while ! [ $n -eq 1536 ]
do
    printf "0 " >> blockIdMap.ppm
    n=$(( n+1 ))
done
magick blockIdMap.ppm blockIdMap.png
mv new_block.properties ../block.properties
mv blockIdMap.png ../lib/textures/blockIdMap.png
