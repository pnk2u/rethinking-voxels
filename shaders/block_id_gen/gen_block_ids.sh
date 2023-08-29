#!/bin/bash
./blockmapping.sh old_block.properties blockstatelist.txt
echo "int blockIdMap[] = int[](
" > ../lib/vx/blockIdMap.glsl
cat id_map.txt | sed 's/.*://' | tr '\n' ',' | sed 's/^,//' | sed 's/,$//' | sed 's/\([0-9]*,[0-9]*,[0-9]*,[0-9]*,[0-9]*,\)/\1\n/' >> ../lib/vx/blockIdMap.glsl
echo ");" >> ../lib/vx/blockIdMap.glsl
