// N>1
if (participateInSorting) {
    flipPair(index, 0, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();


// N>2
if (participateInSorting) {
    flipPair(index, 1, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 0, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();


// N>4
if (participateInSorting) {
    flipPair(index, 2, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 1, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 0, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();


// N>8
if (participateInSorting) {
    flipPair(index, 3, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 2, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 1, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 0, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();


// N>16
if (participateInSorting) {
    flipPair(index, 4, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 3, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 2, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 1, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 0, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();


// N>32
if (participateInSorting) {
    flipPair(index, 5, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 4, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 3, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 2, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 1, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 0, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();


// N>64
if (participateInSorting) {
    flipPair(index, 6, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 5, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 4, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 3, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 2, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 1, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 0, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();


// N>128
if (participateInSorting) {
    flipPair(index, 7, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 6, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 5, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 4, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 3, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 2, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 1, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 0, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();


// N>256
if (participateInSorting) {
    flipPair(index, 8, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 7, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 6, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 5, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 4, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 3, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 2, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 1, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
if (participateInSorting) {
    dispersePair(index, 0, meanPos, meanNormal);
}
barrier();
memoryBarrierShared();
