bool isEmissive(int mat) {
	return (
		mat == 1234  || // generic light source
		mat == 1235  || // generic light source (fallback colour)
		mat == 10024 || // brewing stand
		mat == 10056 || // lava cauldron
		mat == 10068 || // lava
		mat == 10072 || // fire
		mat == 10076 || // soul fire
		mat == 10216 || // crimson wood
		mat == 10224 || // warped wood
#if GLOWING_ORE_MASTER == 2 || (GLOWING_ORE_MASTER == 1 && SHADER_STYLE == 4)
		mat == 10272 || // iron ore
		mat == 10276 ||
		mat == 10284 || // copper ore
		mat == 10288 ||
		mat == 10300 || // gold ore
		mat == 10304 ||
		mat == 10320 || // diamond ore
		mat == 10324 ||
		mat == 10340 || // emerald ore
		mat == 10344 ||
		mat == 10356 || // lapis ore
		mat == 10360 ||
		mat == 10612 || // redstone ore
		mat == 10620 ||
#endif
		mat == 10616 || // lit redstone ore
		mat == 10624 ||
#ifdef EMISSIVE_EMERALD_BLOCK
		mat == 10336 || // emerald block
#endif
#ifdef EMISSIVE_LAPIS_BLOCK
		mat == 10352 || // lapis block
#endif
#ifdef EMISSIVE_REDSTONE_BLOCK
		mat == 10608 || // redstone block
#endif
		mat == 10332 || // amethyst buds
		mat == 10388 || // blue ice
		mat == 10396 || // jack o'lantern
		mat == 10400 || // 1-2 waterlogged sea pickles
		mat == 10401 || // 3-4 waterlogged sea pickles
		mat == 10412 || // glowstone
		mat == 10448 || // sea lantern
		mat == 10452 || // magma block
		mat == 10476 || // crying obsidian
		mat == 10496 || // torch
		mat == 10497 ||
		mat == 10500 || // end rod
		mat == 10501 ||
		mat == 10502 ||
		mat == 10508 || // chorus flower
		mat == 10516 || // lit furnace
		mat == 10528 || // soul torch
		mat == 10529 ||
		mat == 10544 || // glow lichen
		mat == 10548 || // enchanting table
		mat == 10556 || // end portal frame with eye
		mat == 10560 || // lantern
		mat == 10564 || // soul lantern
		mat == 10572 || // dragon egg
		mat == 10576 || // lit smoker
		mat == 10580 || // lit blast furnace
		mat == 10584 || // lit candles
		mat == 10592 || // respawn anchor
		mat == 10596 || // redstone wire
		mat == 10604 || // lit redstone torch
		mat == 10632 || // glow berries
		mat == 10640 || // lit redstone lamp
		mat == 10648 || // shroomlight
		mat == 10652 || // lit campfire
		mat == 10656 || // lit soul campfire
		mat == 10680 || // ochre       froglight
		mat == 10684 || // verdant     froglight
		mat == 10688 || // pearlescent froglight
		mat == 10705 || // active sculk sensor
		mat == 10708 || // spawner
		mat == 10999 || // light block
		mat == 12740 || // lit candle cake
		mat == 30020 || // nether portal
		mat == 31016 || // beacon
		mat == 60000 || // end portal
		mat == 60012 || // ender chest
		mat == 60020 || // conduit
		mat == 50000 || // end crystal
		mat == 50004 || // lightning bolt
		mat == 50012 || // glow item frame
		mat == 50020 || // blaze
		mat == 50048 || // glow squid
		mat == 50052 || // magma cube
		mat == 50080 || // allay
		mat == 50116    // TNT and TNT minecart
	);
}
bool badPixel(vec4 color, vec4 glColor, int mat) {
	switch(mat) {
		case 4431:
		case 4432:
			if (color.g > max(color.r, color.b) + 0.05 && length(glColor.rgb - vec3(1)) < 0.1) {
				return true;
			}
			break;
	}
	return false;
}