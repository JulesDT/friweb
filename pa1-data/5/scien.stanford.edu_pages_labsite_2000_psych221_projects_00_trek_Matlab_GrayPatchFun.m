graypatchfun m generate gray patch image with monitor gamma inverse to test camera gamma hui li 3 7 2000 function gp outfile graypatchfun gamma c1 zeros 192256 2 c2 zeros 192256 32 c3 zeros 192256 64 c4 zeros 192256 96 c5 zeros 192256 128 c6 zeros 192256 160 c7 zeros 192256 192 c8 zeros 192256 224 c9 zeros 192256 255 im c1 c2 c3 c4 c5 c6 c7 c8 c9 im double im 255 figure imshow im monitor gamma correction inverse gp im 1 gamma figure imshow gp imwrite gp graypatch jpg jpg