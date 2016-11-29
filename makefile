
game: gameControl.cpp gameControl.h
	g++ gameControl.cpp -o game.so -O2 -shared -lSDL -fPIC -std=c++11
	g++ ./testGame.cpp gameControl.cpp -o game -lSDL -lSDL_gfx -O2 -std=c++11

clean:
	rm game.so
	rm game
