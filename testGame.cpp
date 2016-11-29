#include "gameControl.h"
#include <unistd.h>

int main(){
	const int w = 80;
	const int h = 80;
	const int boardW = 15;
	const int boardH = 4;
	const int renderScale = 4;
	
	initGame(w, h, boardW, boardH);
	initRenderer(w * renderScale, h * renderScale, 32, renderScale);
	
	int hRenderBuffer = createIntBuffer(w * h);
	
	while(handleEvents()){
		
		int lOrR = getLeftOrRight();
		moveBoard(lOrR);
		getNowImage(hRenderBuffer);
		render(hRenderBuffer, w, h);
		updateGame(4);
		usleep(13000);
	}

	return 0;
}
