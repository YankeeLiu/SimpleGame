#include "gameControl.h"

game::game(int i_w, int i_h, int i_boardW, int i_boardH) : screenW(i_w), screenH(i_h){

	if(i_boardW <= 0 || i_boardH <= 0 || i_boardW > i_w / 2 || i_boardH > i_h / 5){
		boardW = i_w / 5;
		boardH = i_h / 20;
	}else{
		boardW = i_boardW;
		boardH = i_boardH;
	}
	
	map = new unsigned char[i_w * i_h];
	
	memset(map, 0, sizeof(unsigned char) * screenW * screenH);

	reset();
}


game::~game(){
	delete []map;
}


int game::update(int number){

	int blockSz = 5;

	for(int x = 0; x < screenW - blockSz - 1; ++x){
		if (rand() % 5000 < number){
			for(int offsetY = 0; offsetY < blockSz; ++offsetY){
				for(int offsetX = 0; offsetX < blockSz; ++offsetX){
					map[x + offsetX + offsetY * screenW] = 1;
				}
			}
		}
		map[x + (screenH - 1) * screenW] = 0;
	}


	for(int y = screenH - 2; y >= 0; --y){
		for(int x = screenW - 1; x >= 0; --x){
			if(map[x + y * screenW]){
				map[x + (y + 1) * screenW] = map[x + y * screenW];
				map[x + y * screenW] = 0;
			}
		}
	}
	
	if(checkCollision()){
		reset();
		score = 1;
		return score;
	}
	
	return -1;
}


void game::moveBoard(int direction){
	int step = direction > 0 ? 1 : -1;
	
	if (boardX + step >= 0 && boardX + boardW + step < screenW){
		boardX += step;
	}
}


void game::getMap(unsigned char* dst){
	memcpy(dst, map, sizeof(unsigned char) * screenW * screenH);
}


void game::reset(){
	boardX = (screenW - boardW) / 2;
	boardY = (screenH - boardH) / 8 * 7;
	score = 0;
	memset(map + screenH / 2 * screenW, 0, sizeof(unsigned char) * screenW * screenH / 2);
}


void game::renderToArray(int *dst, int colorObstacal, int colorBoard){
	memset(dst, 0, sizeof(int) * screenW * screenH);
	for(int y = 0; y < screenH; ++y){
		for(int x = 0; x < screenW; ++x){
			if(map[x + y * screenW]){
				dst[x + y * screenW] = colorObstacal;
			}					
		}
	}

	for(int y = boardY; y < boardY + boardH; ++y){
		for(int x = boardX; x < boardX + boardW; ++x){
			dst[x + y * screenW] = colorBoard;
		}
	}
}


bool game::checkCollision(){
	for(int y = boardY; y < boardY + boardH; ++y){
		for(int x = boardX; x < boardX + boardW; ++x){
			if(map[x + y * screenW]){
				return true;
			}
		}
	}
	
	return false;
}


int game::getScore(){
	return score;
}

//==================================================

renderer::renderer(int w, int h, int bpp, int scale) :screenW(w), screenH(h), bpp(bpp), scale(scale) {
	SDL_Init(SDL_INIT_EVERYTHING);
	screen = SDL_SetVideoMode(w, h, 32, SDL_ASYNCBLIT | SDL_HWSURFACE);
	if (!screen){
		throw("error while init sdl");
	}
}


renderer::~renderer(){
	SDL_FreeSurface(screen);
	screen = nullptr;
}


bool renderer::handleEvents(){
	while (SDL_PollEvent(&event)) {
		keys = SDL_GetKeyState(NULL);
		switch (event.type) {
		case SDL_MOUSEMOTION:
			mouseX = event.button.x;
			mouseY = event.button.y;
			break;
		case SDL_KEYDOWN:
			keycodes = SDL_GetKeyState(NULL);
			break;
		case SDL_KEYUP:
			keycodes = nullptr;
			break;
		case SDL_QUIT:
			return false;
			break;
		}
	}
	return true;
}


void renderer::render(int *map, size_t w, size_t h){
	Draw_FillRect(screen, 0, 0, screen->w, screen->h, 0xffffff);
	for (size_t y = 0; y < h; ++y){
		for (size_t x = 0; x < w; ++x){
			Draw_FillRect(screen, x * scale, y * scale, scale, scale, map[x + y * w] < 0 ? 0 : map[x + y * w]);
		}
	}
	SDL_Flip(screen);
}


void renderer::getMouse(Uint16 &x, Uint16 &y){
	x = mouseX / scale;
	y = mouseY / scale;
}


Uint8 *renderer::getKey(){
	return keycodes;
}

//==================================================

memMgr::memMgr(){
	for(int i = 0; i < 100; ++i){
		freePosList[i] = 99 - i;
		memlist[i] = nullptr;
	}
}

memMgr::~memMgr(){
	for(int i = ptrFreePosList + 1; i < 100; ++i){
		delete [](memlist[freePosList[i]]);
	}
}

int memMgr::allocateMem(int size){
	if(ptrFreePosList == 0) return 0;
	int *ptr = new int[size];
	memlist[freePosList[ptrFreePosList]] = ptr;
	return freePosList[ptrFreePosList--];
}

void memMgr::releaseMem(int buffer){
	delete []memlist[buffer];
	freePosList[++ptrFreePosList] = buffer;
}

int memMgr::getValue(int buffer, int pos){
	return memlist[buffer][pos];
}

void memMgr::setValue(int buffer, int pos, int value){
	memlist[buffer][pos] = value;
}

int* memMgr::getBuffer(int buffer){
	return memlist[buffer];
}

int memMgr::registBuffer(int *buffer){
	if(ptrFreePosList == 0) return 0;
	memlist[freePosList[ptrFreePosList]] = buffer;
	return freePosList[ptrFreePosList--];
}

//=======================================

memMgr memMan;
renderer *mapRenderer = nullptr;
game *myGame = nullptr;

void initGame(int w, int h, int boardW, int boardH){
	if(myGame){
		delete myGame;
	} 

	myGame = new game(w, h, boardW, boardH);
}

void getNowImage(int buffer){
	if(myGame){
		myGame->renderToArray(memMan.getBuffer(buffer));
	}
}


int rgb(int r, int g, int b){
		return ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
}

void initRenderer(int w, int h, int bpp, int scale){
		if(mapRenderer) delete mapRenderer;
		mapRenderer = new renderer(w, h, bpp, scale);
}

bool handleEvents(){
	if(mapRenderer) 
		return mapRenderer->handleEvents();
	return false;
}


void render(int map, int w, int h){
	if(mapRenderer)
		mapRenderer->render(memMan.getBuffer(map), w, h);
}

int getLeftOrRight(){
	Uint8 *keys = mapRenderer->getKey();
	if (!keys) return 0;
	if ((keys[SDLK_LEFT] && keys[SDLK_RIGHT]) || (!keys[SDLK_LEFT] && !keys[SDLK_RIGHT])){
		return 0;
	} 
	else if(keys[SDLK_LEFT]){
		return 1;
	}
	else{
		return 2;
	}
}

int createIntBuffer(int size){
	return memMan.allocateMem(size);
}

void destroyIntBuffer(int buffer){
	memMan.releaseMem(buffer);
}

int getValue(int buffer, int i){
	return memMan.getValue(buffer, i);
}

void setValue(int buffer, int i, int v){
	memMan.setValue(buffer, i, v);
}

int *getBuffer(int buffer){
	return memMan.getBuffer(buffer);
}


int updateGame(int number){
	if(myGame){
		return myGame->update(number);
	}
}

void moveBoard(int key){
	if(myGame && key){
		myGame->moveBoard(key == 1 ? -1 : 1);
	}
}


