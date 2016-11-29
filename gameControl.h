#include <SDL/SDL.h>
#include <SDL/SDL_draw.h>

#include <cstdlib>

class game{
public:
	game(int i_w, int i_h, int boardW = 0, int boardH = 0);
	~game();

public:
	int update(int number);
	void getMap(unsigned char *dst);
	void reset();
	void renderToArray(int *dst, int colorObstacal = 0x00ff00, int colorBoard = 0xff0000);
	int getScore();
	int moveBoard(int direction);
	
private:
	bool checkCollision();

private:

	unsigned char* map = nullptr;
	int screenW = 0, screenH = 0;
	int boardX = 0, boardY = 0;
	int boardW = 0, boardH = 0;
	int score = 0;
};


class renderer{
public:
	renderer(int w, int h, int bpp = 32, int scale = 1);
	~renderer();
	bool handleEvents();
	void render(int *map, size_t w, size_t h);
	void getMouse(Uint16 &x, Uint16 &y);
	Uint8 *getKey();
private:
	SDL_Surface *screen;
	int screenW = 0;
	int screenH = 0;
	int scale = 1;
	int bpp;
	Uint8 *keys;
	SDL_Event event;
	Uint16 mouseX = 0;
	Uint16 mouseY = 0;
	Uint8 *keycodes = nullptr;
};

class memMgr{
public:
	memMgr();
	~memMgr();
	int allocateMem(int size);
	void releaseMem(int buffer);
	int getValue(int buffer, int pos);
	void setValue(int buffer, int pos, int value);
	int* getBuffer(int buffer);
	int registBuffer(int *buffer);
private:
	int *memlist[100];
	int freePosList[100];
	int ptrFreePosList = 99;
};

extern "C"{

void initGame(int w, int h, int boardW, int boardH);

void getNowImage(int buffer);

int rgb(int r, int g, int b);

void initRenderer(int w, int h, int bpp, int scale);

bool handleEvents();

void render(int map, int w, int h);

int getLeftOrRight();

int createIntBuffer(int size);

void destroyIntBuffer(int buffer);

int getValue(int buffer, int i);

void setValue(int buffer, int i, int v);

int *getBuffer(int buffer);

int updateGame(int number);

int moveBoard(int key);
};
