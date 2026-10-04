CC = gcc
CFLAGS = -O3 -Wall -Wextra

all: engine

engine: main.c
	$(CC) $(CFLAGS) -o bio_engine main.c

clean:
	rm -f bio_engine
