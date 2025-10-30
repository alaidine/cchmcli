SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.o)
CFLAGS=-Wall -Wextra
LDFLAGS=-ldatachannel
EXE=cchmcli

all: $(OBJ)
	$(CC) -o $(EXE) $(OBJ) $(LDFLAGS)

clean:
	rm -f $(OBJ)
	rm $(EXE)
