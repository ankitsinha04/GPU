NVCC        = nvcc
LD_FLAGS    = -lcudart -L/usr/local/cuda-8.0/lib64
EXE	    = mnist
OBJ	    = main.o support.o

default: $(EXE)

main.o: main.cu kernel.cu support.h
	$(NVCC) -c -o $@ main.cu 

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu 

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

