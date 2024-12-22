dll_windows:
	cc -shared -O3 -o w2vloader.dll source/w2vloader.c

so_linux:
	gcc -shared -O3 -fPIC -o w2vloader.so source/w2vloader.c

# List of file extensions to clean
EXTENSIONS = .dll .so .o .lib .pdb

# Clean target
clean:
	@for ext in $(EXTENSIONS); do \
		rm -f w2vloader$$ext; \
	done

.PHONY: clean
