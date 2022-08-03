all: 

lowmc_x86_128: lowmc_x86_128.c bench.c
	gcc -o lowmc_x86_128.x lowmc_x86_128.c bench.c -march=native -g -O3 -Wall -Wextra -fno-stack-protector
	./lowmc_x86_128.x

lowmc_x86_192: lowmc_x86_192.c bench.c
	gcc -o lowmc_x86_192.x lowmc_x86_192.c bench.c -march=native -g -O3 -Wall -Wextra -fno-stack-protector
	./lowmc_x86_192.x

lowmc_8way_128: lowmc_8way_128.c constants_8way_128.c bench.c
	gcc -o lowmc_8way_128.x lowmc_8way_128.c constants_8way_128.c bench.c -march=native -g -O3 -Wall -Wextra -fno-stack-protector
	./lowmc_8way_128.x

lowmc_8way_128_base: lowmc_8way_128_base.c constants_8way_128.c bench.c
	gcc -o lowmc_8way_128_base.x lowmc_8way_128_base.c constants_8way_128.c bench.c -march=native -g -O3 -Wall -Wextra -fno-stack-protector
	./lowmc_8way_128_base.x

lowmc_8way_192: lowmc_8way_192.c constants_8way_192.c bench.c
	gcc -o lowmc_8way_192.x lowmc_8way_192.c constants_8way_192.c bench.c -march=native -g -O3 -Wall -Wextra -fno-stack-protector
	./lowmc_8way_192.x

lowmc_8way_192_base: lowmc_8way_192_base.c constants_8way_192.c bench.c
	gcc -o lowmc_8way_192_base.x lowmc_8way_192_base.c constants_8way_192.c bench.c -march=native -g -O3 -Wall -Wextra -fno-stack-protector
	./lowmc_8way_192_base.x

lowmc_16way_129: lowmc_16way_129.c constants_16way_129.c bench.c
	gcc -o lowmc_16way_129.x lowmc_16way_129.c constants_16way_129.c bench.c -march=native -g -O3 -Wall -Wextra -fno-stack-protector
	./lowmc_16way_129.x

lowmc_16way_192: lowmc_16way_192.c constants_16way_192.c bench.c
	gcc -o lowmc_16way_192.x lowmc_16way_192.c constants_16way_192.c bench.c -march=native -g -O3 -Wall -Wextra -fno-stack-protector
	./lowmc_16way_192.x

clean: 
	rm -f *.x *.o
