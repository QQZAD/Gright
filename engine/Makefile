cc := g++
prom := dpdk
inc := /usr/local/include
src := $(shell find ./ -name "*.cpp")
src += $(shell find ../config -name "*.cpp")
src += $(shell find ../gpu_core/packet -name "*.cpp")
tar := -march=native -mno-avx512f
obj := $(src:%.cpp=%.o)
libs := -lrte_eal -lrte_ethdev -lrte_mbuf -lrte_mempool
$(prom): $(obj)
	$(cc) -g $(tar) -o $(prom) $(obj) $(libs)
%.o: %.cpp
	$(cc) -c $(tar) $< -o $@ -I $(inc)
clean:
	rm -rf $(prom) $(obj)
