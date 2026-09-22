SHELL := /bin/bash
# Build / run the torchvision classification cases, one directory per model.
# Each <model>/ holds <model>_tv.s (hwacha-cc assembly), tv_main.c (generic host), hwlib.s (library
# kernels), <model>_tv_check.bin (input + PyTorch reference, embedded via .incbin) and the weights:
# <model>_tv_weights.bin referenced by .incbin from <model>_tv_weights.bin.S.
#
#   make                 build every case            -> <model>/<model>_tv.riscv
#   make <model>         build one case
#   make run             build + run all on Spike, one PASS/FAIL line each
#   make <model>.spike   run one case
#   make clean           remove the .riscv files
#   make gen-<model>     regenerate one case from PyTorch (export_tv.py / gen.sh, see models.txt)
#   make gen-all         generate every model in models.txt that has no assembly yet (gen_all.sh)
ROOT    ?= /home/tesla/hwacha-compiler
COMMON  := $(ROOT)/esp-tests/benchmarks/common
CC      := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc
# simulator memory in MB: the host's arena is everything above the loaded image, so a case gets its
# weights size plus 1.5 GB of headroom, rounded up to a GB (min 2 GB); MEM=<MB> overrides for all cases
mem_for = $(if $(MEM),$(MEM),$(shell s=$$(stat -c %s $(1)/$(1)_tv_weights.bin 2>/dev/null || echo 0); echo $$(( ((s/1048576 + 1536 + 1023) / 1024) * 1024 )) ))
SPIKE   := $(ROOT)/install/bin/spike --isa=rv64gc --extension=hwacha
CFLAGS  := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -fno-math-errno -march=rv64gcxhwacha -mabi=lp64d \
           -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(CURDIR)/tv.ld
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

# every sub-directory that contains <name>/<name>_tv.s is a case
CASES := $(sort $(foreach s,$(wildcard */*_tv.s),$(if $(filter $(notdir $(s)),$(patsubst %/,%,$(dir $(s)))_tv.s),$(patsubst %/,%,$(dir $(s))))))
RISCV := $(foreach c,$(CASES),$(c)/$(c)_tv.riscv)

.PHONY: all run clean list gen-all $(CASES) $(addsuffix .spike,$(CASES))
all: $(RISCV)
list:
	@printf '%s\n' $(CASES)

# the runtime is built as named objects: tv.ld keeps their code (and the host's, which holds the trap
# handler the crt jumps to) next to .text.init, in front of the weights -- the crt's JALs reach 1 MB
crt.o: $(COMMON)/crt.S
	$(CC) $(CFLAGS) -c -o $@ $<
syscalls.o: $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -c -o $@ $<

define CASE_RULE
$(1): $(1)/$(1)_tv.riscv
$(1)/$(1)_tv.riscv: $(1)/tv_main.c $(1)/$(1)_tv.s $(1)/hwlib.s $(1)/$(1)_tv_weights.bin.S $(1)/$(1)_tv_weights.bin $(1)/$(1)_tv_check.bin crt.o syscalls.o tv.ld
	cd $(1) && $(CC) $(CFLAGS) -DCHECK_FILE='"$(1)_tv_check.bin"' -DMODEL='"$(1)"' -DMEM_MB=$(call mem_for,$(1)) -c -o tv_main.o tv_main.c && \
	  $(CC) $(CFLAGS) -o $(1)_tv.riscv $(CURDIR)/crt.o $(CURDIR)/syscalls.o tv_main.o $(1)_tv.s hwlib.s $(1)_tv_weights.bin.S $(LDFLAGS)
$(1).spike: $(1)/$(1)_tv.riscv
	$(SPIKE) -m$(call mem_for,$(1)) $(1)/$(1)_tv.riscv
endef
$(foreach c,$(CASES),$(eval $(call CASE_RULE,$(c))))

# build and run every case, one PASS/FAIL line each (tv_main.c prints "tv PASS" / "tv FAIL")
run:
	@pass=0; fail=0; for c in $(CASES); do \
	  if $(MAKE) -s $$c/$${c}_tv.riscv >/tmp/tv_$$c.log 2>&1; then \
	    r=$$($(MAKE) -s $$c.spike 2>&1 | grep -oE "tv (PASS|FAIL)" | tail -1); \
	    echo "$$c $${r#tv }" | sed "s/ $$/ NO-OUTPUT/"; case "$$r" in *PASS) pass=$$((pass+1));; *) fail=$$((fail+1));; esac; \
	  else echo "$$c BUILD-FAIL (see /tmp/tv_$$c.log)"; fail=$$((fail+1)); fi; \
	done; echo "---- $$pass passed, $$fail failed of $$(echo $(CASES) | wc -w)"
clean:
	rm -f $(RISCV) $(RISCV:_tv.riscv=)/tv_main.o crt.o syscalls.o

# regeneration from PyTorch (needs the hwacha-cc build tree, mlir-opt and the torch-mlir venv)
gen-%:
	@read -r n hw kw <<<"$$(grep -E '^$* ' models.txt)"; [ -n "$$n" ] || { echo "$* not in models.txt"; exit 1; }; \
	./gen.sh $$n $$hw "$$kw"
gen-all:
	./gen_all.sh
