# Build / run the torchtune module (api_ref_modules) cases, one directory per function.
# Each <case>/ holds <case>.s (hwacha-cc assembly), mod_main.c (generic host) and <case>_check.bin
# (input + PyTorch reference, embedded via .incbin); a few also carry hwlib.s (library kernels).
#
#   make                 build every case            -> <case>/<case>.riscv
#   make <case>          build one case
#   make run             build + run all on Spike, one PASS/FAIL line each
#   make <case>.spike    run one case
#   make clean
#   make gen-<case>      regenerate one case from PyTorch (export_ttf.py via gen.sh)
#   make gen-all         generate every function export_ttf.py knows that has no assembly yet
ROOT    ?= /home/tesla/hwacha-compiler
COMMON  := $(ROOT)/esp-tests/benchmarks/common
CC      := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc
SPIKE   := $(ROOT)/install/bin/spike --isa=rv64gc --extension=hwacha
CFLAGS  := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -fno-math-errno -march=rv64gcxhwacha -mabi=lp64d \
           -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld

# every sub-directory that contains <name>/<name>.s is a case
CASES := $(sort $(foreach s,$(wildcard */*.s),$(if $(filter $(notdir $(s)),$(patsubst %/,%,$(dir $(s))).s),$(patsubst %/,%,$(dir $(s))))))
RISCV := $(foreach c,$(CASES),$(c)/$(c).riscv)

.PHONY: all run clean list gen-all $(CASES) $(addsuffix .spike,$(CASES))
all: $(RISCV)
list:
	@printf '%s\n' $(CASES)

# per-case rule; hwlib.s is linked in only when the case directory has one
define CASE_RULE
$(1): $(1)/$(1).riscv
$(1)/$(1).riscv: $(1)/mod_main.c $(1)/$(1).s $(1)/$(1)_check.bin $(wildcard $(1)/hwlib.s) $(COMMON)/crt.S $(COMMON)/syscalls.c
	cd $(1) && $(CC) $(CFLAGS) -DCHECK_FILE='"$(1)_check.bin"' -DMODEL='"$(1)"' -o $(1).riscv \
	  mod_main.c $(1).s $(notdir $(wildcard $(1)/hwlib.s)) $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)
$(1).spike: $(1)/$(1).riscv
	$(SPIKE) $(1)/$(1).riscv
endef
$(foreach c,$(CASES),$(eval $(call CASE_RULE,$(c))))

# build and run every case, one PASS/FAIL line each
run:
	@pass=0; fail=0; for c in $(CASES); do \
	  if $(MAKE) -s $$c/$$c.riscv >/tmp/tm_$$c.log 2>&1; then \
	    r=$$($(SPIKE) $$c/$$c.riscv 2>&1 | grep -oE "$$c (PASS|FAIL)" | tail -1); \
	    echo "$${r:-$$c NO-OUTPUT}"; case "$$r" in *PASS) pass=$$((pass+1));; *) fail=$$((fail+1));; esac; \
	  else echo "$$c BUILD-FAIL (see /tmp/tm_$$c.log)"; fail=$$((fail+1)); fi; \
	done; echo "---- $$pass passed, $$fail failed of $$(echo $(CASES) | wc -w)"
clean:
	rm -f $(RISCV)

# regeneration from PyTorch (needs the hwacha-cc build tree, mlir-opt and the torch-mlir venv ../.tmenv)
gen-%:
	./gen.sh $*
gen-all:
	@for n in $$(/home/tesla/hwacha-application/.tmenv/bin/python export_ttf.py --list 2>/dev/null); do \
	  [ -f $$n/$$n.s ] || ./gen.sh $$n; done
