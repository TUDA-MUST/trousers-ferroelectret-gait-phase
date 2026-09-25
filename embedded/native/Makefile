# Native build of the estimator self-tests (Linux / Raspberry Pi / any C99 compiler).
#
#   make                                  # host defaults
#   make CFLAGS="-O2 -mcpu=cortex-a53"    # 64-bit Raspberry Pi OS on a Pi Zero 2 W
#   make CFLAGS="-O2 -mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard"   # 32-bit Pi OS
#   make F64=1                            # double-precision variant
#   ./bench_native
#
# Copy the whole esp32_ao folder to the target (the generated headers in include/ are
# needed), then run make inside native/.

CC      ?= gcc
CFLAGS  ?= -O2
LIB      = ../lib/gaitphase
INC      = -I$(LIB) -I../include
SRCS     = bench_native.c $(LIB)/gp_bandpass.c $(LIB)/gp_ao.c $(LIB)/gp_events.c \
           $(LIB)/gp_correction.c $(LIB)/gp_estimator.c $(LIB)/gp_cao.c $(LIB)/gp_tcn.c
DEFS     =
ifeq ($(F64),1)
DEFS    += -DGP_USE_DOUBLE
endif

bench_native: $(SRCS)
	$(CC) $(CFLAGS) -std=c99 -Wall $(DEFS) $(INC) -o $@ $(SRCS) -lm

clean:
	rm -f bench_native

.PHONY: clean
