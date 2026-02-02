#!/usr/bin/env python3
import sys

i = 0
while True:
    line = sys.stdin.readline()
    if not line:
        break
    i += 1
    sys.stdout.write(f"{i:04d} {line}")
    sys.stdout.flush()
