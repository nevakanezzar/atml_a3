python b34_3.py 0 0.01 0 2>&1 | tee results.a0.0.01.0
python b34_3.py 1 0.01 0 2>&1 | tee results.b0.0.01.1
python b34_3.py 2 0.01 0 2>&1 | tee results.c0.0.01.2

python b34_3.py 0 0.001 0 2>&1 | tee results.d0.0.001.0
python b34_3.py 1 0.001 0 2>&1 | tee results.e0.0.001.1
python b34_3.py 2 0.001 0 2>&1 | tee results.f0.0.001.2

python b34_3.py 0 0.0001 0 2>&1 | tee results.g0.0.0001.0
python b34_3.py 1 0.0001 0 2>&1 | tee results.h0.0.0001.1
python b34_3.py 2 0.0001 0 2>&1 | tee results.i0.0.0001.2

