@echo off
if exist .\eseguibile (
cd .\eseguibile
del "*"
cd ..
) else (
mkdir eseguibile
)
nvcc .\generator.cpp -o .\eseguibile\generator.exe
nvcc -rdc=true -D FILE_OUT -D DEBUG  -D MAX_THREADS=1024 -D MAX_BLOCKS_A=1 -D MAX_BLOCKS_AI=1 -D NTHR=1 -D MAX_BLOCKS_B=16 -D TIME -std=c++17 -lineinfo -Xcompiler -openmp -Xlinker /HEAP:0x8096 .\progetto_G.cu -D HIDE -o .\eseguibile\progetto_1_1_16.exe
for /l %%a in (2, 1, 6) do ( 
for /l %%b in (1, 1, 5) do ( 
nvcc -rdc=true -D FILE_OUT -D DEBUG  -D MAX_THREADS=1024 -D MAX_BLOCKS_A=%%a -D MAX_BLOCKS_AI=%%b -D MAX_BLOCKS_B=16 -D TIME -std=c++17 -lineinfo -Xcompiler -openmp -Xlinker /HEAP:0x8096 .\progetto_G.cu -D HIDE -o .\eseguibile\progetto_%%a_%%b_16.exe
nvcc -rdc=true -D FILE_OUT -D DEBUG  -D MAX_THREADS=1024 -D MAX_BLOCKS_A=%%a -D MAX_BLOCKS_AI=%%b -D MAX_BLOCKS_B=16 -D TIME -std=c++17 -lineinfo -Xcompiler -openmp -Xlinker /HEAP:0x8096 .\progetto_G.cu -D HIDE -D NO_INPUT -D NTAB -o .\eseguibile\progetto_%%a_%%b_16_H.exe
nvcc -rdc=true -D FILE_OUT -D DEBUG  -D MAX_THREADS=1024 -D MAX_BLOCKS_A=%%a -D MAX_BLOCKS_AI=%%b -D MAX_BLOCKS_B=16 -D TIME -std=c++17 -lineinfo -Xcompiler -openmp -Xlinker /HEAP:0x8096 .\progetto_G.cu -D NO_INPUT -D NTAB -o .\eseguibile\progetto_%%a_%%b_16_H1.exe
)
)
cd .\eseguibile
del "*.exp"
del "*.lib"
cd ..