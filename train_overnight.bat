@echo off
cd C:\Users\rudha\OneDrive\Desktop\ucl_pipeline
call conda activate ucl

echo Run 1: 320x512 200 epochs
python scripts\train_seg.py --out models\v4_320x512_e200 --resize 320 512 --epochs 200
echo Run 1 done.

echo Run 2: 256x384 200 epochs
python scripts\train_seg.py --out models\v5_256x384_e200 --resize 256 384 --epochs 200
echo Run 2 done.

echo Run 3: 320x512 400 epochs
python scripts\train_seg.py --out models\v6_320x512_e400 --resize 320 512 --epochs 400
echo Run 3 done.

echo All runs complete.
pause
