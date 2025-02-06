#!/bin/bash

cd ..
git clone https://github.com/ASSERT-KTH/repairbench.git
cd rtsm
path="../repairbench"
python converter/repair_converter.py $path
python converter/repair_converter.py $path 2025-01-01
python converter/repair_converter.py $path 2024-10-01
mv *.csv benchmarks

yes | rm -r $path