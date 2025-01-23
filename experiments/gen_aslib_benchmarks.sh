#!/bin/bash

cd ..
git clone https://github.com/coseal/aslib_data.git
cd rtsm
path="../aslib_data"
for file in $(ls $path); do
    echo $file
    python converter/aslib_scenario_converter.py $path/$file
    mv ${file}_cost.csv ./benchmarks
done

yes | rm -r $path