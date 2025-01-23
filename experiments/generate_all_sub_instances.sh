# This scripts generate all the sub instances based on the benchmarks folder

SEEDS="1 2 3 4 5 6 7 8 9 10"
FRACTIONS=".25 .5 .75"

SRC="./benchmarks"
DST="./subinstances"
mkdir -p $DST

for file in $(ls $SRC/*.csv); do
    filename="${file##*/}"
    echo $filename
    for seed in $SEEDS; do
        for fraction in $FRACTIONS; do
            dst_file=$DST/${file}.$seed.$fraction
            if [ ! -f $dst_file ]; then
                python ./scripts/gen_sub_instance.py -s $seed -o  $file  $fraction || rm $dst_file >/dev/null
            fi
        done
    done
done
