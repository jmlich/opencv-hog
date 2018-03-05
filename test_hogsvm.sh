#!/bin/bash

#DATASET_SIZE=125
#DATASET_SIZE=500
#DATASET_SIZE=1000
DATASET_SIZE=5000
#DATASET_SIZE=25446
#DATASET_SIZE=125240
#DATASET_SIZE=150686

dataset_size_cache="./$0.last_DATASET_SIZE"
fn="anot.txt"

if [ -f "$dataset_size_cache" ]; then
    DATASET_SIZE_PREV=$(cat "$dataset_size_cache")
else
    DATASET_SIZE_PREV=0
fi
echo $DATASET_SIZE > "$dataset_size_cache"

if [ "$DATASET_SIZE_PREV" -eq "$DATASET_SIZE" ]; then
    PREPARE_DATASET=0
else
    PREPARE_DATASET=1
fi



if [ "$PREPARE_DATASET" = 1 ]; then

    if [ -f "./anot.$DATASET_SIZE.txt" ]; then
        echo "cp ./anot.$DATASET_SIZE.txt ./anot.txt"
        cp "./anot.$DATASET_SIZE.txt" "./anot.txt"
    else

        positive_dirs="/var/www/html/fire/labelme_samples-80/"
#        positive_dirs="/var/www/html/fire/labelme_samples-center/"


        negative_dirs="/var/www/html/fire/sun-dataset/SUNOUT"

        IFS=$'\n'

        pfn="positive_anot.txt"
        nfn="negative_anot.txt"

        rm -f "$pfn" "$nfn"

        for i in $negative_dirs; do
            echo $i +
            find "$i" -type f -name '*.jpg' | sed 's/$/,non-fire/'|sort -R >> $nfn
        done

        for i in $positive_dirs; do
            echo $i +
            find "$i" -type f -name '*.jpg'| sed 's/$/,fire/' |sort -R >> "$pfn"
        done

        # print size of set
        wc -l "$pfn"
        wc -l "$nfn"

        # merge anot files together
        paste -d "\n" "$pfn" "$nfn" | sed '/^\s*$/d'  > "anot.txt"

        # trim to DATASET_SIZE firts samples
        tmp=$(mktemp "/tmp/$0.XXXXXXX");  head "$fn" -n "$DATASET_SIZE" >"$tmp"; mv "$tmp" "anot.txt"
        rm -f "$pfn" "$nfn"

        cp "$fn" ./anot.$DATASET_SIZE.txt
    fi

fi # if [ "$PREPARE_DATASET" = 1 ];

wc -l "$fn"

t1=$(date +%s)
echo "Started at $(date --iso-8601=s)"

./hogsvm.py ./anot.txt

t2=$(date +%s)
echo "Ended at $(date --iso-8601=s)"

echo "Elapsed $((t2 - t1)) seconds"
