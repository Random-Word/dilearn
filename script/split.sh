#!/bin/bash

for i in `seq 1 9`; do
    if [ ! -d "data/train_photos/$1" ]; then
        mkdir data/train_photos/$i
        mv data/train_photos/$i*.jpg data/train_photos/$i/
    fi
done
