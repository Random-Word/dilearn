#!/bin/bash

for i in `seq 1 9`; do
    mkdir data/train_photos/$i
    mv data/train_photos/$i*.jpg data/train_photos/$i/
done
