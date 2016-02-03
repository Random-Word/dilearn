all: data/X.npy data/y.npy

data/X.npy data/y.npy: data/train_photos/1/1.jpg data/train_photos/9/9.jpg
	./script/proc_images.py

data/train_photos/1/1.jpg data/train_photos/9/9.jpg: data/train_photos
	./script/split.sh

data/train_photos: data/train_photos.tgz
	tar -xzf data/train_photos.tgz -C data/

data/train_photos.tgz:
	aws s3 sync s3://capstone-data data

clean:
	rm data/*
