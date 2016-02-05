all: data/weights.h5 data/X.npy data/y.npy

data/weights.h5: data/X.npy data/y.npy data/vgg16_weights.hd5
	./script/vgg16.py

data/X.npy data/y.npy: data/train_photos/1 data/train_photos/9
	./script/proc_images.py

data/train_photos/1 data/train_photos/9: data/train_photos
	./script/split.sh

data/train_photos: data/train_photos.tgz
	tar -xzf data/train_photos.tgz -C data

data/train_photos.tgz:
	aws s3 sync s3://capstone-data data

data/vgg16_weights.hd5:
	gdown https://docs.google.com/uc?id=0Bz7KyqmuGsilT0J5dmRCM0ROVHc&export=download data/

clean:
	rm data/*
