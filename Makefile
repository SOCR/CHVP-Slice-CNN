all: registration pure_scaling split train

registration:
	echo "registration start"
	python3 src/registration.py
	python3 src/re_label.py -img_dir="../Data/Registration"

pure_scaling:
	echo "pure scaling start"
	python3 src/pure_scaling.py
	python3 src/re_label.py -img_dir="../Data/Pure_Scaling"


split:
	python3 ./2D_CNN/src/train_test_split.py -ratio=0.7 -prefix="reg" -data_source="../Data/Registration" -corpus_path="./2D_CNN/corpus"
	python3 ./2D_CNN/src/train_test_split.py -ratio=0.7 -prefix="pure" -data_source="../Data/Pure_Scaling" -corpus_path="./2D_CNN/corpus"

train:
	chmod +777 ./2D_CNN/train.sh
	# the args: 
	# num_epoch, drop_out, slices, suffix, train suffix of models
	./train.sh 70 0 30 reg_30_00_0 reg
	echo "train complete"

clean:
	- rm -r ./2D_CNN/models
	mkdir ./2D_CNN/models