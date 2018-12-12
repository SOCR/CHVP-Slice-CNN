all: registration pure_scaling split train

registration:
	echo "registration start"
	python src/registration.py
	python src/re_label.py -img_dir="../Data/Registration"

pure_scaling:
	echo "pure scaling start"
	python src/pure_scaling.py
	python src/re_label.py -img_dir="../Data/Pure_Scaling"


split:
	python ./src/train_test_split.py -ratio=0.7 -prefix="reg" -data_source="../Data/Registration" -corpus_path="./corpus"
	python ./src/train_test_split.py -ratio=0.7 -prefix="pure" -data_source="../Data/Pure_Scaling" -corpus_path="./corpus"

train:
	chmod +777 ./train.sh
	# the args: 
	# num_epoch, drop_out, slices, suffix, train suffix of models
	./train.sh 70 0 30 reg_30_00_0 reg
	echo "train complete"

clean:
	- rm -r ./2D_CNN/models
	mkdir ./2D_CNN/models
