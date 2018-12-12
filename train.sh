num_epoch=$1
drop_out=$2
num_sessions_each_dim=$3
suffix=$4
train_suffix=$5
generated_model="my_model_${suffix}"
save_model="save_model_${suffix}"
train_file_dir="train_data_${train_suffix}"
test_file_dir="test_data_${train_suffix}"

cd 2D_CNN
python3 src/model_generator_CNN.py -model_name=$generated_model -num_sessions_each_dim=$num_sessions_each_dim -dropout=$drop_out
python3 src/train_CNN.py -load=$generated_model -save=$save_model -num_epoch=$num_epoch -train_file_dir=$train_file_dir -test_file_dir=$test_file_dir -num_sessions_each_dim=$num_sessions_each_dim
cd ..

# python3 src/model_generator_CNN.py -model_name=$generated_model -num_sessions_each_dim=$num_sessions_each_dim -dropout=$drop_out
# python3 src/train_CNN.py -load=$generated_model -save=$save_model -num_epoch=$num_epoch -train_file_dir=$train_file_dir -test_file_dir=$test_file_dir -num_sessions_each_dim=$num_sessions_each_dim