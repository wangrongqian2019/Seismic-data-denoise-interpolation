
### train
data_path =
test_data_path =

result_path =

sample_size_train = 8000

learning_rate = 3e-4
end_lr = 1e-6

checkpoint_epoch = 0
num_epochs = 300
batchsize = 16
regular = 0

receiver_num = 560
img_resolution1 = 2000
img_resolution2 = 200

### test

val_data_path = test_data_path
Output_path='./OUTPUTS/'

test_checkpoint_epoch = 0

sample_id_test = 5400
sample_size_test = 1

num_epochs_test = 500
test_tag = 0
sample_size_val = 100
sample_id = 5000