import tensorflow as tf
import glob

train_files = sorted(glob.glob('./train/*.tfrecord'))
for f_i, file in enumerate(train_files): 
    print(f_i) 
    total_images += sum([1 for _ in tf.python_io.tf_record_iterator(file)])