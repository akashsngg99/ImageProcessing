import tensorflow as tf
import numpy as np

dst_test_file = "./test"

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('../data/lenet/model.ckpt-24000.meta')  # load graph
  for var in tf.trainable_variables():  # get the param names
    print(var.name) # print parameters' names
  new_saver.restore(sess, tf.train.latest_checkpoint('../data/lenet/'))  # find the newest training result
  all_vars = tf.trainable_variables()

  f = open(dst_test_file, 'w+')
  for v in all_vars:
    f.writelines(v.name)

    v_4d = np.array(sess.run(v))
    if len(v_4d.shape) == 4:
      v_4d = v_4d.swapaxes(0, 2)
      v_4d = v_4d.swapaxes(1, 3)
    # print(len(v_4d.shape))

    print(v_4d.shape)
    if len(v_4d.shape) == 1:
      # print(v_4d.shape)
      for v in v_4d:
        f.writelines(str(v))

    elif len(v_4d.shape) == 2:
      for v_1d in v_4d:
        # print(v_1d.shape)
        for v in v_1d:
          f.writelines(str(v))

    elif len(v_4d.shape) == 3:
      for v_2d in v_4d:
        # print(v_2d.shape)
        for v_1d in v_2d:
          # print(v_1d.shape)
          for v in v_1d:
            f.writelines(str(v))

    elif len(v_4d.shape) == 4:
      for v_3d in v_4d:
        # print(v_3d.shape)
        for v_2d in v_3d:
          # print(v_2d.shape)
          for v_1d in v_2d:
            # print(v_1d.shape)
            for v in v_1d:
              f.writelines(str(v))

    # print(v_4d)