# Pytorch tensorflow summary for using tensorboard.
# This utility is motivated by https://github.com/lanpa/tensorboard-pytorch.

import os
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

class PytorchEmbedding(object):
	def __init__(self, log_dir):
		self.log_dir = log_dir
		self.sess = tf.Session()
		self.writer = tf.summary.FileWriter(log_dir)
		self.config = projector.ProjectorConfig()

	def add_embedding(self, embedding, name='embedding', metadata_path=None):
		embedding_var = tf.Variable(embedding, name=name)
		embedding = self.config.embeddings.add()
		embedding.tensor_name = embedding_var.name
		if metadata_path != None:
			embedding.metadata_path = metadata_path
		projector.visualize_embeddings(self.writer, self.config)

	def save_checkpoint(self):
		self.sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"), 1)
