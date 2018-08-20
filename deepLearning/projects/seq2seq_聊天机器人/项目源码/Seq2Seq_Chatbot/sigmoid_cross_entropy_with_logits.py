#encoding=utf8
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
input_data = tfe.Variable(np.random.rand(1, 3), dtype=tf.float32)#logits

labels=[[1.0, 0.0, 0.0]]

_sentinel=None
#nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits",
                         #labels=labels, logits=input_data)
with ops.name_scope( "logistic_loss", [input_data, labels]) as name:
  logits = ops.convert_to_tensor(input_data, name="logits")
  labels = ops.convert_to_tensor(labels, name="labels")
  try:
    labels.get_shape().merge_with(logits.get_shape())
    # print(sess.run(labels.get_shape().merge_with(logits.get_shape())))
  except ValueError:
    raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                     (logits.get_shape(), labels.get_shape()))
  zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
  #print(sess.run(zeros))
  cond = (logits >= zeros)
  #print(sess.run(cond))
  relu_logits = array_ops.where(cond, logits, zeros)
  #print(sess.run(relu_logits))
  neg_abs_logits = array_ops.where(cond, -logits, logits)  
  output =math_ops.add(
      relu_logits - logits * labels,
      math_ops.log1p(math_ops.exp(neg_abs_logits)),#tf.log1p(x,name=None) 求x加1的自然对数.math_ops.exp为e^x次方
      name=name)

#z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
#= z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
#= z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
#= z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
#= (1 - z) * x + log(1 + exp(-x))
#= x - x * z + log(1 + exp(-x))  

  #init = tf.global_variables_initializer()
  #sess.run(init)
  #print(sess.run(input_data))
  print(output)
    # [[ 0.5583781   1.06925142  1.08170223]]