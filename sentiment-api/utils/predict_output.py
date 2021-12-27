import numpy as np

def get_final_output(pred, classes):
  predictions = pred[0]
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
  for i in range(pred.shape[1]):
    return classes[i], predictions[i]

def get_output_confident_type(pred, classes):
  predictions = pred[0]
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
  return '{} has confidence = {:.4f}'.format(classes[0], predictions[0])

def get_output_confident_opposite_type(pred, classes):
  predictions = pred[0]
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
  return '{} has confidence = {:.4f}'.format(classes[1], predictions[1])