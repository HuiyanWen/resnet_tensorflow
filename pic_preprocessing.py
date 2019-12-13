"""Read and preprocess image data.
  Image processing occurs on a single image at a time. Image are read and
  preprocessed in parallel across multiple threads. The resulting images
  are concatenated together to form a single batch for training or evaluation.
"""

import math
import tensorflow as tf


def rotate_image(image, scope=None):
    """Rotate image
    thread_id comes from {0, 1, 2, 3} uniformly,
    we will apply rotation on 1/4 images of the trainning set
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for name_scope.
    Returns:
      rotated image
    """
    with tf.name_scope(name=scope, default_name='rotate_image'):
        angle = tf.random_uniform([], minval=-15 * math.pi / 180, maxval=15 * math.pi / 180, dtype=tf.float32,
                                  name="angle")
        distorted_image = tf.cond(
            tf.random_uniform([], 0, 1, tf.float32) < tf.constant(0.25, dtype=tf.float32),
            lambda: tf.contrib.image.rotate(image, angle, interpolation='BILINEAR'),
            lambda: image
        )
        return distorted_image

def distort_color(image, scope=None):
    """Distort the color of the image.
    thread_id comes from {0, 1, 2, 3} uniformly,
    and we will apply color distortion when thresd_id = 0 or 1,
    thus, only 1/2 images of the trainning set will be distorted
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for name_scope.
    Returns:
      color-distorted image
    """
    with tf.name_scope(name=scope, default_name='distort_color'):
        image = tf.cond(
            tf.random_uniform([], 0, 1, tf.float32) < tf.constant(0.6, dtype=tf.float32),
            lambda: tf.image.random_brightness(image, max_delta=50. / 255.),
            lambda: image
        )
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

def distort_image(image, height, width,scope=None):
    """Distort one image for training a network.
    Args:
      image: Tensor containing single image
      height: integer, image height
      width: integer, image width
      object_cover: float
      area_cover: float
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      thread_id: integer indicating the preprocessing thread.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    # Rotate image
    image = tf.reshape(image, [height, width, 1])
    distorted_image = rotate_image(image)
    # Distored image color
    distorted_image = distort_color(distorted_image)
    return distorted_image

def eval_image(image, height, width, scope=None):
  """Prepare one image for evaluation.
  Args:
    image: Tensor containing single image
    height: integer
    width: integer
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(values=[image, height, width], name=scope, default_name='eval_image'):
    # Crop the central region of the image with an area containing 80% of the original image.
    image = tf.image.central_crop(image, central_fraction=0.80)
    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image, [0])
    return image


def image_preprocessing(image, output_height, output_width):
    """Decode and preprocess one image for evaluation or training.
    Args:
      image: Tensor containing single image
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      output_height: integer
      output_width: integer
      train: boolean
    Returns:
      3-D float Tensor containing an appropriately scaled image
    Raises:
      ValueError: if user does not provide bounding box
    """
    image = tf.subtract(1.0, image)
    image = distort_image(image, output_height, output_width)
    image = tf.reshape(image, shape=[output_height, output_width, 1])
    return image

def preprocess_image(image, output_height, output_width):
  return image_preprocessing(image, output_height, output_width)
