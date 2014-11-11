package org.apache.mahout.math.drm.logical

import org.apache.mahout.math.drm.{DrmLike, CheckpointedDrm}

import scala.reflect.ClassTag

abstract class AggregateAction[U: ClassTag, K:ClassTag]{
  protected[drm] var A: DrmLike[K]

  def classTagU: ClassTag[U] = implicitly[ClassTag[U]]

  def classTagK: ClassTag[K] = implicitly[ClassTag[K]]
  }