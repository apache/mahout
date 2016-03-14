package org.apache.mahout.sparkbindings.drm

import org.apache.mahout.math.drm.CheckpointedDrm
import org.apache.mahout.sparkbindings.DrmRdd

/** Additional Spark-specific operations. Requires underlying DRM to be running on Spark backend. */
class CheckpointedDrmSparkOps[K](drm: CheckpointedDrm[K]) {

  assert(drm.isInstanceOf[CheckpointedDrmSpark[K]], "must be a Spark-backed matrix")

  private[sparkbindings] val sparkDrm = drm.asInstanceOf[CheckpointedDrmSpark[K]]

  /** Spark matrix customization exposure */
  def rdd = sparkDrm.rddInput.asRowWise()

}
