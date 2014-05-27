package org.apache.mahout.math.drm

object CacheHint extends Enumeration {

  type CacheHint = Value

  val NONE,
  DISK_ONLY,
  DISK_ONLY_2,
  MEMORY_ONLY,
  MEMORY_ONLY_2,
  MEMORY_ONLY_SER,
  MEMORY_ONLY_SER_2,
  MEMORY_AND_DISK,
  MEMORY_AND_DISK_2,
  MEMORY_AND_DISK_SER,
  MEMORY_AND_DISK_SER_2 = Value

}
