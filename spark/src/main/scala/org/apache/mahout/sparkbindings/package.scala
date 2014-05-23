/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout

import org.apache.spark.{SparkConf, SparkContext}
import java.io._
import scala.collection.mutable.ArrayBuffer
import org.apache.mahout.common.IOUtils
import org.apache.log4j.Logger
import org.apache.mahout.math.drm._
import scala.reflect.ClassTag
import org.apache.mahout.sparkbindings.drm.{SparkBCast, CheckpointedDrmSparkOps, CheckpointedDrmSpark}
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.mahout.math.{VectorWritable, Vector, MatrixWritable, Matrix}
import org.apache.hadoop.io.Writable

/** Public api for Spark-specific operators */
package object sparkbindings {

  private[sparkbindings] val log = Logger.getLogger("org.apache.mahout.sparkbindings")

  /** Row-wise organized DRM rdd type */
  type DrmRdd[K] = RDD[DrmTuple[K]]

  /**
   * Blockifed DRM rdd (keys of original DRM are grouped into array corresponding to rows of Matrix
   * object value
   */
  type BlockifiedDrmRdd[K] = RDD[BlockifiedDrmTuple[K]]

  /**
   * Create proper spark context that includes local Mahout jars
   * @param masterUrl
   * @param appName
   * @param customJars
   * @return
   */
  def mahoutSparkContext(masterUrl: String, appName: String,
      customJars: TraversableOnce[String] = Nil,
      sparkConf: SparkConf = new SparkConf(),
      addMahoutJars: Boolean = true
      ): SparkDistributedContext = {
    val closeables = new java.util.ArrayDeque[Closeable]()

    try {

      if (addMahoutJars) {
        var mhome = System.getenv("MAHOUT_HOME")
        if (mhome == null) mhome = System.getProperty("mahout.home")

        if (mhome == null)
          throw new IllegalArgumentException("MAHOUT_HOME is required to spawn mahout-based spark jobs.")

        // Figure Mahout classpath using $MAHOUT_HOME/mahout classpath command.

        val fmhome = new File(mhome)
        val bin = new File(fmhome, "bin")
        val exec = new File(bin, "mahout")
        if (!exec.canExecute)
          throw new IllegalArgumentException("Cannot execute %s.".format(exec.getAbsolutePath))

        val p = Runtime.getRuntime.exec(Array(exec.getAbsolutePath, "-spark", "classpath"))

        closeables.addFirst(new Closeable {
          def close() {
            p.destroy()
          }
        })

        val r = new BufferedReader(new InputStreamReader(p.getInputStream))
        closeables.addFirst(r)

        val w = new StringWriter()
        closeables.addFirst(w)

        var continue = true;
        val jars = new ArrayBuffer[String]()
        do {
          val cp = r.readLine()
          if (cp == null)
            throw new IllegalArgumentException("Unable to read output from \"mahout classpath\"")

          val j = cp.split(File.pathSeparatorChar)
          if (j.size > 10) {
            // assume this is a valid classpath line
            jars ++= j
            continue = false
          }
        } while (continue)

        //      if (s_log.isDebugEnabled) {
        //        s_log.debug("Mahout jars:")
        //        jars.foreach(j => s_log.debug(j))
        //      }

        // context specific jars
        val mcjars = jars.filter(j =>
          j.matches(".*mahout-math-.*\\.jar") ||
              j.matches(".*mahout-math-scala-.*\\.jar") ||
              j.matches(".*mahout-mrlegacy-.*\\.jar") ||
              j.matches(".*mahout-spark-.*\\.jar")
        ).filter(n => !n.matches(".*-tests.jar") && !n.matches(".*-sources.jar")) ++
            SparkContext.jarOfClass(classOf[DrmLike[_]])

        if (log.isDebugEnabled) {
          log.debug("Mahout jars:")
          mcjars.foreach(j => log.debug(j))
        }

        sparkConf.setJars(jars = mcjars.toSeq ++ customJars)

      } else {
        // In local mode we don't care about jars, do we?
        sparkConf.setJars(customJars.toSeq)
      }

      sparkConf.setAppName(appName).setMaster(masterUrl)
          .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
          .set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator")

      if (System.getenv("SPARK_HOME") != null) {
        sparkConf.setSparkHome(System.getenv("SPARK_HOME"))
      }

      new SparkDistributedContext(new SparkContext(config = sparkConf))

    } finally {
      IOUtils.close(closeables)
    }
  }

  implicit def sdc2sc(sdc: SparkDistributedContext): SparkContext = sdc.sc

  implicit def sc2sdc(sc: SparkContext): SparkDistributedContext = new SparkDistributedContext(sc)

  implicit def dc2sc(dc:DistributedContext):SparkContext = {
    assert (dc.isInstanceOf[SparkDistributedContext],"distributed context must be Spark-specific.")
    sdc2sc(dc.asInstanceOf[SparkDistributedContext])
  }

  /** Broadcast transforms */
  implicit def sb2bc[T](b:Broadcast[T]):BCast[T] = new SparkBCast(b)

  /** Adding Spark-specific ops */
  implicit def cpDrm2cpDrmSparkOps[K: ClassTag](drm: CheckpointedDrm[K]): CheckpointedDrmSparkOps[K] =
    new CheckpointedDrmSparkOps[K](drm)

  implicit def drm2cpDrmSparkOps[K:ClassTag](drm:DrmLike[K]):CheckpointedDrmSparkOps[K] = drm:CheckpointedDrm[K]

  private[sparkbindings] implicit def m2w(m: Matrix): MatrixWritable = new MatrixWritable(m)

  private[sparkbindings] implicit def w2m(w: MatrixWritable): Matrix = w.get()

  private[sparkbindings] implicit def v2w(v: Vector): VectorWritable = new VectorWritable(v)

  private[sparkbindings] implicit def w2v(w:VectorWritable):Vector = w.get()

  def drmWrap[K : ClassTag](
      rdd: DrmRdd[K],
      nrow: Int = -1,
      ncol: Int = -1
      ): CheckpointedDrm[K] =
    new CheckpointedDrmSpark[K](
      rdd = rdd,
      _nrow = nrow,
      _ncol = ncol
    )


}
