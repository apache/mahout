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

package org.apache.mahout.sparkbindings.test

import org.scalatest.Suite
import org.apache.spark.SparkConf
import org.apache.mahout.sparkbindings._
import org.apache.mahout.test.{DistributedMahoutSuite, MahoutSuite}
import org.apache.mahout.math.drm.DistributedContext

trait DistributedSparkSuite extends DistributedMahoutSuite with LoggerConfiguration {
  this: Suite =>

  protected implicit var mahoutCtx: DistributedContext = _
  protected var masterUrl = null.asInstanceOf[String]

  override protected def beforeEach() {
    super.beforeEach()

    masterUrl = "local[2]"
    mahoutCtx = mahoutSparkContext(masterUrl = this.masterUrl,
      appName = "MahoutLocalContext",
      // Do not run MAHOUT_HOME jars in unit tests.
      addMahoutJars = false,
      sparkConf = new SparkConf()
          .set("spark.kryoserializer.buffer.mb", "15")
          .set("spark.akka.frameSize", "30")
          .set("spark.default.parallelism", "10")
    )
  }

  override protected def afterEach() {
    if (mahoutCtx != null) {
      try {
        mahoutCtx.close()
      } finally {
        mahoutCtx = null
      }
    }
    super.afterEach()
  }
}
