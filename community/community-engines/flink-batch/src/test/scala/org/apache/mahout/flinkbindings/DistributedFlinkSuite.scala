/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.mahout.flinkbindings

import java.util.concurrent.TimeUnit

import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.flink.test.util.{ForkableFlinkMiniCluster, TestBaseUtils}
import org.apache.mahout.math.drm.DistributedContext
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{ConfigMap, Suite}

import scala.concurrent.duration.FiniteDuration

trait DistributedFlinkSuite extends DistributedMahoutSuite { this: Suite =>

  protected implicit var mahoutCtx: DistributedContext = _
  protected var env: ExecutionEnvironment = null

  var cluster: Option[ForkableFlinkMiniCluster] = None
  val parallelism = 4
  protected val DEFAULT_AKKA_ASK_TIMEOUT: Long = 1000
  protected var DEFAULT_TIMEOUT: FiniteDuration = new FiniteDuration(DEFAULT_AKKA_ASK_TIMEOUT, TimeUnit.SECONDS)

  def initContext() {
    mahoutCtx = wrapContext(env)
  }

  override def beforeEach() {
    initContext()
  }

  override def afterEach() {
    super.afterEach()
  }

  override protected def afterAll(configMap: ConfigMap): Unit = {
    super.afterAll(configMap)
    cluster.foreach(c => TestBaseUtils.stopCluster(c, DEFAULT_TIMEOUT))
  }

  override protected def beforeAll(configMap: ConfigMap): Unit = {
    super.beforeAll(configMap)

    val cl = TestBaseUtils.startCluster(
      1,
      parallelism,
      false,
      false,
      true)

    env = ExecutionEnvironment.createLocalEnvironment(parallelism)

    cluster = Some(cl)
    initContext()
  }

}
