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

import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.mahout.math.drm.DistributedContext
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.Suite


trait DistributedFlinkSuite extends DistributedMahoutSuite { this: Suite =>

  protected implicit var mahoutCtx: DistributedContext = _
  protected var env: ExecutionEnvironment = null

  def initContext() {
    env = ExecutionEnvironment.getExecutionEnvironment
    mahoutCtx = wrapContext(env)
  }

  override def beforeEach() {
    initContext()
  }

  override def afterEach() {
    super.afterEach()
//    env.execute("Mahout Flink Binding Test Suite")
  }

}