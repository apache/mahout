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
import org.apache.flink.configuration.GlobalConfiguration
import org.apache.mahout.math.drm.DistributedContext
import org.apache.mahout.math.drm.DistributedEngine

class FlinkDistributedContext(val env: ExecutionEnvironment) extends DistributedContext {

  val mahoutHome = getMahoutHome()

  GlobalConfiguration.loadConfiguration(mahoutHome + "/conf/flink-config.yaml")

  val conf = GlobalConfiguration.getConfiguration

  var degreeOfParallelism: Int = 0

  if (conf != null) {
    degreeOfParallelism = conf.getInteger("parallelism.default", Runtime.getRuntime.availableProcessors)
  } else {
    degreeOfParallelism = Runtime.getRuntime.availableProcessors
  }

  env.setParallelism(degreeOfParallelism)
  
  val engine: DistributedEngine = FlinkEngine

  override def close() {
    // TODO
  }

}
