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

package org.apache.mahout.drivers

import org.apache.mahout.math.drm.DistributedContext

/** Extended by a platform specific version of this class to create a Mahout CLI driver. */
abstract class MahoutDriver {

  implicit protected var mc: DistributedContext = _
  implicit protected var parser: MahoutOptionParser = _

  var _useExistingContext: Boolean = false // used in the test suite to reuse one context per suite

  /** must be overriden to setup the DistributedContext mc*/
  protected def start() : Unit

  /** Override (optionally) for special cleanup */
  protected def stop(): Unit = {
    if (!_useExistingContext) mc.close
  }

  /** This is where you do the work, call start first, then before exiting call stop */
  protected def process(): Unit

  /** Parse command line and call process */
  def main(args: Array[String]): Unit

}
