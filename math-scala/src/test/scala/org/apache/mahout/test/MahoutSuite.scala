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
package org.apache.mahout.test

import java.io.File
import org.scalatest._
import org.apache.mahout.common.RandomUtils

trait MahoutSuite extends BeforeAndAfterEach with LoggerConfiguration with Matchers {
  this: Suite =>

  final val TmpDir = "tmp/"

  override protected def beforeEach() {
    super.beforeEach()
    RandomUtils.useTestSeed()
  }

  override protected def beforeAll(configMap: ConfigMap) {
    super.beforeAll(configMap)

    // just in case there is an existing tmp dir clean it before every suite
    deleteDirectory(new File(TmpDir))
  }

  override protected def afterEach() {

    // clean the tmp dir after every test
    deleteDirectory(new File(TmpDir))

    super.afterEach()
  }

  /** Delete directory no symlink checking and exceptions are not caught */
  private def deleteDirectory(path: File): Unit = {
    if (path.isDirectory)
      for (files <- path.listFiles) deleteDirectory(files)
    path.delete
  }
}
