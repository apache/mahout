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

package org.apache.mahout.h2o.common

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{SequenceFile, Writable}

/**
 * Deprecated Hadoop 1 api which we currently explicitly import via Mahout dependencies. May not work
 * with Hadoop 2.0
 */
object Hadoop1HDFSUtil extends HDFSUtil {

  
  def readDrmHeader(path: String): DrmMetadata = {
    val dfsPath = new Path(path)
    val fs = dfsPath.getFileSystem(new Configuration())

    val partFilePath:Path = fs.listStatus(dfsPath)

        // Filter out anything starting with .
        .filter { s => !s.getPath.getName.startsWith("\\.") && !s.getPath.getName.startsWith("_") && !s.isDir }

        // Take path
        .map(_.getPath)

        // Take only one, if any
        .headOption

        // Require there's at least one partition file found.
        .getOrElse {
      throw new IllegalArgumentException(s"No partition files found in ${dfsPath.toString}.")
    }

    val reader = new SequenceFile.Reader(fs, partFilePath, fs.getConf)
    try {
      new DrmMetadata(
        keyTypeWritable = reader.getKeyClass.asSubclass(classOf[Writable]),
        valueTypeWritable = reader.getValueClass.asSubclass(classOf[Writable])
      )
    } finally {
      reader.close()
    }

  }

}
