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

package org.apache.mahout.common

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}

/**
 * Returns a [[java.lang.String]], which is comma delimited list of URIs discovered based on parameters
 * in the constructor.
 * The String is formatted to be input into [[org.apache.spark.SparkContext#textFile()]]
 * @param pathURI Where to start looking for inFiles, may be a list of comma delimited URIs
 * @param filePattern regex that must match the entire filename to have the file returned
 * @param recursive true traverses the filesystem recursively, default = false
 */

case class HDFSPathSearch(pathURI: String, filePattern: String = "", recursive: Boolean = false) {

  val conf = new Configuration()
  val fs = FileSystem.get(conf)

  /**
   * Returns a string of comma delimited URIs matching the filePattern
   * When pattern matching dirs are never returned, only traversed.
   */
  def uris: String = {
    if (!filePattern.isEmpty){ // have file pattern so
    val pathURIs = pathURI.split(",")
      var files = ""
      for ( uri <- pathURIs ){
        files = findFiles(uri, filePattern, files)
      }
      if (files.length > 0 && files.endsWith(",")) files = files.dropRight(1) // drop the last comma
      files
    }else{
      pathURI
    }
  }

  /**
   * Find matching files in the dir, recursively call self when another directory is found
   * Only files are matched, directories are traversed but never return a match
   */
  private def findFiles(dir: String, filePattern: String = ".*", files: String = ""): String = {
    val seed = fs.getFileStatus(new Path(dir))
    var f: String = files

    if (seed.isDir) {
      val fileStatuses: Array[FileStatus] = fs.listStatus(new Path(dir))
      for (fileStatus <- fileStatuses) {
        if (fileStatus.getPath().getName().matches(filePattern)
          && !fileStatus.isDir) {
          // found a file
          if (fileStatus.getLen() != 0) {
            // file is not empty
            f = f + fileStatus.getPath.toUri.toString + ","
          }
        } else if (fileStatus.isDir && recursive) {
          f = findFiles(fileStatus.getPath.toString, filePattern, f)
        }
      }
    } else { f = dir }// was a filename not dir
    f
  }

}
