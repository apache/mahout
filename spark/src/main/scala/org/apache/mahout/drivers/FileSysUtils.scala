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

/**
  * Returns a [[java.lang.String]]comma delimited list of URIs discovered based on parameters in the constructor.
  * The String is formatted to be input into [[org.apache.spark.SparkContext.textFile()]]
  *
  * @param pathURI Where to start looking for inFiles, only HDFS and local URI are currently
  *                supported
  * @param filePattern regex that must match the entire filename to have the file included in the returned list
  * @param recursive true traverses the filesystem recursively
  */

case class FileSysUtils(pathURI: String, filePattern: String = "", recursive: Boolean = false) {
  // todo: There is an HDFS filestatus method that collects multiple inFiles, see if this is the right thing to use
  // todo: check to see if the input is a supported URI for collection or recursive search but just pass through otherwise
  def uris = {pathURI}
}
