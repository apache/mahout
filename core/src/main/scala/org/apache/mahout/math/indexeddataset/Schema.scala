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

package org.apache.mahout.math.indexeddataset

import scala.collection.mutable.HashMap

/**
 * Syntactic sugar for mutable.HashMap[String, Any]
 * @param params list of mappings for instantiation {{{val mySchema = new Schema("one" -> 1, "two" -> "2", ...)}}}
 */
class Schema(params: Tuple2[String, Any]*) extends HashMap[String, Any] {
  // note: this require a mutable HashMap, do we care?
  this ++= params

  /**
   * Constructor for copying an existing Schema
   * @param schemaToClone return a copy of this Schema
   */
  def this(schemaToClone: Schema){
    this()
    this ++= schemaToClone
  }
}

// These can be used to keep the text in and out fairly standard to Mahout, where an application specific
// format is not required. These apply to formatting of [[org.apache.mahout.math.indexeddataset.IndexedDataset]]
// which can be used to create a Mahout DRM for DSL ops.

/**
 * Simple default Schema for typical text delimited element file input
 * This tells the reader to input elements of the default (rowID<comma, tab, or space>columnID
 * <comma, tab, or space>here may be other ignored text...)
 */
object DefaultIndexedDatasetElementReadSchema extends Schema(
  "delim" -> "[,\t ]", //comma, tab or space
  "filter" -> "",
  "rowIDColumn" -> 0,
  "columnIDPosition" -> 1,
  "filterColumn" -> -1)

/**
 * Default Schema for text delimited [[org.apache.mahout.math.indexeddataset.IndexedDataset]] file output with
 * one row per line.
 * The default form:
 * (rowID<tab>columnID1:score1<space>columnID2:score2...)
 */
object DefaultIndexedDatasetWriteSchema extends Schema(
  "rowKeyDelim" -> "\t",
  "columnIdStrengthDelim" -> ":",
  "elementDelim" -> " ",
  "omitScore" -> false)

/**
 * Default Schema for typical text delimited [[org.apache.mahout.math.indexeddataset.IndexedDataset]] file
 * row-wise input. This tells the reader to input text lines of the form:
 * (rowID<tab>columnID1:score1,columnID2:score2,...)
 */
object DefaultIndexedDatasetReadSchema extends Schema(
  "rowKeyDelim" -> "\t",
  "columnIdStrengthDelim" -> ":",
  "elementDelim" -> " ",
  "omitScore" -> false)

/**
 * Default Schema for reading a text delimited [[org.apache.mahout.math.indexeddataset.IndexedDataset]] file  where
 * the score of any element is ignored.
 * This tells the reader to input DRM lines of the form
 * (rowID<tab>columnID1:score1<space>columnID2:score2...) remember the score is ignored.
 * Alternatively the format can be
 * (rowID<tab>columnID1<space>columnID2 ...) where presence indicates a score of 1. This is the default
 * output format for [[IndexedDatasetWriteBooleanSchema]]
 */
object IndexedDatasetReadBooleanSchema extends Schema(
  "rowKeyDelim" -> "\t",
  "columnIdStrengthDelim" -> ":",
  "elementDelim" -> " ",
  "omitScore" -> true)

/**
 * Default Schema for typical text delimited [[org.apache.mahout.math.indexeddataset.IndexedDataset]] file output
 * where the score of a element is omitted. This tells the writer to output
 * [[org.apache.mahout.math.indexeddataset.IndexedDataset]] row of the form
 * (rowID<tab>columnID1<space>columnID2...)
 */
object IndexedDatasetWriteBooleanSchema extends Schema(
  "rowKeyDelim" -> "\t",
  "columnIdStrengthDelim" -> ":",
  "elementDelim" -> " ",
  "omitScore" -> true)

