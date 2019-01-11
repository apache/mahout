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

package org.apache.mahout.cf.taste.hadoop;

/**
 * <h1>Input</h1>
 *
 * <p>
 * Intended for use with {@link org.apache.hadoop.mapreduce.lib.input.TextInputFormat};
 * accepts line number / line pairs as
 * {@link org.apache.hadoop.io.LongWritable}/{@link org.apache.hadoop.io.Text} pairs.
 * </p>
 *
 * <p>
 * Each line is assumed to be of the form {@code userID,itemID,preference}, or {@code userID,itemID}.
 * </p>
 *
 * <h1>Output</h1>
 *
 * <p>
 * Outputs the user ID as a {@link org.apache.mahout.math.VarLongWritable} mapped to the item ID and preference as a
 * {@link EntityPrefWritable}.
 * </p>
 */
public final class ToItemPrefsMapper extends ToEntityPrefsMapper {

  public ToItemPrefsMapper() {
    super(false);
  }
  
}
