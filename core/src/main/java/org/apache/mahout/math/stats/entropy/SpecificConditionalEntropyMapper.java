/**
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

package org.apache.mahout.math.stats.entropy;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.math.VarIntWritable;

import java.io.IOException;

/**
 * Converts the key from {@link StringTuple} with values [key, value] to {@link Text} with value key.
 */
public class SpecificConditionalEntropyMapper extends Mapper<StringTuple, VarIntWritable, Text, VarIntWritable> {

  private final Text resultKey = new Text();

  @Override
  protected void map(StringTuple key, VarIntWritable value, Context context)
      throws IOException, InterruptedException {
    resultKey.set(key.stringAt(0));
    context.write(resultKey, value);
  }

}
