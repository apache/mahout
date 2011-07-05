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

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VarIntWritable;

import java.io.IOException;

/**
 * Emits the key and the count of 1 as {@link VarIntWritable}.
 */
public final class KeyCounterMapper extends Mapper<Writable, Object, Writable, VarIntWritable> {

  private static final VarIntWritable ONE = new VarIntWritable(1);

  @Override
  protected void map(Writable key, Object value, Context context) throws IOException, InterruptedException {
    context.write(key, ONE);
  }

}
