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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;

/**
 * counts all unique users, we ensure that we see userIDs sorted in ascending order via
 * secondary sort, so we don't have to buffer all of them
 */
public class CountUsersReducer extends
    Reducer<CountUsersKeyWritable,VarLongWritable, VarIntWritable,NullWritable> {

  @Override
  protected void reduce(CountUsersKeyWritable key,
                        Iterable<VarLongWritable> userIDs,
                        Context context) throws IOException, InterruptedException {

    long lastSeenUserID = Long.MIN_VALUE;
    int numberOfUsers = 0;

    for (VarLongWritable writable : userIDs) {
      long currentUserID = writable.get();
      if (currentUserID > lastSeenUserID) {
        lastSeenUserID = currentUserID;
        numberOfUsers++;
      }
    }
    context.write(new VarIntWritable(numberOfUsers), NullWritable.get());
  }

}
