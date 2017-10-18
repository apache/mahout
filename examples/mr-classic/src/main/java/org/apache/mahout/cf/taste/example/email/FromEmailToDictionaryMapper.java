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

package org.apache.mahout.cf.taste.example.email;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VarIntWritable;

import java.io.IOException;

/**
 *  Assumes the input is in the format created by {@link org.apache.mahout.text.SequenceFilesFromMailArchives}
 */
public final class FromEmailToDictionaryMapper extends Mapper<Text, Text, Text, VarIntWritable> {

  private String separator;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    separator = context.getConfiguration().get(EmailUtility.SEPARATOR);
  }

  @Override
  protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {
    //From is in the value
    String valStr = value.toString();
    int idx = valStr.indexOf(separator);
    if (idx == -1) {
      context.getCounter(EmailUtility.Counters.NO_FROM_ADDRESS).increment(1);
    } else {
      String full = valStr.substring(0, idx);
      //do some cleanup to normalize some things, like: Key: karthik ananth <karthik.jcecs@gmail.com>: Value: 178
      //Key: karthik ananth [mailto:karthik.jcecs@gmail.com]=20: Value: 179
      //TODO: is there more to clean up here?
      full = EmailUtility.cleanUpEmailAddress(full);

      if (EmailUtility.WHITESPACE.matcher(full).matches()) {
        context.getCounter(EmailUtility.Counters.NO_FROM_ADDRESS).increment(1);
      } else {
        context.write(new Text(full), new VarIntWritable(1));
      }
    }

  }
}
