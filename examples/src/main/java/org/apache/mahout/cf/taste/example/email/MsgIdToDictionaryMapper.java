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
 * Assumes the input is in the format created by {@link org.apache.mahout.text.SequenceFilesFromMailArchives}
 */
public final class MsgIdToDictionaryMapper extends Mapper<Text, Text, Text, VarIntWritable> {

  @Override
  protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {
    //message id is in the key: /201008/AANLkTikvVnhNH+Y5AGEwqd2=u0CFv2mCm0ce6E6oBnj1@mail.gmail.com
    String keyStr = key.toString();
    int idx = keyStr.lastIndexOf('@'); //find the last @
    if (idx == -1) {
      context.getCounter(EmailUtility.Counters.NO_MESSAGE_ID).increment(1);
    } else {
      //found the @, now find the last slash before the @ and grab everything after that
      idx = keyStr.lastIndexOf('/', idx);
      String msgId = keyStr.substring(idx + 1);
      if (EmailUtility.WHITESPACE.matcher(msgId).matches()) {
        context.getCounter(EmailUtility.Counters.NO_MESSAGE_ID).increment(1);
      } else {
        context.write(new Text(msgId), new VarIntWritable(1));
      }
    }
  }
}
