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

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.map.OpenObjectIntHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public final class MailToRecMapper extends Mapper<Text, Text, Text, LongWritable> {

  private static final Logger log = LoggerFactory.getLogger(MailToRecMapper.class);

  private final OpenObjectIntHashMap<String> fromDictionary = new OpenObjectIntHashMap<String>();
  private final OpenObjectIntHashMap<String> msgIdDictionary = new OpenObjectIntHashMap<String>();
  private String separator = "\n";
  private int fromIdx;
  private int refsIdx;

  public enum Counters {
    REFERENCE, ORIGINAL
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    String fromPrefix = conf.get(EmailUtility.FROM_PREFIX);
    String msgPrefix = conf.get(EmailUtility.MSG_IDS_PREFIX);
    fromIdx = conf.getInt(EmailUtility.FROM_INDEX, 0);
    refsIdx = conf.getInt(EmailUtility.REFS_INDEX, 1);
    EmailUtility.loadDictionaries(conf, fromPrefix, fromDictionary, msgPrefix, msgIdDictionary);
    log.info("From Dictionary size: {} Msg Id Dictionary size: {}", fromDictionary.size(), msgIdDictionary.size());
    separator = context.getConfiguration().get(EmailUtility.SEPARATOR);
  }

  @Override
  protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {

    int msgIdKey = Integer.MIN_VALUE;


    int fromKey = Integer.MIN_VALUE;
    String valStr = value.toString();
    String[] splits = StringUtils.splitByWholeSeparatorPreserveAllTokens(valStr, separator);

    if (splits != null && splits.length > 0) {
      if (splits.length > refsIdx) {
        String from = EmailUtility.cleanUpEmailAddress(splits[fromIdx]);
        fromKey = fromDictionary.get(from);
      }
      //get the references
      if (splits.length > refsIdx) {
        String[] theRefs = EmailUtility.parseReferences(splits[refsIdx]);
        if (theRefs != null && theRefs.length > 0) {
          //we have a reference, the first one is the original message id, so map to that one if it exists
          msgIdKey = msgIdDictionary.get(theRefs[0]);
          context.getCounter(Counters.REFERENCE).increment(1);
        }
      }
    }
    //we don't have any references, so use the msg id
    if (msgIdKey == Integer.MIN_VALUE) {
      //get the msg id and the from and output the associated ids
      String keyStr = key.toString();
      int idx = keyStr.lastIndexOf('/');
      if (idx != -1) {
        String msgId = keyStr.substring(idx + 1);
        msgIdKey = msgIdDictionary.get(msgId);
        context.getCounter(Counters.ORIGINAL).increment(1);
      }
    }

    if (msgIdKey != Integer.MIN_VALUE && fromKey != Integer.MIN_VALUE) {
      context.write(new Text(fromKey + "," + msgIdKey), new LongWritable(1));
    }
  }


}
