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

package org.apache.mahout.classifier.bayes.common;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.MultipleOutputFormat;
import org.apache.hadoop.util.Progressable;

import java.io.IOException;

/**
 * This class extends the MultipleOutputFormat, allowing to write the output data to different output files in sequence
 * file output format.
 */
public class BayesWeightSummerOutputFormat extends MultipleOutputFormat<WritableComparable<?>, Writable> {

  private SequenceFileOutputFormat<WritableComparable<?>, Writable> theSequenceFileOutputFormat = null;

  @Override
  protected RecordWriter<WritableComparable<?>, Writable> getBaseRecordWriter(
      FileSystem fs, JobConf job, String name, Progressable arg3)
      throws IOException {
    if (theSequenceFileOutputFormat == null) {
      theSequenceFileOutputFormat = new SequenceFileOutputFormat<WritableComparable<?>, Writable>();
    }
    return theSequenceFileOutputFormat.getRecordWriter(fs, job, name, arg3);
  }

  @Override
  protected String generateFileNameForKeyValue(WritableComparable<?> k, Writable v,
                                               String name) {
    Text key = (Text) k;

    char firstChar = key.toString().charAt(0);
    if (firstChar == '*') { //sum of weight of all features for all label Sigma_kSigma_j
      return "Sigma_kSigma_j/" + name;
    } else if (firstChar == ',') { //sum of weight for all labels for a feature Sigma_j
      return "Sigma_j/" + name;
    } else if (firstChar == '_') { //sum of weights for all features for a label Sigma_k
      return "Sigma_k/" + name;
    }
    return "JunkFileThisShouldNotHappen";
  }

}
