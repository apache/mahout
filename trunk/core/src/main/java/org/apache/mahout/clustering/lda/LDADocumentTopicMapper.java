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

package org.apache.mahout.clustering.lda;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

public class LDADocumentTopicMapper
    extends Mapper<WritableComparable<?>,VectorWritable,WritableComparable<?>,VectorWritable> {

  private LDAInference infer;

  @Override
  protected void map(WritableComparable<?> key,
                     VectorWritable wordCountsWritable,
                     Context context) throws IOException, InterruptedException {

    Vector wordCounts = wordCountsWritable.get();
    try {
      LDAInference.InferredDocument doc = infer.infer(wordCounts);
      context.write(key, new VectorWritable(doc.getGamma().normalize(1)));
    } catch (ArrayIndexOutOfBoundsException e1) {
      throw new IllegalStateException(
         "This is probably because the --numWords argument is set too small.  \n"
         + "\tIt needs to be >= than the number of words (terms actually) in the corpus and can be \n"
         + "\tlarger if some storage inefficiency can be tolerated.", e1);
    }
  }

  public void configure(LDAState myState) {
    this.infer = new LDAInference(myState);
  }

  public void configure(Configuration job) {
    LDAState myState = LDADriver.createState(job);
    configure(myState);
  }

  @Override
  protected void setup(Context context) {
    configure(context.getConfiguration());
  }
}
