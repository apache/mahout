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

package org.apache.mahout.classifier.naivebayes.test;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.naivebayes.AbstractNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.ComplementaryNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.regex.Pattern;

/**
 * Run the input through the model and see if it matches.
 * <p/>
 * The output value is the generated label, the Pair is the expected label and true if they match:
 */
public class BayesTestMapper extends Mapper<Text, VectorWritable, Text, VectorWritable> {

  private static final Pattern SLASH = Pattern.compile("/");

  private AbstractNaiveBayesClassifier classifier;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    Path modelPath = HadoopUtil.getSingleCachedFile(conf);
    NaiveBayesModel model = NaiveBayesModel.materialize(modelPath, conf);
    boolean compl = Boolean.parseBoolean(conf.get(TestNaiveBayesDriver.COMPLEMENTARY));
    if (compl) {
      classifier = new ComplementaryNaiveBayesClassifier(model);
    } else {
      classifier = new StandardNaiveBayesClassifier(model);
    }
  }

  @Override
  protected void map(Text key, VectorWritable value, Context context) throws IOException, InterruptedException {
    Vector result = classifier.classifyFull(value.get());
    //the key is the expected value
    context.write(new Text(SLASH.split(key.toString())[1]), new VectorWritable(result));
  }
}
