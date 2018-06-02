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

package org.apache.mahout.vectorizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.LuceneTextValueEncoder;

/**
 * Converts a given set of sequence files into SparseVectors
 */
public final class EncodedVectorsFromSequenceFiles extends AbstractJob {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new EncodedVectorsFromSequenceFiles(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.analyzerOption().create());
    addOption(buildOption("sequentialAccessVector", "seq",
                          "(Optional) Whether output vectors should be SequentialAccessVectors. "
                              + "If set true else false",
                          false, false, null));
    addOption(buildOption("namedVector", "nv",
                          "Create named vectors using the key.  False by default", false, false, null));
    addOption("cardinality", "c",
              "The cardinality to use for creating the vectors.  Default is 5000", "5000");
    addOption("encoderFieldName", "en",
              "The name of the encoder to be passed to the FeatureVectorEncoder constructor. Default is text. "
                  + "Note this is not the class name of a FeatureValueEncoder, but is instead the construction "
                  + "argument.",
              "text");
    addOption("encoderClass", "ec",
              "The class name of the encoder to be used. Default is " + LuceneTextValueEncoder.class.getName(),
              LuceneTextValueEncoder.class.getName());
    addOption(DefaultOptionCreator.overwriteOption().create());
    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();

    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }

    Class<? extends Analyzer> analyzerClass = getAnalyzerClassFromOption();

    Configuration conf = getConf();

    boolean sequentialAccessOutput = hasOption("sequentialAccessVector");

    boolean namedVectors = hasOption("namedVector");
    int cardinality = 5000;
    if (hasOption("cardinality")) {
      cardinality = Integer.parseInt(getOption("cardinality"));
    }
    String encoderName = "text";
    if (hasOption("encoderFieldName")) {
      encoderName = getOption("encoderFieldName");
    }
    String encoderClass = LuceneTextValueEncoder.class.getName();
    if (hasOption("encoderClass")) {
      encoderClass = getOption("encoderClass");
      ClassUtils.instantiateAs(encoderClass, FeatureVectorEncoder.class, new Class[] { String.class },
          new Object[] { encoderName }); //try instantiating it
    }

    SimpleTextEncodingVectorizer vectorizer = new SimpleTextEncodingVectorizer();
    VectorizerConfig config = new VectorizerConfig(conf, analyzerClass.getName(), encoderClass, encoderName,
        sequentialAccessOutput, namedVectors, cardinality);

    vectorizer.createVectors(input, output, config);

    return 0;
  }

}
