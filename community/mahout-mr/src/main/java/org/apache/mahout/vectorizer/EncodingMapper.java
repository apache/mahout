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
package org.apache.mahout.vectorizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.lucene.AnalyzerUtils;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.LuceneTextValueEncoder;

import java.io.IOException;

/**
 * The Mapper that does the work of encoding text
 */
public class EncodingMapper extends Mapper<Text, Text, Text, VectorWritable> {

  public static final String USE_NAMED_VECTORS = "namedVectors";
  public static final String USE_SEQUENTIAL = "sequential";
  public static final String ANALYZER_NAME = "analyzer";
  public static final String ENCODER_FIELD_NAME = "encoderFieldName";
  public static final String ENCODER_CLASS = "encoderClass";
  public static final String CARDINALITY = "cardinality";
  private boolean sequentialVectors;
  private boolean namedVectors;
  private FeatureVectorEncoder encoder;
  private int cardinality;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    sequentialVectors = conf.getBoolean(USE_SEQUENTIAL, false);
    namedVectors = conf.getBoolean(USE_NAMED_VECTORS, false);
    String analyzerName = conf.get(ANALYZER_NAME, StandardAnalyzer.class.getName());
    Analyzer analyzer;
    try {
      analyzer = AnalyzerUtils.createAnalyzer(analyzerName);
    } catch (ClassNotFoundException e) {
      //TODO: hmmm, don't like this approach
      throw new IOException("Unable to create Analyzer for name: " + analyzerName, e);
    }

    String encoderName = conf.get(ENCODER_FIELD_NAME, "text");
    cardinality = conf.getInt(CARDINALITY, 5000);
    String encClass = conf.get(ENCODER_CLASS);
    encoder = ClassUtils.instantiateAs(encClass,
            FeatureVectorEncoder.class,
            new Class[]{String.class},
            new Object[]{encoderName});
    if (encoder instanceof LuceneTextValueEncoder) {
      ((LuceneTextValueEncoder) encoder).setAnalyzer(analyzer);
    }
  }

  @Override
  protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {
    Vector vector;
    if (sequentialVectors) {
      vector = new SequentialAccessSparseVector(cardinality);
    } else {
      vector = new RandomAccessSparseVector(cardinality);
    }
    if (namedVectors) {
      vector = new NamedVector(vector, key.toString());
    }
    encoder.addToVector(value.toString(), vector);
    context.write(new Text(key.toString()), new VectorWritable(vector));
  }
}
