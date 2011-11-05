package org.apache.mahout.vectorizer;
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


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs a Map/Reduce job that encodes {@link org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder} the
 * input and writes it to the output as a sequence file.
 *<p/>
 * Only works on basic text, where the value in the SequenceFile is a blob of text.
 */
//TODO: find commonalities w/ DictionaryVectorizer and abstract them out
public class SimpleTextEncodingVectorizer implements Vectorizer {
  private transient static Logger log = LoggerFactory.getLogger(SimpleTextEncodingVectorizer.class);

  public SimpleTextEncodingVectorizer() {
  }


  @Override
  public void createVectors(final Path input, final Path output, final VectorizerConfig config) throws Exception {
    //do this for convenience of using prepareJob
    Job job = HadoopUtil.prepareJob(input, output, SequenceFileInputFormat.class, EncodingMapper.class, Text.class, VectorWritable.class,
            SequenceFileOutputFormat.class, config.conf);
    Configuration conf = job.getConfiguration();
    conf.set(EncodingMapper.USE_SEQUENTIAL, String.valueOf(config.sequentialAccess));
    conf.set(EncodingMapper.USE_NAMED_VECTORS, String.valueOf(config.namedVectors));
    conf.set(EncodingMapper.ANALYZER_NAME, config.analyzerClassName);
    conf.set(EncodingMapper.ENCODER_FIELD_NAME, config.encoderName);
    conf.set(EncodingMapper.ENCODER_CLASS, config.encoderClass);
    conf.set(EncodingMapper.CARDINALITY, String.valueOf(config.cardinality));
    job.setNumReduceTasks(0);
    boolean finished = job.waitForCompletion(true);

    log.info("result of run: " + finished);
    //TODO: something useful w/ this result should it be meaningful.
  }


}

