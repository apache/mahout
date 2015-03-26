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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * <p>Runs a Map/Reduce job that encodes {@link org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder} the
 * input and writes it to the output as a sequence file.</p>
 *
 * <p>Only works on basic text, where the value in the SequenceFile is a blob of text.</p>
 */
//TODO: find commonalities w/ DictionaryVectorizer and abstract them out
public class SimpleTextEncodingVectorizer implements Vectorizer {

  private static final Logger log = LoggerFactory.getLogger(SimpleTextEncodingVectorizer.class);

  @Override
  public void createVectors(Path input, Path output, VectorizerConfig config)
    throws IOException, ClassNotFoundException, InterruptedException {
    //do this for convenience of using prepareJob
    Job job = HadoopUtil.prepareJob(input, output,
                                    SequenceFileInputFormat.class,
                                    EncodingMapper.class,
                                    Text.class,
                                    VectorWritable.class,
                                    SequenceFileOutputFormat.class,
                                    config.getConf());
    Configuration conf = job.getConfiguration();
    conf.set(EncodingMapper.USE_SEQUENTIAL, String.valueOf(config.isSequentialAccess()));
    conf.set(EncodingMapper.USE_NAMED_VECTORS, String.valueOf(config.isNamedVectors()));
    conf.set(EncodingMapper.ANALYZER_NAME, config.getAnalyzerClassName());
    conf.set(EncodingMapper.ENCODER_FIELD_NAME, config.getEncoderName());
    conf.set(EncodingMapper.ENCODER_CLASS, config.getEncoderClass());
    conf.set(EncodingMapper.CARDINALITY, String.valueOf(config.getCardinality()));
    job.setNumReduceTasks(0);
    boolean finished = job.waitForCompletion(true);

    log.info("result of run: {}", finished);
    if (!finished) {
      throw new IllegalStateException("Job failed!");
    }
  }

}

