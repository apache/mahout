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

package org.apache.mahout.math.hadoop;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.Iterator;

public class TimesSquaredJob {

  private static final Logger log = LoggerFactory.getLogger(TimesSquaredJob.class);

  public static final String INPUT_VECTOR = "timesSquared.inputVector";
  public static final String IS_SPARSE_OUTPUT = "timesSquared.outputVector.sparse";
  public static final String OUTPUT_VECTOR_DIMENSION = "timesSquared.output.dimension";

  public static final String OUTPUT_VECTOR_FILENAME = "timesSquaredOutputVector";

  private TimesSquaredJob() {}

  public static JobConf createTimesSquaredJobConf(Vector v, 
                                                  Path matrixInputPath, 
                                                  Path outputVectorPath) throws IOException {
    return createTimesSquaredJobConf(v,
                                     matrixInputPath,
                                     outputVectorPath,
                                     TimesSquaredMapper.class,
                                     VectorSummingReducer.class);
  }

  public static JobConf createTimesSquaredJobConf(Vector v,
                                                  Path matrixInputPath,
                                                  Path outputVectorPathBase,
                                                  Class<? extends TimesSquaredMapper> mapClass,
                                                  Class<? extends VectorSummingReducer> redClass) throws IOException {
    JobConf conf = new JobConf(TimesSquaredJob.class);
    conf.setJobName("TimesSquaredJob: " + matrixInputPath + " timesSquared(" + v.getName() + ')');
    FileSystem fs = FileSystem.get(conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    outputVectorPathBase = fs.makeQualified(outputVectorPathBase);

    long now = System.nanoTime();
    Path inputVectorPath = new Path(outputVectorPathBase, INPUT_VECTOR + '/' + now);
    SequenceFile.Writer inputVectorPathWriter = new SequenceFile.Writer(fs,
            conf, inputVectorPath, NullWritable.class, VectorWritable.class);
    VectorWritable inputVW = new VectorWritable(v);
    inputVectorPathWriter.append(NullWritable.get(), inputVW);
    inputVectorPathWriter.close();
    URI ivpURI = inputVectorPath.toUri();
    DistributedCache.setCacheFiles(new URI[] {ivpURI}, conf);
    fs.deleteOnExit(inputVectorPath);

    conf.set(INPUT_VECTOR, ivpURI.toString());
    conf.setBoolean(IS_SPARSE_OUTPUT, !(v instanceof DenseVector));
    conf.setInt(OUTPUT_VECTOR_DIMENSION, v.size());
    FileInputFormat.addInputPath(conf, matrixInputPath);
    conf.setInputFormat(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(conf, new Path(outputVectorPathBase, OUTPUT_VECTOR_FILENAME));
    conf.setMapperClass(mapClass);
    conf.setMapOutputKeyClass(NullWritable.class);
    conf.setMapOutputValueClass(VectorWritable.class);
    conf.setReducerClass(redClass);
    conf.setCombinerClass(redClass);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.setOutputKeyClass(NullWritable.class);
    conf.setOutputValueClass(VectorWritable.class);
    return conf;
  }

  public static Vector retrieveTimesSquaredOutputVector(JobConf conf) throws IOException {
    Path outputPath = FileOutputFormat.getOutputPath(conf);
    FileSystem fs = FileSystem.get(conf);
    Path outputFile = new Path(outputPath, "part-00000");
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, outputFile, conf);
    NullWritable n = NullWritable.get();
    VectorWritable v = new VectorWritable();
    reader.next(n,v);
    Vector vector = v.get();
    reader.close();
    fs.deleteOnExit(outputFile);
    return vector;
  }

  public static class TimesSquaredMapper extends MapReduceBase
      implements Mapper<WritableComparable<?>,VectorWritable, NullWritable,VectorWritable> {

    private Vector inputVector;
    private Vector outputVector;
    private OutputCollector<NullWritable,VectorWritable> out;

    @Override
    public void configure(JobConf conf) {
      try {
        URI[] localFiles = DistributedCache.getCacheFiles(conf);
        if (localFiles == null || localFiles.length < 1) {
          throw new IllegalArgumentException(
            "missing paths from the DistributedCache");
        }
        Path inputVectorPath = new Path(localFiles[0].getPath());
        FileSystem fs = inputVectorPath.getFileSystem(conf);

        SequenceFile.Reader reader = new SequenceFile.Reader(fs,
          inputVectorPath,
          conf);
        VectorWritable val = new VectorWritable();
        NullWritable nw = NullWritable.get();
        reader.next(nw, val);
        reader.close();
        inputVector = val.get();
        if(!(inputVector instanceof SequentialAccessSparseVector || inputVector instanceof DenseVector)) {
          inputVector = new SequentialAccessSparseVector(inputVector);
        }
        outputVector = conf.getBoolean(IS_SPARSE_OUTPUT, false)
                     ? new RandomAccessSparseVector(inputVector.size(), 10)
                     : new DenseVector(inputVector.size());
      } catch (IOException ioe) {
        throw new IllegalStateException(ioe);
      }
    }

    @Override
    public void map(WritableComparable<?> rowNum,
                    VectorWritable v,
                    OutputCollector<NullWritable,VectorWritable> out,
                    Reporter rep) throws IOException {
      this.out = out;
      double d = scale(v);
      if (d == 1.0) {
        outputVector.assign(v.get(), Functions.plus);
      } else if (d != 0.0) {
        outputVector.assign(v.get(), Functions.plusMult(d));
      }
    }

    protected double scale(VectorWritable v) {
      return v.get().dot(inputVector);
    }

    @Override
    public void close() throws IOException {
      out.collect(NullWritable.get(), new VectorWritable(outputVector));
    }

  }

  public static class VectorSummingReducer extends MapReduceBase
      implements Reducer<NullWritable,VectorWritable,NullWritable,VectorWritable> {

    private Vector outputVector;

    @Override
    public void configure(JobConf conf) {
      int outputDimension = conf.getInt(OUTPUT_VECTOR_DIMENSION, Integer.MAX_VALUE);
      outputVector = conf.getBoolean(IS_SPARSE_OUTPUT, false)
                   ? new RandomAccessSparseVector(outputDimension, 10)
                   : new DenseVector(outputDimension);
    }

    @Override
    public void reduce(NullWritable n,
                       Iterator<VectorWritable> vectors,
                       OutputCollector<NullWritable,VectorWritable> out,
                       Reporter reporter) throws IOException {
      while(vectors.hasNext()) {
        VectorWritable v = vectors.next();
        if(v != null) {
          v.get().addTo(outputVector);
        }
      }
      out.collect(NullWritable.get(), new VectorWritable(outputVector));
    }
  }

}
