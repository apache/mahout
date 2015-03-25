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

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import com.google.common.base.Preconditions;

import java.io.IOException;
import java.net.URI;

public final class TimesSquaredJob {

  public static final String INPUT_VECTOR = "DistributedMatrix.times.inputVector";
  public static final String IS_SPARSE_OUTPUT = "DistributedMatrix.times.outputVector.sparse";
  public static final String OUTPUT_VECTOR_DIMENSION = "DistributedMatrix.times.output.dimension";

  public static final String OUTPUT_VECTOR_FILENAME = "DistributedMatrix.times.outputVector";

  private TimesSquaredJob() { }

  public static Job createTimesSquaredJob(Vector v, Path matrixInputPath, Path outputVectorPath)
    throws IOException {
    return createTimesSquaredJob(new Configuration(), v, matrixInputPath, outputVectorPath);
  }
  
  public static Job createTimesSquaredJob(Configuration initialConf, Vector v, Path matrixInputPath,
                                          Path outputVectorPath) throws IOException {

    return createTimesSquaredJob(initialConf, v, matrixInputPath, outputVectorPath, TimesSquaredMapper.class,
                                 VectorSummingReducer.class);
  }

  public static Job createTimesJob(Vector v, int outDim, Path matrixInputPath, Path outputVectorPath)
    throws IOException {

    return createTimesJob(new Configuration(), v, outDim, matrixInputPath, outputVectorPath);
  }
    
  public static Job createTimesJob(Configuration initialConf, Vector v, int outDim, Path matrixInputPath,
                                   Path outputVectorPath) throws IOException {

    return createTimesSquaredJob(initialConf, v, outDim, matrixInputPath, outputVectorPath, TimesMapper.class,
                                 VectorSummingReducer.class);
  }

  public static Job createTimesSquaredJob(Vector v, Path matrixInputPath, Path outputVectorPathBase,
      Class<? extends TimesSquaredMapper> mapClass, Class<? extends VectorSummingReducer> redClass) throws IOException {

    return createTimesSquaredJob(new Configuration(), v, matrixInputPath, outputVectorPathBase, mapClass, redClass);
  }
  
  public static Job createTimesSquaredJob(Configuration initialConf, Vector v, Path matrixInputPath,
      Path outputVectorPathBase, Class<? extends TimesSquaredMapper> mapClass,
      Class<? extends VectorSummingReducer> redClass) throws IOException {

    return createTimesSquaredJob(initialConf, v, v.size(), matrixInputPath, outputVectorPathBase, mapClass, redClass);
  }

  public static Job createTimesSquaredJob(Vector v, int outputVectorDim, Path matrixInputPath,
      Path outputVectorPathBase, Class<? extends TimesSquaredMapper> mapClass,
      Class<? extends VectorSummingReducer> redClass) throws IOException {

    return createTimesSquaredJob(new Configuration(), v, outputVectorDim, matrixInputPath, outputVectorPathBase,
        mapClass, redClass);
  }
  
  public static Job createTimesSquaredJob(Configuration initialConf, Vector v, int outputVectorDim,
      Path matrixInputPath, Path outputVectorPathBase, Class<? extends TimesSquaredMapper> mapClass,
      Class<? extends VectorSummingReducer> redClass) throws IOException {

    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), initialConf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    outputVectorPathBase = fs.makeQualified(outputVectorPathBase);

    long now = System.nanoTime();
    Path inputVectorPath = new Path(outputVectorPathBase, INPUT_VECTOR + '/' + now);


    SequenceFile.Writer inputVectorPathWriter = null;

    try {
      inputVectorPathWriter = new SequenceFile.Writer(fs, initialConf, inputVectorPath, NullWritable.class,
                                                      VectorWritable.class);
      inputVectorPathWriter.append(NullWritable.get(), new VectorWritable(v));
    } finally {
      Closeables.close(inputVectorPathWriter, false);
    }

    URI ivpURI = inputVectorPath.toUri();
    DistributedCache.setCacheFiles(new URI[] { ivpURI }, initialConf);

    Job job = HadoopUtil.prepareJob(matrixInputPath, new Path(outputVectorPathBase, OUTPUT_VECTOR_FILENAME),
        SequenceFileInputFormat.class, mapClass, NullWritable.class, VectorWritable.class, redClass,
        NullWritable.class, VectorWritable.class, SequenceFileOutputFormat.class, initialConf);
    job.setCombinerClass(redClass);
    job.setJobName("TimesSquaredJob: " + matrixInputPath);

    Configuration conf = job.getConfiguration();
    conf.set(INPUT_VECTOR, ivpURI.toString());
    conf.setBoolean(IS_SPARSE_OUTPUT, !v.isDense());
    conf.setInt(OUTPUT_VECTOR_DIMENSION, outputVectorDim);

    return job;
  }

  public static Vector retrieveTimesSquaredOutputVector(Path outputVectorTmpPath, Configuration conf)
    throws IOException {
    Path outputFile = new Path(outputVectorTmpPath, OUTPUT_VECTOR_FILENAME + "/part-r-00000");
    SequenceFileValueIterator<VectorWritable> iterator =
        new SequenceFileValueIterator<>(outputFile, true, conf);
    try {
      return iterator.next().get();
    } finally {
      Closeables.close(iterator, true);
    }
  }

  public static class TimesSquaredMapper<T extends WritableComparable>
      extends Mapper<T,VectorWritable, NullWritable,VectorWritable> {

    private Vector outputVector;
    private Vector inputVector;

    Vector getOutputVector() {
      return outputVector;
    }

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      try {
        Configuration conf = ctx.getConfiguration();
        Path[] localFiles = DistributedCache.getLocalCacheFiles(conf);
        Preconditions.checkArgument(localFiles != null && localFiles.length >= 1,
            "missing paths from the DistributedCache");

        Path inputVectorPath = HadoopUtil.getSingleCachedFile(conf);

        SequenceFileValueIterator<VectorWritable> iterator =
            new SequenceFileValueIterator<>(inputVectorPath, true, conf);
        try {
          inputVector = iterator.next().get();
        } finally {
          Closeables.close(iterator, true);
        }

        int outDim = conf.getInt(OUTPUT_VECTOR_DIMENSION, Integer.MAX_VALUE);
        outputVector = conf.getBoolean(IS_SPARSE_OUTPUT, false)
            ? new RandomAccessSparseVector(outDim, 10)
            : new DenseVector(outDim);
      } catch (IOException ioe) {
        throw new IllegalStateException(ioe);
      }
    }

    @Override
    protected void map(T key, VectorWritable v, Context context) throws IOException, InterruptedException {

      double d = scale(v);
      if (d == 1.0) {
        outputVector.assign(v.get(), Functions.PLUS);
      } else if (d != 0.0) {
        outputVector.assign(v.get(), Functions.plusMult(d));
      }
    }

    protected double scale(VectorWritable v) {
      return v.get().dot(inputVector);
    }

    @Override
    protected void cleanup(Context ctx) throws IOException, InterruptedException {
      ctx.write(NullWritable.get(), new VectorWritable(outputVector));
    }

  }

  public static class TimesMapper extends TimesSquaredMapper<IntWritable> {


    @Override
    protected void map(IntWritable rowNum, VectorWritable v, Context context) throws IOException, InterruptedException {
      double d = scale(v);
      if (d != 0.0) {
        getOutputVector().setQuick(rowNum.get(), d);
      }
    }
  }

  public static class VectorSummingReducer extends Reducer<NullWritable,VectorWritable,NullWritable,VectorWritable> {

    private Vector outputVector;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Configuration conf = ctx.getConfiguration();
      int outputDimension = conf.getInt(OUTPUT_VECTOR_DIMENSION, Integer.MAX_VALUE);
      outputVector = conf.getBoolean(IS_SPARSE_OUTPUT, false)
                   ? new RandomAccessSparseVector(outputDimension, 10)
                   : new DenseVector(outputDimension);
    }

    @Override
    protected void reduce(NullWritable key, Iterable<VectorWritable> vectors, Context ctx)
      throws IOException, InterruptedException {

      for (VectorWritable v : vectors) {
        if (v != null) {
          outputVector.assign(v.get(), Functions.PLUS);
        }
      }
      ctx.write(NullWritable.get(), new VectorWritable(outputVector));
    }
  }

}
