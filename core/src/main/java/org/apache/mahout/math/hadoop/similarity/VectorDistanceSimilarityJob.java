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

package org.apache.mahout.math.hadoop.similarity;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.VectorWritable;

import com.google.common.base.Preconditions;

import java.io.IOException;

/**
 * This class does a Map-side join between seed vectors (the map side can also be a Cluster) and a list of other vectors
 * and emits the a tuple of seed id, other id, distance.  It is a more generic version of KMean's mapper
 */
public class VectorDistanceSimilarityJob extends AbstractJob {

  public static final String SEEDS = "seeds";
  public static final String SEEDS_PATH_KEY = "seedsPath";
  public static final String DISTANCE_MEASURE_KEY = "vectorDistSim.measure";
  public static final String OUT_TYPE_KEY = "outType";
  public static final String MAX_DISTANCE = "maxDistance";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new VectorDistanceSimilarityJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(SEEDS, "s", "The set of vectors to compute distances against.  Must fit in memory on the mapper");
    addOption(MAX_DISTANCE, "mx", "set an upper-bound on distance (double) such that any pair of vectors with a"
        + " distance greater than this value is ignored in the output. Ignored for non pairwise output!");
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(OUT_TYPE_KEY, "ot", "[pw|v] -- Define the output style: pairwise, the default, (pw) or vector (v).  "
        + "Pairwise is a tuple of <seed, other, distance>, vector is <other, <Vector of size the number of seeds>>.",
        "pw");

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    Path seeds = new Path(getOption(SEEDS));
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = SquaredEuclideanDistanceMeasure.class.getName();
    }
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
    String outType = getOption(OUT_TYPE_KEY, "pw");
    
    Double maxDistance = null;

    if ("pw".equals(outType)) {
      String maxDistanceArg = getOption(MAX_DISTANCE);
      if (maxDistanceArg != null) {
        maxDistance = Double.parseDouble(maxDistanceArg);
        Preconditions.checkArgument(maxDistance > 0.0d, "value for " + MAX_DISTANCE + " must be greater than zero");
      }
    }

    run(getConf(), input, seeds, output, measure, outType, maxDistance);
    return 0;
  }

  public static void run(Configuration conf,
                         Path input,
                         Path seeds,
                         Path output,
                         DistanceMeasure measure, String outType)
    throws IOException, ClassNotFoundException, InterruptedException {
    run(conf, input, seeds, output, measure, outType, null);
  }      
  
  public static void run(Configuration conf,
          Path input,
          Path seeds,
          Path output,
          DistanceMeasure measure, String outType, Double maxDistance)
    throws IOException, ClassNotFoundException, InterruptedException {
    if (maxDistance != null) {
      conf.set(MAX_DISTANCE, String.valueOf(maxDistance));
    }
    conf.set(DISTANCE_MEASURE_KEY, measure.getClass().getName());
    conf.set(SEEDS_PATH_KEY, seeds.toString());
    Job job = new Job(conf, "Vector Distance Similarity: seeds: " + seeds + " input: " + input);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    if ("pw".equalsIgnoreCase(outType)) {
      job.setMapOutputKeyClass(StringTuple.class);
      job.setOutputKeyClass(StringTuple.class);
      job.setMapOutputValueClass(DoubleWritable.class);
      job.setOutputValueClass(DoubleWritable.class);
      job.setMapperClass(VectorDistanceMapper.class);
    } else if ("v".equalsIgnoreCase(outType)) {
      job.setMapOutputKeyClass(Text.class);
      job.setOutputKeyClass(Text.class);
      job.setMapOutputValueClass(VectorWritable.class);
      job.setOutputValueClass(VectorWritable.class);
      job.setMapperClass(VectorDistanceInvertedMapper.class);
    } else {
      throw new IllegalArgumentException("Invalid outType specified: " + outType);
    }

    job.setNumReduceTasks(0);
    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setJarByClass(VectorDistanceSimilarityJob.class);
    HadoopUtil.delete(conf, output);
    if (!job.waitForCompletion(true)) {
      throw new IllegalStateException("VectorDistance Similarity failed processing " + seeds);
    }
  }
}
