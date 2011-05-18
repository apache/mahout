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

package org.apache.mahout.clustering.spectral.eigencuts;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.spectral.common.VertexWritable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class EigencutsAffinityCutsJob {

  private static final Logger log = LoggerFactory.getLogger(EigencutsAffinityCutsJob.class);

  private EigencutsAffinityCutsJob() {
  }

  enum CUTSCOUNTER {
    NUM_CUTS
  }

  /**
   * Runs a single iteration of defining cluster boundaries, based on
   * previous calculations and the formation of the "cut matrix".
   * 
   * @param currentAffinity Path to the current affinity matrix.
   * @param cutMatrix Path to the sensitivity matrix.
   * @param nextAffinity Output path for the new affinity matrix.
   */
  public static long runjob(Path currentAffinity, Path cutMatrix, Path nextAffinity, Configuration conf)
    throws IOException, ClassNotFoundException, InterruptedException {
    
    // these options allow us to differentiate between the two vectors
    // in the mapper and reducer - we'll know from the working path
    // which SequenceFile we're accessing
    conf.set(EigencutsKeys.AFFINITY_PATH, currentAffinity.getName());
    conf.set(EigencutsKeys.CUTMATRIX_PATH, cutMatrix.getName());
    
    Job job = new Job(conf, "EigencutsAffinityCutsJob");
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(VertexWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setMapperClass(EigencutsAffinityCutsMapper.class);
    job.setCombinerClass(EigencutsAffinityCutsCombiner.class);
    job.setReducerClass(EigencutsAffinityCutsReducer.class);
    
    //FileInputFormat.addInputPath(job, currentAffinity);
    FileInputFormat.addInputPath(job, cutMatrix);
    FileOutputFormat.setOutputPath(job, nextAffinity);
    
    job.waitForCompletion(true);
    
    return job.getCounters().findCounter(CUTSCOUNTER.NUM_CUTS).getValue();
  }
  
  public static class EigencutsAffinityCutsMapper
    extends Mapper<IntWritable, VectorWritable, Text, VertexWritable> {
    
    @Override
    protected void map(IntWritable key, VectorWritable row, Context context) 
      throws IOException, InterruptedException {
      
      // all this method does is construct a bunch of vertices, mapping those
      // together which have the same *combination* of indices; for example,
      // (1, 3) will have the same key as (3, 1) but a different key from (1, 1)
      // and (3, 3) (which, incidentally, will also not be grouped together)
      String type = context.getWorkingDirectory().getName();
      Vector vector = row.get();
      for (Vector.Element e : vector) {
        String newkey = Math.max(key.get(), e.index()) + "_" + Math.min(key.get(), e.index());
        context.write(new Text(newkey), new VertexWritable(key.get(), e.index(), e.get(), type));
      }
    }
  }
  
  public static class EigencutsAffinityCutsCombiner
    extends Reducer<Text, VertexWritable, Text, VertexWritable> {
    
    @Override
    protected void reduce(Text t, Iterable<VertexWritable> vertices, 
        Context context) throws IOException, InterruptedException {
      // there should be exactly 4 items in the iterable; two from the
      // first Path source, and two from the second with matching (i, j) indices
      
      // the idea here is that we want the two vertices of the "cut" matrix,
      // and if either of them has a non-zero value, we want to:
      //
      // 1) zero out the two affinity vertices, and 
      // 2) add their former values to the (i, i) and (j, j) coordinates
      //
      // though obviously we want to perform these steps in reverse order
      Configuration conf = context.getConfiguration();
      log.debug("{}", t);
      boolean zero = false;
      int i = -1;
      int j = -1;
      double k = 0;
      int count = 0;
      for (VertexWritable v : vertices) {
        count++;
        if (v.getType().equals(conf.get(EigencutsKeys.AFFINITY_PATH))) {
          i = v.getRow();
          j = v.getCol();
          k = v.getValue();
        } else if (v.getValue() != 0.0) {
          zero = true;
        }
      }
      // if there are only two vertices, we have a diagonal
      // we want to preserve whatever is currently in the diagonal,
      // since this is acting as a running sum of all other values
      // that have been "cut" so far - simply return this element as is
      if (count == 2) {
        VertexWritable vw = new VertexWritable(i, j, k, "unimportant");
        context.write(new Text(String.valueOf(i)), vw);
        return;
      }
      
      // do we zero out the values?
      VertexWritable outI = new VertexWritable();
      VertexWritable outJ = new VertexWritable();
      if (zero) {
        // increment the cut counter
        context.getCounter(CUTSCOUNTER.NUM_CUTS).increment(1);
        
        // we want the values to exist on the diagonal
        outI.setCol(i);
        outJ.setCol(j);
        
        // also, set the old values to zero
        VertexWritable zeroI = new VertexWritable();
        VertexWritable zeroJ = new VertexWritable();
        zeroI.setCol(j);
        zeroI.setValue(0);
        zeroJ.setCol(i);
        zeroJ.setValue(0);
        zeroI.setType("unimportant");
        zeroJ.setType("unimportant");
        context.write(new Text(String.valueOf(i)), zeroI);
        context.write(new Text(String.valueOf(j)), zeroJ);
      } else {
        outI.setCol(j);
        outJ.setCol(i);
      }
      
      // set the values and write them
      outI.setValue(k);
      outJ.setValue(k);
      outI.setType("unimportant");
      outJ.setType("unimportant");
      context.write(new Text(String.valueOf(i)), outI);
      context.write(new Text(String.valueOf(j)), outJ);
    }
  }
  
  public static class EigencutsAffinityCutsReducer 
    extends Reducer<Text, VertexWritable, IntWritable, VectorWritable> {
    
    @Override
    protected void reduce(Text row, Iterable<VertexWritable> entries, 
        Context context) throws IOException, InterruptedException {
      // now to assemble the vectors
      RandomAccessSparseVector output = new RandomAccessSparseVector(
          context.getConfiguration().getInt(EigencutsKeys.AFFINITY_DIMENSIONS, Integer.MAX_VALUE), 100);
      int rownum = Integer.parseInt(row.toString());
      for (VertexWritable e : entries) {
        // first, are we setting a diagonal?
        if (e.getCol() == rownum) {
          // add to what's already present
          output.setQuick(e.getCol(), output.getQuick(e.getCol()) + e.getValue());
        } else {
          // simply set the value
          output.setQuick(e.getCol(), e.getValue());
        }
      }
      context.write(new IntWritable(rownum), new VectorWritable(output));
    }
  }
}
