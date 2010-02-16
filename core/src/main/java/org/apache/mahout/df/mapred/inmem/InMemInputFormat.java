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

package org.apache.mahout.df.mapred.inmem;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.mapred.Builder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Custom InputFormat that generates InputSplits given the desired number of
 * trees.<br>
 * each input split contains a subset of the trees.<br>
 * The number of splits is equal to the number of requested splits
 */
public class InMemInputFormat implements InputFormat<IntWritable, NullWritable> {
  
  private static final Logger log = LoggerFactory.getLogger(InMemInputSplit.class);
  
  private Random rng;
  
  private Long seed;
  
  private boolean isSingleSeed;
  
  /**
   * Used for DEBUG purposes only. if true and a seed is available, all the
   * mappers use the same seed, thus all the mapper should take the same time to
   * build their trees.
   * 
   * @param conf
   * @return
   */
  private static boolean isSingleSeed(Configuration conf) {
    return conf.getBoolean("debug.mahout.rf.single.seed", false);
  }
  
  @Override
  public RecordReader<IntWritable, NullWritable> getRecordReader(
    InputSplit split, JobConf conf, Reporter reporter) throws IOException {
    return new InMemRecordReader((InMemInputSplit) split);
  }
  
  @Override
  public InputSplit[] getSplits(JobConf conf, int numSplits) throws IOException {
    int nbTrees = Builder.getNbTrees(conf);
    int splitSize = nbTrees / numSplits;
    
    seed = Builder.getRandomSeed(conf);
    isSingleSeed = isSingleSeed(conf);
    
    if (rng != null && seed != null) {
      log.warn("getSplits() was called more than once and the 'seed' is set, "
        + "this can lead to no-repeatable behavior");
    }
    
    rng = seed == null || isSingleSeed ? null : RandomUtils.getRandom();
    
    int id = 0;
    
    InputSplit[] splits = new InputSplit[numSplits];
    
    for (int index = 0; index < numSplits - 1; index++) {
      splits[index] = new InMemInputSplit(id, splitSize, nextSeed());
      id += splitSize;
    }
    
    // take care of the remainder
    splits[numSplits - 1] = new InMemInputSplit(id, nbTrees - id, nextSeed());
    
    return splits;
  }
  
  /**
   * Return the seed for the next InputSplit
   * 
   * @return
   */
  private Long nextSeed() {
    if (seed == null) {
      return null;
    } else if (isSingleSeed) {
      return seed;
    } else {
      return rng.nextLong();
    }
  }
  
  public static class InMemRecordReader implements
  RecordReader<IntWritable, NullWritable> {
    
    private final InMemInputSplit split;
    
    private int pos;
    
    public InMemRecordReader(InMemInputSplit split) {
      this.split = split;
    }
    
    @Override
    public void close() throws IOException {
    }
    
    @Override
    public IntWritable createKey() {
      return new IntWritable();
    }
    
    @Override
    public NullWritable createValue() {
      return NullWritable.get();
    }
    
    @Override
    public long getPos() throws IOException {
      return pos;
    }
    
    @Override
    public float getProgress() throws IOException {
      if (pos == 0) {
        return 0.0f;
      } else {
        return (float) (pos - 1) / split.nbTrees;
      }
    }
    
    @Override
    public boolean next(IntWritable key, NullWritable value) throws IOException {
      if (pos < split.nbTrees) {
        key.set(split.firstId + pos);
        pos++;
        return true;
      } else {
        return false;
      }
    }
    
  }
  
  /**
   * Custom InputSplit that indicates how many trees are built by each mapper
   */
  public static class InMemInputSplit implements InputSplit {
    
    private static final String[] NO_LOCATIONS = new String[0];
    
    /** Id of the first tree of this split */
    private int firstId;
    
    private int nbTrees;
    
    private Long seed;
    
    public InMemInputSplit() {
    }
    
    public InMemInputSplit(int firstId, int nbTrees, Long seed) {
      this.firstId = firstId;
      this.nbTrees = nbTrees;
      this.seed = seed;
    }
    
    /**
     * Return the Id of the first tree of this split
     * 
     * @return
     */
    public int getFirstId() {
      return firstId;
    }
    
    /**
     * Return the number of trees
     * 
     * @return
     */
    public int getNbTrees() {
      return nbTrees;
    }
    
    /**
     * Return the random seed
     * 
     * @return null if no seed is available
     */
    public Long getSeed() {
      return seed;
    }
    
    @Override
    public long getLength() throws IOException {
      return nbTrees;
    }
    
    @Override
    public String[] getLocations() throws IOException {
      return NO_LOCATIONS;
    }
    
    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null || !(obj instanceof InMemInputSplit)) {
        return false;
      }
      
      InMemInputSplit split = (InMemInputSplit) obj;
      
      if (seed == null && split.seed != null) {
        return false;
      }
      
      return firstId == split.firstId && nbTrees == split.nbTrees
      && (seed == null || seed.equals(split.seed));
    }
    
    @Override
    public int hashCode() {
      return firstId + nbTrees + (seed == null ? 0 : seed.intValue());
    }
    
    @Override
    public String toString() {
      return String.format("[firstId:%d, nbTrees:%d, seed:%d]", firstId,
        nbTrees, seed);
    }
    
    @Override
    public void readFields(DataInput in) throws IOException {
      firstId = in.readInt();
      nbTrees = in.readInt();
      boolean isSeed = in.readBoolean();
      seed = isSeed ? in.readLong() : null;
    }
    
    @Override
    public void write(DataOutput out) throws IOException {
      out.writeInt(firstId);
      out.writeInt(nbTrees);
      out.writeBoolean(seed != null);
      if (seed != null) {
        out.writeLong(seed);
      }
    }
    
    public static InMemInputSplit read(DataInput in) throws IOException {
      InMemInputSplit split = new InMemInputSplit();
      split.readFields(in);
      return split;
    }
  }
  
}
