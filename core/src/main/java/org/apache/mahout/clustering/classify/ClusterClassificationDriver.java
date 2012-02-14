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

package org.apache.mahout.clustering.classify;

import static org.apache.mahout.clustering.classify.ClusterClassificationConfigKeys.OUTLIER_REMOVAL_THRESHOLD;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusterClassifier;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Classifies the vectors into different clusters found by the clustering algorithm.
 */
public class ClusterClassificationDriver extends AbstractJob {
	  
    /**
	   * CLI to run Cluster Classification Driver.
	   */
	  @Override
	  public int run(String[] args) throws Exception {
	    
	    addInputOption();
	    addOutputOption();
	    addOption(DefaultOptionCreator.methodOption().create());
	    addOption(DefaultOptionCreator.clustersInOption()
	            .withDescription("The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.")
	            .create());

	    if (parseArguments(args) == null) {
	      return -1;
	    }
	    
	    Path input = getInputPath();
	    Path output = getOutputPath();

	    if (getConf() == null) {
	      setConf(new Configuration());
	    }
	    Path clustersIn = new Path(getOption(DefaultOptionCreator.CLUSTERS_IN_OPTION));
	    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(
	      DefaultOptionCreator.SEQUENTIAL_METHOD);
	    
	    double clusterClassificationThreshold = 0.0;
	    if (hasOption(DefaultOptionCreator.OUTLIER_THRESHOLD)) {
	      clusterClassificationThreshold = Double.parseDouble(getOption(DefaultOptionCreator.OUTLIER_THRESHOLD));
	    }
	    
      run(input, clustersIn, output, clusterClassificationThreshold , runSequential);
      
	    return 0;
	  }
	  
	  /**
	   * Constructor to be used by the ToolRunner.
	   */
	  private ClusterClassificationDriver() {}
	  
	  public static void main(String[] args) throws Exception {
	    ToolRunner.run(new Configuration(), new ClusterClassificationDriver(), args);
	  }
	  
	  /**
	   * Uses {@link ClusterClassifier} to classify input vectors into their respective clusters.
	   * 
	   * @param input 
	   *         the input vectors
	   * @param clusteringOutputPath
	   *         the output path of clustering ( it reads clusters-*-final file from here )
	   * @param output
	   *         the location to store the classified vectors
	   * @param clusterClassificationThreshold
	   *         the threshold value of probability distribution function from 0.0 to 1.0. 
	   *         Any vector with pdf less that this threshold will not be classified for the cluster.
	   * @param runSequential
	   *         Run the process sequentially or in a mapreduce way.
	   * @throws IOException
	   * @throws InterruptedException
	   * @throws ClassNotFoundException
	   */
	  public static void run(Path input, Path clusteringOutputPath, Path output, Double clusterClassificationThreshold, boolean runSequential) throws IOException,
	                                                                        InterruptedException,
	                                                                        ClassNotFoundException {
	    if (runSequential) {
	      classifyClusterSeq(input, clusteringOutputPath, output, clusterClassificationThreshold);
	    } else {
	      Configuration conf = new Configuration();
	      classifyClusterMR(conf, input, clusteringOutputPath, output, clusterClassificationThreshold);
	    }
	    
	  }
	  
	  private static void classifyClusterSeq(Path input, Path clusters, Path output, Double clusterClassificationThreshold) throws IOException {
	    List<Cluster> clusterModels = populateClusterModels(clusters);
	    ClusterClassifier clusterClassifier = new ClusterClassifier(clusterModels);
      selectCluster(input, clusterModels, clusterClassifier, output, clusterClassificationThreshold);
      
	  }

	  /**
	   * Populates a list with clusters present in clusters-*-final directory.
	   * 
	   * @param clusterOutputPath
	   *             The output path of the clustering.
	   * @return
	   *             The list of clusters found by the clustering.
	   * @throws IOException
	   */
    private static List<Cluster> populateClusterModels(Path clusterOutputPath) throws IOException {
      List<Cluster> clusterModels = new ArrayList<Cluster>();
      Configuration conf = new Configuration();
      Cluster cluster = null;
      FileSystem fileSystem = clusterOutputPath.getFileSystem(conf);
      FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath, PathFilters.finalPartFilter());
      Iterator<?> it = new SequenceFileDirValueIterator<Writable>(clusterFiles[0].getPath(),
                                                                  PathType.LIST,
                                                                  PathFilters.partFilter(),
                                                                  null,
                                                                  false,
                                                                  conf);
      while (it.hasNext()) {
        cluster = (Cluster) it.next();
        clusterModels.add(cluster);
      }
      return clusterModels;
    }
	  
    /**
     * Classifies the vector into its respective cluster.
     * 
     * @param input 
     *            the path containing the input vector.
     * @param clusterModels
     *            the clusters
     * @param clusterClassifier
     *            used to classify the vectors into different clusters
     * @param output
     *            the path to store classified data
     * @param clusterClassificationThreshold
     * @throws IOException
     */
	  private static void selectCluster(Path input, List<Cluster> clusterModels, ClusterClassifier clusterClassifier, Path output, Double clusterClassificationThreshold) throws IOException {
	    Configuration conf = new Configuration();
	    SequenceFile.Writer writer = new SequenceFile.Writer(input.getFileSystem(conf), conf, new Path(
          output, "part-m-" + 0), IntWritable.class,
          VectorWritable.class);
	    for (VectorWritable vw : new SequenceFileDirValueIterable<VectorWritable>(
	        input, PathType.LIST, PathFilters.logsCRCFilter(), conf)) {
        Vector pdfPerCluster = clusterClassifier.classify(vw.get());
        if(shouldClassify(pdfPerCluster, clusterClassificationThreshold)) {
          int maxValueIndex = pdfPerCluster.maxValueIndex();
          Cluster cluster = clusterModels.get(maxValueIndex);
          writer.append(new IntWritable(cluster.getId()), vw);
        }
	    }
	    writer.close();
    }

	  /**
	   * Decides whether the vector should be classified or not based on the max pdf value of the clusters and threshold value.
	   * 
	   * @param pdfPerCluster
	   *         pdf of vector belonging to different clusters.
	   * @param clusterClassificationThreshold
	   *         threshold below which the vectors won't be classified.
	   * @return whether the vector should be classified or not.
	   */
    private static boolean shouldClassify(Vector pdfPerCluster, Double clusterClassificationThreshold) {
      return pdfPerCluster.maxValue() >= clusterClassificationThreshold;
    }

	  private static void classifyClusterMR(Configuration conf, Path input, Path clustersIn, Path output, Double clusterClassificationThreshold) throws IOException,
	                                                                                InterruptedException,
	                                                                                ClassNotFoundException {
	    Job job = new Job(conf, "Cluster Classification Driver running over input: " + input);
	    job.setJarByClass(ClusterClassificationDriver.class);
	    
	    conf.setFloat(OUTLIER_REMOVAL_THRESHOLD, clusterClassificationThreshold.floatValue());
	    
	    conf.set(ClusterClassificationConfigKeys.CLUSTERS_IN, input.toString());
	    
	    job.setInputFormatClass(SequenceFileInputFormat.class);
	    job.setOutputFormatClass(SequenceFileOutputFormat.class);
	    
	    job.setMapperClass(ClusterClassificationMapper.class);
	    job.setNumReduceTasks(0);
	    
	    job.setOutputKeyClass(IntWritable.class);
	    job.setOutputValueClass(WeightedVectorWritable.class);
	    
	    FileInputFormat.addInputPath(job, input);
	    FileOutputFormat.setOutputPath(job, output);
	    if (!job.waitForCompletion(true)) {
	      throw new InterruptedException("Cluster Classification Driver Job failed processing " + input);
	    }
	  }
	  
	}
