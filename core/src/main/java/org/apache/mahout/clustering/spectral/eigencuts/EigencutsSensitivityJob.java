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
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.spectral.common.VectorCache;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * <p>There are a quite a few operations bundled within this mapper. Gather 'round
 * and listen, all of ye.</p>
 * 
 * <p>The input to this job is eight items:</p>
 * <ol><li>B<sub>0</sub>, which is a command-line parameter fed through the Configuration object</li>
 * <li>diagonal matrix, a constant vector fed through the Hadoop cache</li>
 * <li>list of eigenvalues, a constant vector fed through the Hadoop cache</li>
 * <li>eigenvector, the input value to the mapper</li>
 * <li>epsilon</li>
 * <li>delta</li>
 * <li>tau</li>
 * <li>output, the Path to the output matrix of sensitivities</li></ol>
 * 
 * <p>The first three items are constant and are used in all of the map
 * tasks. The row index indicates which eigenvalue from the list to use, and
 * also serves as the output identifier. The diagonal matrix and the 
 * eigenvector are both of equal length and are iterated through twice
 * within each map task, unfortunately lending each task to a runtime of 
 * n<sup>2</sup>. This is unavoidable.</p>
 * 
 * <p>For each (i, j) combination of elements within the eigenvector, a complex
 * equation is run that explicitly computes the sensitivity to perturbation of 
 * the flow of probability within the specific edge of the graph. Each
 * sensitivity, as it is computed, is simultaneously applied to a non-maximal
 * suppression step: for a given sensitivity S_ij, it must be suppressed if
 * any other S_in or S_mj has a more negative value. Thus, only the most
 * negative S_ij within its row i or its column j is stored in the return
 * array, leading to an output (per eigenvector!) with maximum length n, 
 * minimum length 1.</p>
 * 
 * <p>Overall, this creates an n-by-n (possibly sparse) matrix with a maximum
 * of n^2 non-zero elements, minimum of n non-zero elements.</p>
 */
public final class EigencutsSensitivityJob {

  private EigencutsSensitivityJob() {
  }

  /**
   * Initializes the configuration tasks, loads the needed data into
   * the HDFS cache, and executes the job.
   * 
   * @param eigenvalues Vector of eigenvalues
   * @param diagonal Vector representing the diagonal matrix
   * @param eigenvectors Path to the DRM of eigenvectors
   * @param output Path to the output matrix (will have between n and full-rank
   *                non-zero elements)
   */
  public static void runJob(Vector eigenvalues,
                            Vector diagonal,
                            Path eigenvectors,
                            double beta,
                            double tau,
                            double delta,
                            double epsilon,
                            Path output)
    throws IOException, ClassNotFoundException, InterruptedException {
    
    // save the two vectors to the distributed cache
    Configuration jobConfig = new Configuration();
    Path eigenOutputPath = new Path(output.getParent(), "eigenvalues");
    Path diagOutputPath = new Path(output.getParent(), "diagonal");
    jobConfig.set(EigencutsKeys.VECTOR_CACHE_BASE, output.getParent().getName());
    VectorCache.save(new IntWritable(EigencutsKeys.EIGENVALUES_CACHE_INDEX), 
        eigenvalues, eigenOutputPath, jobConfig);
    VectorCache.save(new IntWritable(EigencutsKeys.DIAGONAL_CACHE_INDEX), 
        diagonal, diagOutputPath, jobConfig);
    
    // set up the rest of the job
    jobConfig.set(EigencutsKeys.BETA, Double.toString(beta));
    jobConfig.set(EigencutsKeys.EPSILON, Double.toString(epsilon));
    jobConfig.set(EigencutsKeys.DELTA, Double.toString(delta));
    jobConfig.set(EigencutsKeys.TAU, Double.toString(tau));
    
    Job job = new Job(jobConfig, "EigencutsSensitivityJob");
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(EigencutsSensitivityNode.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(EigencutsSensitivityMapper.class);
    job.setReducerClass(EigencutsSensitivityReducer.class);
    
    FileInputFormat.addInputPath(job, eigenvectors);
    FileOutputFormat.setOutputPath(job, output);
    
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }  
}
