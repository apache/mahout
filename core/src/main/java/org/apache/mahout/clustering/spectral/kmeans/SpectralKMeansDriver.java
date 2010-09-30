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

package org.apache.mahout.clustering.spectral.kmeans;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.clustering.spectral.common.AffinityMatrixInputJob;
import org.apache.mahout.clustering.spectral.common.MatrixDiagonalizeJob;
import org.apache.mahout.clustering.spectral.common.UnitVectorizerJob;
import org.apache.mahout.clustering.spectral.common.VectorMatrixMultiplicationJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.decomposer.DistributedLanczosSolver;
import org.apache.mahout.math.hadoop.decomposer.EigenVerificationJob;

/**
 * Implementation of the EigenCuts spectral clustering algorithm.
 */
public class SpectralKMeansDriver extends AbstractJob {

  public static final boolean DEBUG = false;

  public static final double OVERSHOOT_MULTIPLIER = 2.0;

  public static void main(String args[]) throws Exception {
    ToolRunner.run(new SpectralKMeansDriver(), args);
  }

  @Override
  public int run(String[] arg0) throws Exception {
    // set up command line options
    Configuration conf = new Configuration();
    addOption("input", "i", "Path to input affinity matrix data", true);
    addOption("output", "o", "Output of clusterings", true);
    addOption("dimensions", "d", "Square dimensions of affinity matrix", true);
    addOption("clusters", "k", "Number of clusters and top eigenvectors", true);
    Map<String, String> parsedArgs = parseArguments(arg0);
    if (parsedArgs == null) {
      return 0;
    }

    // TODO: Need to be able to read all k-means parameters, though
    // they will be optional parameters to the algorithm
    // read the values of the command line
    Path input = new Path(parsedArgs.get("--input"));
    Path output = new Path(parsedArgs.get("--output"));
    int numDims = Integer.parseInt(parsedArgs.get("--dimensions"));
    int clusters = Integer.parseInt(parsedArgs.get("--clusters"));

    // create a few new Paths for temp files and transformations
    Path outputCalc = new Path(output, "calculations");
    Path outputTmp = new Path(output, "temporary");

    // Take in the raw CSV text file and split it ourselves,
    // creating our own SequenceFiles for the matrices to read later 
    // (similar to the style of syntheticcontrol.canopy.InputMapper)
    Path affSeqFiles = new Path(outputCalc, "seqfile-" + (System.nanoTime() & 0xFF));
    AffinityMatrixInputJob.runJob(input, affSeqFiles, numDims, numDims);

    // Next step: construct the affinity matrix using the newly-created
    // sequence files
    DistributedRowMatrix A = new DistributedRowMatrix(affSeqFiles,
                                                      new Path(outputTmp, "afftmp-" + (System.nanoTime() & 0xFF)),
                                                      numDims,
                                                      numDims);
    JobConf depConf = new JobConf(conf);
    A.configure(depConf);

    // Next step: construct the diagonal matrix D (represented as a vector)
    // and calculate the normalized Laplacian of the form:
    // L = D^(-0.5)AD^(-0.5)
    Vector D = MatrixDiagonalizeJob.runJob(affSeqFiles, numDims);
    DistributedRowMatrix L = VectorMatrixMultiplicationJob.runJob(affSeqFiles, D, new Path(outputCalc, "laplacian-"
        + (System.nanoTime() & 0xFF)));
    L.configure(new JobConf(conf));

    // Next step: perform eigen-decomposition using LanczosSolver
    // since some of the eigen-output is spurious and will be eliminated
    // upon verification, we have to aim to overshoot and then discard
    // unnecessary vectors later
    int overshoot = (int) ((double) clusters * SpectralKMeansDriver.OVERSHOOT_MULTIPLIER);
    List<Double> eigenValues = new ArrayList<Double>(overshoot);
    Matrix eigenVectors = new DenseMatrix(overshoot, numDims);
    DistributedLanczosSolver solver = new DistributedLanczosSolver();
    Path lanczosSeqFiles = new Path(outputCalc, "eigenvectors-" + (System.nanoTime() & 0xFF));
    solver.runJob(conf,
                  L.getRowPath(),
                  new Path(outputTmp, "lanczos-" + (System.nanoTime() & 0xFF)),
                  L.numRows(),
                  L.numCols(),
                  true,
                  overshoot,
                  eigenVectors,
                  eigenValues,
                  lanczosSeqFiles.toString());

    // perform a verification
    EigenVerificationJob verifier = new EigenVerificationJob();
    Path verifiedEigensPath = new Path(outputCalc, "eigenverifier");
    verifier.runJob(lanczosSeqFiles, L.getRowPath(), verifiedEigensPath, true, 1.0, 0.0, clusters);
    Path cleanedEigens = verifier.getCleanedEigensPath();
    DistributedRowMatrix W = new DistributedRowMatrix(cleanedEigens, new Path(cleanedEigens, "tmp"), clusters, numDims);
    W.configure(new JobConf());
    DistributedRowMatrix Wtrans = W.transpose();
    //		DistributedRowMatrix Wt = W.transpose();

    // next step: normalize the rows of Wt to unit length
    Path unitVectors = new Path(outputCalc, "unitvectors-" + (System.nanoTime() & 0xFF));
    UnitVectorizerJob.runJob(Wtrans.getRowPath(), unitVectors);
    DistributedRowMatrix Wt = new DistributedRowMatrix(unitVectors, new Path(unitVectors, "tmp"), clusters, numDims);
    Wt.configure(new JobConf());

    //		Iterator<MatrixSlice> i = W.iterator();
    //		int x = 0;
    //		while (i.hasNext()) {
    //			Vector v = i.next().vector();
    //			System.out.println("EIGENVECTOR " + (++x));
    //			for (int c = 0; c < v.size(); c++) {
    //				System.out.print(v.get(c) + " ");
    //			}
    //			System.out.println();
    //		}
    //		System.exit(0);

    // Finally, perform k-means clustering on the rows of L (or W)
    // generate random initial clusters
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    Path initialclusters = RandomSeedGenerator.buildRandom(Wt.getRowPath(),
                                                           new Path(output, Cluster.INITIAL_CLUSTERS_DIR),
                                                           clusters,
                                                           measure);
    KMeansDriver.run(new Configuration(), Wt.getRowPath(), initialclusters, output, measure, 0.001, 10, true, false);

    // Read through the cluster assignments
    Path clusteredPointsPath = new Path(output, "clusteredPoints");
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(clusteredPointsPath, "part-m-00000"), conf);
    // The key is the clusterId
    IntWritable clusterId = new IntWritable(0);
    // The value is the weighted vector
    WeightedVectorWritable value = new WeightedVectorWritable();

    //	    Map<Integer, Integer> map = new HashMap<Integer, Integer>();
    int id = 0;
    while (reader.next(clusterId, value)) {
      //	    	Integer key = new Integer(clusterId.get());
      //	    	if (map.containsKey(key)) {
      //	    		Integer count = map.remove(key);
      //	    		map.put(key, new Integer(count.intValue() + 1));
      //	    	} else {
      //	    		map.put(key, new Integer(1));
      //	    	}
      System.out.println((id++) + ": " + clusterId.get());
      clusterId = new IntWritable(0);
      value = new WeightedVectorWritable();
    }
    reader.close();

    // TODO: output format???

    return 0;
  }
}
