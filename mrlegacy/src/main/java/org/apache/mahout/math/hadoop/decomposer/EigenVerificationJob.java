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

package org.apache.mahout.math.hadoop.decomposer;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.decomposer.EigenStatus;
import org.apache.mahout.math.decomposer.SimpleEigenVerifier;
import org.apache.mahout.math.decomposer.SingularVectorVerifier;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>
 * Class for taking the output of an eigendecomposition (specified as a Path location), and verifies correctness, in
 * terms of the following: if you have a vector e, and a matrix m, then let e' = m.timesSquared(v); the error w.r.t.
 * eigenvector-ness is the cosine of the angle between e and e':
 * </p>
 *
 * <pre>
 *   error(e,e') = e.dot(e') / (e.norm(2)*e'.norm(2))
 * </pre>
 * <p>
 * A set of eigenvectors should also all be very close to orthogonal, so this job computes all inner products between
 * eigenvectors, and checks that this is close to the identity matrix.
 * </p>
 * <p>
 * Parameters used in the cleanup (other than in the input/output path options) include --minEigenvalue, which specifies
 * the value below which eigenvector/eigenvalue pairs will be discarded, and --maxError, which specifies the maximum
 * error (as defined above) to be tolerated in an eigenvector.
 * </p>
 * <p>
 * If all the eigenvectors can fit in memory, --inMemory allows for a speedier completion of this task by doing so.
 * </p>
 */
public class EigenVerificationJob extends AbstractJob {

  public static final String CLEAN_EIGENVECTORS = "cleanEigenvectors";

  private static final Logger log = LoggerFactory.getLogger(EigenVerificationJob.class);

  private SingularVectorVerifier eigenVerifier;

  private VectorIterable eigensToVerify;

  private VectorIterable corpus;

  private double maxError;

  private double minEigenValue;

  // private boolean loadEigensInMemory;

  private Path tmpOut;

  private Path outPath;

  private int maxEigensToKeep;

  private Path cleanedEigensPath;

  public void setEigensToVerify(VectorIterable eigens) {
    eigensToVerify = eigens;
  }

  @Override
  public int run(String[] args) throws Exception {
    Map<String,List<String>> argMap = handleArgs(args);
    if (argMap == null) {
      return -1;
    }
    if (argMap.isEmpty()) {
      return 0;
    }
    // parse out the arguments
    runJob(getConf(), new Path(getOption("eigenInput")), new Path(getOption("corpusInput")), getOutputPath(),
        getOption("inMemory") != null, Double.parseDouble(getOption("maxError")),
        // Double.parseDouble(getOption("minEigenvalue")),
        Integer.parseInt(getOption("maxEigens")));
    return 0;
  }

  /**
   * Run the job with the given arguments
   *
   * @param corpusInput
   *          the corpus input Path
   * @param eigenInput
   *          the eigenvector input Path
   * @param output
   *          the output Path
   * @param tempOut
   *          temporary output Path
   * @param maxError
   *          a double representing the maximum error
   * @param minEigenValue
   *          a double representing the minimum eigenvalue
   * @param inMemory
   *          a boolean requesting in-memory preparation
   * @param conf
   *          the Configuration to use, or null if a default is ok (saves referencing Configuration in calling classes
   *          unless needed)
   */
  public int run(Path corpusInput, Path eigenInput, Path output, Path tempOut, double maxError, double minEigenValue,
      boolean inMemory, Configuration conf) throws IOException {
    this.outPath = output;
    this.tmpOut = tempOut;
    this.maxError = maxError;
    this.minEigenValue = minEigenValue;

    if (eigenInput != null && eigensToVerify == null) {
      prepareEigens(conf, eigenInput, inMemory);
    }
    DistributedRowMatrix c = new DistributedRowMatrix(corpusInput, tempOut, 1, 1);
    c.setConf(conf);
    corpus = c;

    // set up eigenverifier and orthoverifier TODO: allow multithreaded execution

    eigenVerifier = new SimpleEigenVerifier();

    // we don't currently verify orthonormality here.
    // VectorIterable pairwiseInnerProducts = computePairwiseInnerProducts();

    Map<MatrixSlice,EigenStatus> eigenMetaData = verifyEigens();

    List<Map.Entry<MatrixSlice,EigenStatus>> prunedEigenMeta = pruneEigens(eigenMetaData);

    saveCleanEigens(new Configuration(), prunedEigenMeta);
    return 0;
  }

  private Map<String,List<String>> handleArgs(String[] args) throws IOException {
    addOutputOption();
    addOption("eigenInput", "ei",
        "The Path for purported eigenVector input files (SequenceFile<WritableComparable,VectorWritable>.", null);
    addOption("corpusInput", "ci", "The Path for corpus input files (SequenceFile<WritableComparable,VectorWritable>.");
    addOption(DefaultOptionCreator.outputOption().create());
    addOption(DefaultOptionCreator.helpOption());
    addOption("inMemory", "mem", "Buffer eigen matrix into memory (if you have enough!)", "false");
    addOption("maxError", "err", "Maximum acceptable error", "0.05");
    addOption("minEigenvalue", "mev", "Minimum eigenvalue to keep the vector for", "0.0");
    addOption("maxEigens", "max", "Maximum number of eigenvectors to keep (0 means all)", "0");

    return parseArguments(args);
  }

  private void saveCleanEigens(Configuration conf, Collection<Map.Entry<MatrixSlice,EigenStatus>> prunedEigenMeta)
      throws IOException {
    Path path = new Path(outPath, CLEAN_EIGENVECTORS);
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    SequenceFile.Writer seqWriter = new SequenceFile.Writer(fs, conf, path, IntWritable.class, VectorWritable.class);
    try {
      IntWritable iw = new IntWritable();
      int numEigensWritten = 0;
      int index = 0;
      for (Map.Entry<MatrixSlice,EigenStatus> pruneSlice : prunedEigenMeta) {
        MatrixSlice s = pruneSlice.getKey();
        EigenStatus meta = pruneSlice.getValue();
        EigenVector ev = new EigenVector(s.vector(), meta.getEigenValue(), Math.abs(1 - meta.getCosAngle()), s.index());
        // log.info("appending {} to {}", ev, path);
        Writable vw = new VectorWritable(ev);
        iw.set(index++);
        seqWriter.append(iw, vw);

        // increment the number of eigenvectors written and see if we've
        // reached our specified limit, or if we wish to write all eigenvectors
        // (latter is built-in, since numEigensWritten will always be > 0
        numEigensWritten++;
        if (numEigensWritten == maxEigensToKeep) {
          log.info("{} of the {} total eigens have been written", maxEigensToKeep, prunedEigenMeta.size());
          break;
        }
      }
    } finally {
      Closeables.close(seqWriter, false);
    }
    cleanedEigensPath = path;
  }

  private List<Map.Entry<MatrixSlice,EigenStatus>> pruneEigens(Map<MatrixSlice,EigenStatus> eigenMetaData) {
    List<Map.Entry<MatrixSlice,EigenStatus>> prunedEigenMeta = Lists.newArrayList();

    for (Map.Entry<MatrixSlice,EigenStatus> entry : eigenMetaData.entrySet()) {
      if (Math.abs(1 - entry.getValue().getCosAngle()) < maxError && entry.getValue().getEigenValue() > minEigenValue) {
        prunedEigenMeta.add(entry);
      }
    }

    Collections.sort(prunedEigenMeta, new Comparator<Map.Entry<MatrixSlice,EigenStatus>>() {
      @Override
      public int compare(Map.Entry<MatrixSlice,EigenStatus> e1, Map.Entry<MatrixSlice,EigenStatus> e2) {
        // sort eigens on eigenvalues in descending order
        Double eg1 = e1.getValue().getEigenValue();
        Double eg2 = e2.getValue().getEigenValue();
        return eg1.compareTo(eg2);
      }
    });

    // iterate thru' the eigens, pick up ones with max orthogonality with the selected ones
    List<Map.Entry<MatrixSlice,EigenStatus>> selectedEigenMeta = Lists.newArrayList();
    Map.Entry<MatrixSlice,EigenStatus> e1 = prunedEigenMeta.remove(0);
    selectedEigenMeta.add(e1);
    int selectedEigenMetaLength = selectedEigenMeta.size();
    int prunedEigenMetaLength = prunedEigenMeta.size();

    while (prunedEigenMetaLength > 0) {
      double sum = Double.MAX_VALUE;
      int index = 0;
      for (int i = 0; i < prunedEigenMetaLength; i++) {
        Map.Entry<MatrixSlice,EigenStatus> e = prunedEigenMeta.get(i);
        double tmp = 0;
        for (int j = 0; j < selectedEigenMetaLength; j++) {
          Map.Entry<MatrixSlice,EigenStatus> ee = selectedEigenMeta.get(j);
          tmp += ee.getKey().vector().times(e.getKey().vector()).norm(2);
        }
        if (tmp < sum) {
          sum = tmp;
          index = i;
        }
      }
      Map.Entry<MatrixSlice,EigenStatus> e = prunedEigenMeta.remove(index);
      selectedEigenMeta.add(e);
      selectedEigenMetaLength++;
      prunedEigenMetaLength--;
    }

    return selectedEigenMeta;
  }

  private Map<MatrixSlice,EigenStatus> verifyEigens() {
    Map<MatrixSlice,EigenStatus> eigenMetaData = Maps.newHashMap();

    for (MatrixSlice slice : eigensToVerify) {
      EigenStatus status = eigenVerifier.verify(corpus, slice.vector());
      eigenMetaData.put(slice, status);
    }
    return eigenMetaData;
  }

  private void prepareEigens(Configuration conf, Path eigenInput, boolean inMemory) {
    DistributedRowMatrix eigens = new DistributedRowMatrix(eigenInput, tmpOut, 1, 1);
    eigens.setConf(conf);
    if (inMemory) {
      List<Vector> eigenVectors = Lists.newArrayList();
      for (MatrixSlice slice : eigens) {
        eigenVectors.add(slice.vector());
      }
      eigensToVerify = new SparseRowMatrix(eigenVectors.size(), eigenVectors.get(0).size(),
          eigenVectors.toArray(new Vector[eigenVectors.size()]), true, true);

    } else {
      eigensToVerify = eigens;
    }
  }

  public Path getCleanedEigensPath() {
    return cleanedEigensPath;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new EigenVerificationJob(), args);
  }

  /**
   * Progammatic invocation of run()
   *
   * @param eigenInput
   *          Output of LanczosSolver
   * @param corpusInput
   *          Input of LanczosSolver
   */
  public void runJob(Configuration conf, Path eigenInput, Path corpusInput, Path output, boolean inMemory,
      double maxError, int maxEigens) throws IOException {
    // no need to handle command line arguments
    outPath = output;
    tmpOut = new Path(outPath, "tmp");
    maxEigensToKeep = maxEigens;
    this.maxError = maxError;
    if (eigenInput != null && eigensToVerify == null) {
      prepareEigens(new Configuration(conf), eigenInput, inMemory);
    }

    DistributedRowMatrix c = new DistributedRowMatrix(corpusInput, tmpOut, 1, 1);
    c.setConf(new Configuration(conf));
    corpus = c;

    eigenVerifier = new SimpleEigenVerifier();

    Map<MatrixSlice,EigenStatus> eigenMetaData = verifyEigens();
    List<Map.Entry<MatrixSlice,EigenStatus>> prunedEigenMeta = pruneEigens(eigenMetaData);
    saveCleanEigens(conf, prunedEigenMeta);
  }
}
