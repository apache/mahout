/* Licensed to the Apache Software Foundation (ASF) under one or more
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

package org.apache.mahout.math.hadoop.stochasticsvd;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.MatrixColumnMeansJob;

/**
 * Mahout CLI adapter for SSVDSolver
 */
public class SSVDCli extends AbstractJob {

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption("rank", "k", "decomposition rank", true);
    addOption("oversampling", "p", "oversampling", String.valueOf(15));
    addOption("blockHeight",
              "r",
              "Y block height (must be > (k+p))",
              String.valueOf(10000));
    addOption("outerProdBlockHeight",
              "oh",
              "block height of outer products during multiplication, increase for sparse inputs",
              String.valueOf(30000));
    addOption("abtBlockHeight",
              "abth",
              "block height of Y_i in ABtJob during AB' multiplication, increase for extremely sparse inputs",
              String.valueOf(200000));
    addOption("minSplitSize", "s", "minimum split size", String.valueOf(-1));
    addOption("computeU", "U", "compute U (true/false)", String.valueOf(true));
    addOption("uHalfSigma",
              "uhs",
              "Compute U * Sigma^0.5",
              String.valueOf(false));
    addOption("uSigma", "us", "Compute U * Sigma", String.valueOf(false));
    addOption("computeV", "V", "compute V (true/false)", String.valueOf(true));
    addOption("vHalfSigma",
              "vhs",
              "compute V * Sigma^0.5",
              String.valueOf(false));
    addOption("reduceTasks",
              "t",
              "number of reduce tasks (where applicable)",
              true);
    addOption("powerIter",
              "q",
              "number of additional power iterations (0..2 is good)",
              String.valueOf(0));
    addOption("broadcast",
              "br",
              "whether use distributed cache to broadcast matrices wherever possible",
              String.valueOf(true));
    addOption("pca",
              "pca",
              "run in pca mode: compute column-wise mean and subtract from input",
              String.valueOf(false));
    addOption("pcaOffset",
              "xi",
              "path(glob) of external pca mean (optional, dont compute, use external mean");
    addOption(DefaultOptionCreator.overwriteOption().create());

    Map<String, List<String>> pargs = parseArguments(args);
    if (pargs == null) {
      return -1;
    }

    int k = Integer.parseInt(getOption("rank"));
    int p = Integer.parseInt(getOption("oversampling"));
    int r = Integer.parseInt(getOption("blockHeight"));
    int h = Integer.parseInt(getOption("outerProdBlockHeight"));
    int abh = Integer.parseInt(getOption("abtBlockHeight"));
    int q = Integer.parseInt(getOption("powerIter"));
    int minSplitSize = Integer.parseInt(getOption("minSplitSize"));
    boolean computeU = Boolean.parseBoolean(getOption("computeU"));
    boolean computeV = Boolean.parseBoolean(getOption("computeV"));
    boolean cUHalfSigma = Boolean.parseBoolean(getOption("uHalfSigma"));
    boolean cUSigma = Boolean.parseBoolean(getOption("uSigma"));
    boolean cVHalfSigma = Boolean.parseBoolean(getOption("vHalfSigma"));
    int reduceTasks = Integer.parseInt(getOption("reduceTasks"));
    boolean broadcast = Boolean.parseBoolean(getOption("broadcast"));
    String xiPathStr = getOption("pcaOffset");
    Path xiPath = xiPathStr == null ? null : new Path(xiPathStr);
    boolean pca = Boolean.parseBoolean(getOption("pca")) || xiPath != null;

    boolean overwrite = hasOption(DefaultOptionCreator.OVERWRITE_OPTION);

    Configuration conf = getConf();
    if (conf == null) {
      throw new IOException("No Hadoop configuration present");
    }

    Path[] inputPaths = { getInputPath() };
    Path tempPath = getTempPath();
    FileSystem fs = FileSystem.get(getTempPath().toUri(), conf);

    // housekeeping
    if (overwrite) {
      // clear the output path
      HadoopUtil.delete(getConf(), getOutputPath());
      // clear the temp path
      HadoopUtil.delete(getConf(), getTempPath());
    }

    fs.mkdirs(getOutputPath());

    // MAHOUT-817
    if (pca && xiPath == null) {
      xiPath = new Path(tempPath, "xi");
      if (overwrite) {
        fs.delete(xiPath, true);
      }
      MatrixColumnMeansJob.run(conf, inputPaths[0], xiPath);
    }

    SSVDSolver solver =
      new SSVDSolver(conf,
                     inputPaths,
                     new Path(tempPath, "ssvd"),
                     r,
                     k,
                     p,
                     reduceTasks);

    solver.setMinSplitSize(minSplitSize);
    solver.setComputeU(computeU);
    solver.setComputeV(computeV);
    solver.setcUHalfSigma(cUHalfSigma);
    solver.setcVHalfSigma(cVHalfSigma);
    solver.setcUSigma(cUSigma);
    solver.setOuterBlockHeight(h);
    solver.setAbtBlockHeight(abh);
    solver.setQ(q);
    solver.setBroadcast(broadcast);
    solver.setOverwrite(overwrite);

    if (xiPath != null) {
      solver.setPcaMeanPath(new Path(xiPath, "part-*"));
    }

    solver.run();

    Vector svalues = solver.getSingularValues().viewPart(0, k);
    SSVDHelper.saveVector(svalues, getOutputPath("sigma"), conf);

    if (computeU && !fs.rename(new Path(solver.getUPath()), getOutputPath())) {
      throw new IOException("Unable to move U results to the output path.");
    }
    if (cUHalfSigma
        && !fs.rename(new Path(solver.getuHalfSigmaPath()), getOutputPath())) {
      throw new IOException("Unable to move U*Sigma^0.5 results to the output path.");
    }
    if (cUSigma
        && !fs.rename(new Path(solver.getuSigmaPath()), getOutputPath())) {
      throw new IOException("Unable to move U*Sigma results to the output path.");
    }
    if (computeV && !fs.rename(new Path(solver.getVPath()), getOutputPath())) {
      throw new IOException("Unable to move V results to the output path.");
    }
    if (cVHalfSigma
        && !fs.rename(new Path(solver.getvHalfSigmaPath()), getOutputPath())) {
      throw new IOException("Unable to move V*Sigma^0.5 results to the output path.");
    }

    // Delete the temp path on exit
    fs.deleteOnExit(getTempPath());

    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SSVDCli(), args);
  }

}
