/*
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
import java.util.Map;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.decomposer.lanczos.LanczosState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HdfsBackedLanczosState extends LanczosState implements Configurable {

  private static final Logger log = LoggerFactory.getLogger(HdfsBackedLanczosState.class);

  public static final String BASIS_PREFIX = "basis";
  public static final String SINGULAR_PREFIX = "singular";
 //public static final String METADATA_FILE = "metadata";

  private Configuration conf;
  private final Path baseDir;
  private final Path basisPath;
  private final Path singularVectorPath;
  private FileSystem fs;
  
  public HdfsBackedLanczosState(VectorIterable corpus, int desiredRank, Vector initialVector, Path dir) {
    super(corpus, desiredRank, initialVector);
    baseDir = dir;
    //Path metadataPath = new Path(dir, METADATA_FILE);
    basisPath = new Path(dir, BASIS_PREFIX);
    singularVectorPath = new Path(dir, SINGULAR_PREFIX);
    if (corpus instanceof Configurable) {
      setConf(((Configurable)corpus).getConf());
    }
  }

  @Override public void setConf(Configuration configuration) {
    conf = configuration;
    try {
      setupDirs();
      updateHdfsState();
    } catch (IOException e) {
      log.error("Could not retrieve filesystem: {}", conf, e);
    }
  }

  @Override public Configuration getConf() {
    return conf;
  }

  private void setupDirs() throws IOException {
    fs = baseDir.getFileSystem(conf);
    createDirIfNotExist(baseDir);
    createDirIfNotExist(basisPath);
    createDirIfNotExist(singularVectorPath);
  }

  private void createDirIfNotExist(Path path) throws IOException {
    if (!fs.exists(path) && !fs.mkdirs(path)) {
      throw new IOException("Unable to create: " + path);
    }
  }

  @Override
  public void setIterationNumber(int i) {
    super.setIterationNumber(i);
    try {
      updateHdfsState();
    } catch (IOException e) {
      log.error("Could not update HDFS state: ", e);
    }
  }

  protected void updateHdfsState() throws IOException {
    if (conf == null) {
      return;
    }
    int numBasisVectorsOnDisk = 0;
    Path nextBasisVectorPath = new Path(basisPath, BASIS_PREFIX + '_' + numBasisVectorsOnDisk);
    while (fs.exists(nextBasisVectorPath)) {
      nextBasisVectorPath = new Path(basisPath, BASIS_PREFIX + '_' + ++numBasisVectorsOnDisk);
    }
    Vector nextVector;
    while (numBasisVectorsOnDisk < iterationNumber
        && (nextVector = getBasisVector(numBasisVectorsOnDisk)) != null) {
      persistVector(nextBasisVectorPath, numBasisVectorsOnDisk, nextVector);
      nextBasisVectorPath = new Path(basisPath, BASIS_PREFIX + '_' + ++numBasisVectorsOnDisk);
    }
    if (scaleFactor <= 0) {
      scaleFactor = getScaleFactor(); // load from disk if possible
    }
    diagonalMatrix = getDiagonalMatrix(); // load from disk if possible
    Vector norms = new DenseVector(diagonalMatrix.numCols() - 1);
    Vector projections = new DenseVector(diagonalMatrix.numCols());
    int i = 0;
    while (i < diagonalMatrix.numCols() - 1) {
      norms.set(i, diagonalMatrix.get(i, i + 1));
      projections.set(i, diagonalMatrix.get(i, i));
      i++;
    }
    projections.set(i, diagonalMatrix.get(i, i));
    persistVector(new Path(baseDir, "projections"), 0, projections);
    persistVector(new Path(baseDir, "norms"), 0, norms);
    persistVector(new Path(baseDir, "scaleFactor"), 0, new DenseVector(new double[] {scaleFactor}));
    for (Map.Entry<Integer, Vector> entry : singularVectors.entrySet()) {
      persistVector(new Path(singularVectorPath, SINGULAR_PREFIX + '_' + entry.getKey()),
          entry.getKey(), entry.getValue());
    }
    super.setIterationNumber(numBasisVectorsOnDisk);
  }

  protected void persistVector(Path p, int key, Vector vector) throws IOException {
    SequenceFile.Writer writer = null;
    try {
      if (fs.exists(p)) {
        log.warn("{} exists, will overwrite", p);
        fs.delete(p, true);
      }
      writer = new SequenceFile.Writer(fs, conf, p,
          IntWritable.class, VectorWritable.class);
      writer.append(new IntWritable(key), new VectorWritable(vector));
    } finally {
      Closeables.close(writer, false);
    }
  }

  protected Vector fetchVector(Path p, int keyIndex) throws IOException {
    if (!fs.exists(p)) {
      return null;
    }
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, p, conf);
    IntWritable key = new IntWritable();
    VectorWritable vw = new VectorWritable();
    while (reader.next(key, vw)) {
      if (key.get() == keyIndex) {
        return vw.get();
      }
    }
    return null;
  }

  @Override
  public Vector getBasisVector(int i) {
    if (!basis.containsKey(i)) {
      try {
        Vector v = fetchVector(new Path(basisPath, BASIS_PREFIX + '_' + i), i);
        basis.put(i, v);
      } catch (IOException e) {
        log.error("Could not load basis vector: {}", i, e);
      }
    }
    return super.getBasisVector(i);
  }

  @Override
  public Vector getRightSingularVector(int i) {
    if (!singularVectors.containsKey(i)) {
      try {
        Vector v = fetchVector(new Path(singularVectorPath, BASIS_PREFIX + '_' + i), i);
        singularVectors.put(i, v);
      } catch (IOException e) {
        log.error("Could not load singular vector: {}", i, e);
      }
    }
    return super.getRightSingularVector(i);
  }

  @Override
  public double getScaleFactor() {
    if (scaleFactor <= 0) {
      try {
        Vector v = fetchVector(new Path(baseDir, "scaleFactor"), 0);
        if (v != null && v.size() > 0) {
          scaleFactor = v.get(0);
        }
      } catch (IOException e) {
        log.error("could not load scaleFactor:", e);
      }
    }
    return scaleFactor;
  }

  @Override
  public Matrix getDiagonalMatrix() {
    if (diagonalMatrix == null) {
      diagonalMatrix = new DenseMatrix(desiredRank, desiredRank);
    }
    if (diagonalMatrix.get(0, 1) <= 0) {
      try {
        Vector norms = fetchVector(new Path(baseDir, "norms"), 0);
        Vector projections = fetchVector(new Path(baseDir, "projections"), 0);
        if (norms != null && projections != null) {
          int i = 0;
          while (i < projections.size() - 1) {
            diagonalMatrix.set(i, i, projections.get(i));
            diagonalMatrix.set(i, i + 1, norms.get(i));
            diagonalMatrix.set(i + 1, i, norms.get(i));
            i++;
          }
          diagonalMatrix.set(i, i, projections.get(i));
        }
      } catch (IOException e) {
        log.error("Could not load diagonal matrix of norms and projections: ", e);
      }
    }
    return diagonalMatrix;
  }

}
