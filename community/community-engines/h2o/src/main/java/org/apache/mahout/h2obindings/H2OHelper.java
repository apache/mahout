/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.h2obindings;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.DenseVector;

import water.MRTask;
import water.Futures;
import water.fvec.Frame;
import water.fvec.Vec;
import water.fvec.Chunk;
import water.parser.ValueString;
import water.util.ArrayUtils;

import java.util.Map;
import java.util.HashMap;
import java.io.Serializable;

import org.apache.mahout.h2obindings.drm.H2ODrm;
import org.apache.mahout.h2obindings.drm.H2OBCast;

// for makeEmptyStrVec
import water.Key;
import water.DKV;
import water.fvec.CStrChunk;

import scala.Function1;
import scala.Function2;

/**
 * Collection of helper methods for H2O backend.
 */
public class H2OHelper {
  /**
   * Predicate to check if data is sparse in Frame.
   *
   * If the number of missing elements is 32x times the number of present
   * elements, consider it as sparse.
   *
   * @param frame Frame storing matrix data.
   * @return True if data is sparse in Frame.
   */
  public static boolean isSparse(Frame frame) {
    long rows = frame.numRows();
    long cols = frame.numCols();

    /**
     * MRTask to aggregate precalculated per-chunk sparse lengths
     */
    class MRTaskNZ extends MRTask<MRTaskNZ> {
      long sparselen;
      @Override
      public void map(Chunk chks[]) {
        for (Chunk chk : chks) {
          sparselen += chk.sparseLen();
        }
      }
      @Override
      public void reduce(MRTaskNZ other) {
        sparselen += other.sparselen;
      }
    }

    long sparselen = new MRTaskNZ().doAll(frame).sparselen;

    return (((rows * cols) / (sparselen + 1)) > 32);
  }

  /**
   * Create a Mahout Matrix from a DRM.
   *
   * Create either Sparse or Dense Matrix depending on number of missing
   * elements in DRM.
   *
   * @param drm DRM object to create Matrix from.
   * @return created Matrix.
   */
  public static Matrix matrixFromDrm(H2ODrm drm) {
    Frame frame = drm.frame;
    Vec labels = drm.keys;
    Matrix m;

    if (isSparse(frame)) {
      m = new SparseMatrix((int)frame.numRows(), frame.numCols());
    } else {
      m = new DenseMatrix((int)frame.numRows(), frame.numCols());
    }

    int c = 0;
    // Fill matrix, column at a time.
    for (Vec v : frame.vecs()) {
      for (int r = 0; r < frame.numRows(); r++) {
        double d;
        if (!v.isNA(r) && ((d = v.at(r)) != 0.0)) {
          m.setQuick(r, c, d);
        }
      }
      c++;
    }

    // If string keyed, set the stings as rowlabels.
    if (labels != null) {
      Map<String,Integer> map = new HashMap<>();
      ValueString vstr = new ValueString();
      for (long i = 0; i < labels.length(); i++) {
        map.put(labels.atStr(vstr, i).toString(), (int)i);
      }
      m.setRowLabelBindings(map);
    }
    return m;
  }

  /**
   * Calculate Means of elements in a column, and return as a Vector.
   *
   * H2O precalculates means in a Vec, and a Vec corresponds to a column.
   *
   * @param frame Frame backing the H2O DRM.
   * @return Vector of pre-calculated means.
   */
  public static Vector colMeans(Frame frame) {
    double means[] = new double[frame.numCols()];
    for (int i = 0; i < frame.numCols(); i++) {
      means[i] = frame.vecs()[i].mean();
    }
    return new DenseVector(means);
  }

  /**
   * Calculate Sums of elements in a column, and return as a Vector.
   *
   * Run an MRTask Job to add up sums.
   * WARNING: Vulnerable to overflow. No way around it.
   *
   * @param frame Frame backing the H2O DRM.
   * @return Vector of calculated sums.
   */
  public static Vector colSums(Frame frame) {
    /**
     * MRTask to calculate sums of elements in all columns.
     */
    class MRTaskSum extends MRTask<MRTaskSum> {
      public double sums[];
      @Override
      public void map(Chunk chks[]) {
        sums = new double[chks.length];

        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[c].len(); r++) {
            sums[c] += chks[c].atd(r);
          }
        }
      }
      @Override
      public void reduce(MRTaskSum other) {
        ArrayUtils.add(sums, other.sums);
      }
    }
    return new DenseVector(new MRTaskSum().doAll(frame).sums);
  }

  /**
   * Calculate Sum of squares of all elements in the DRM.
   *
   * Run an MRTask Job to add up sums of squares.
   * WARNING: Vulnerable to overflow. No way around it.
   *
   * @param frame Frame backing the H2O DRM.
   * @return Sum of squares of all elements in the DRM.
   */
  public static double sumSqr(Frame frame) {
    /**
     * MRTask to calculate sums of squares of all elements.
     */
    class MRTaskSumSqr extends MRTask<MRTaskSumSqr> {
      public double sumSqr;
      @Override
      public void map(Chunk chks[]) {
        for (Chunk chk : chks) {
          for (int r = 0; r < chk.len(); r++) {
            sumSqr += (chk.atd(r) * chk.atd(r));
          }
        }
      }
      @Override
      public void reduce(MRTaskSumSqr other) {
        sumSqr += other.sumSqr;
      }
    }
    return new MRTaskSumSqr().doAll(frame).sumSqr;
  }

  /**
   * Count non-zero elements in all columns, and return as a Vector.
   *
   * Run an MRTask Job to count non-zero elements per column.
   *
   * @param frame Frame backing the H2O DRM.
   * @return Vector of counted non-zero elements.
   */
  public static Vector nonZeroCnt(Frame frame) {
    /**
     * MRTask to count all non-zero elements.
     */
    class MRTaskNonZero extends MRTask<MRTaskNonZero> {
      public double sums[];
      @Override
      public void map(Chunk chks[]) {
        sums = new double[chks.length];

        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[c].len(); r++) {
            if ((long)chks[c].atd(r) != 0) {
              sums[c] ++;
            }
          }
        }
      }
      @Override
      public void reduce(MRTaskNonZero other) {
        ArrayUtils.add(sums, other.sums);
      }
    }
    return new DenseVector(new MRTaskNonZero().doAll(frame).sums);
  }

  /** Convert String->Integer map to Integer->String map */
  private static Map<Integer,String> reverseMap(Map<String, Integer> map) {
    if (map == null) {
      return null;
    }

    Map<Integer,String> rmap = new HashMap<>();

    for(Map.Entry<String,Integer> entry : map.entrySet()) {
      rmap.put(entry.getValue(),entry.getKey());
    }

    return rmap;
  }

  /**
   * Calculate optimum chunk size for given parameters.
   *
   * Chunk size is the number of elements stored per partition per column.
   *
   * @param nrow Number of rows in the DRM.
   * @param minHint Minimum number of partitions to create, if passed value is not -1.
   * @param exactHint Exact number of partitions to create, if passed value is not -1.
   * @return Calculated optimum chunk size.
   */
  private static int chunkSize(long nrow, int minHint, int exactHint) {
    int chunkSz;
    int partsHint = Math.max(minHint, exactHint);

    if (partsHint < 1) {
      /* XXX: calculate based on cloud size and # of cpu */
      partsHint = 4;
    }

    chunkSz = (int)(((nrow - 1) / partsHint) + 1);
    if (exactHint > 0) {
      return chunkSz;
    }

    if (chunkSz > 1e6) {
      chunkSz = (int)1e6;
    }

    if (minHint > 0) {
      return chunkSz;
    }

    if (chunkSz < 1e3) {
      chunkSz = (int)1e3;
    }

    return chunkSz;
  }

  /**
   * Ingest a Mahout Matrix into an H2O DRM.
   *
   * Frame is the backing data structure behind CheckpointedDrm.
   *
   * @param m Mahout Matrix to ingest data from.
   * @param minHint Hint for minimum number of partitions in created DRM.
   * @param exactHint Hint for exact number of partitions in created DRM.
   * @return Created H2O backed DRM.
   */
  public static H2ODrm drmFromMatrix(Matrix m, int minHint, int exactHint) {
    // First create an empty (0-filled) frame of the required dimensions
    Frame frame = emptyFrame(m.rowSize(), m.columnSize(), minHint, exactHint);
    Vec labels = null;
    Vec.Writer writers[] = new Vec.Writer[m.columnSize()];
    Futures closer = new Futures();

    // "open" vectors for writing efficiently in bulk
    for (int i = 0; i < writers.length; i++) {
      writers[i] = frame.vecs()[i].open();
    }

    for (int r = 0; r < m.rowSize(); r++) {
      for (int c = 0; c < m.columnSize(); c++) {
        writers[c].set(r, m.getQuick(r, c));
      }
    }

    for (int c = 0; c < m.columnSize(); c++) {
      writers[c].close(closer);
    }
    // If string labeled matrix, create aux Vec
    Map<String,Integer> map = m.getRowLabelBindings();
    if (map != null) {
      // label vector must be similarly partitioned like the Frame
      labels = makeEmptyStrVec(frame.anyVec());
      Vec.Writer writer = labels.open();
      Map<Integer,String> rmap = reverseMap(map);
      for (int r = 0; r < m.rowSize(); r++) {
        writer.set(r, rmap.get(r));
      }

      writer.close(closer);
    }

    closer.blockForPending();

    return new H2ODrm(frame, labels);
  }

  /**
   * Create an empty (zero-filled) H2O Frame efficiently.
   *
   * Create a zero filled Frame with specified cardinality.
   * Do not actually fill zeroes in each cell, create pre-compressed chunks.
   * Time taken per column asymptotically at O(nChunks), not O(nrow).
   *
   * @param nrow Number of rows in the Frame.
   * @param ncol Number of columns in the Frame.
   * @param minHint Hint for minimum number of chunks per column in created Frame.
   * @param exactHint Hint for exact number of chunks per column in created Frame.
   * @return Created Frame.
   */
  public static Frame emptyFrame(long nrow, int ncol, int minHint, int exactHint) {
    Vec.VectorGroup vg = new Vec.VectorGroup();

    return emptyFrame(nrow, ncol, minHint, exactHint, vg);
  }

  /**
   * Create an empty (zero-filled) H2O Frame efficiently.
   *
   * Create a zero filled Frame with specified cardinality.
   * Do not actually fill zeroes in each cell, create pre-compressed chunks.
   * Time taken per column asymptotically at O(nChunks), not O(nrow).
   *
   * @param nrow Number of rows in the Frame.
   * @param ncol Number of columns in the Frame.
   * @param minHint Hint for minimum number of chunks per column in created Frame.
   * @param exactHint Hint for exact number of chunks per column in created Frame.
   * @param vg Shared VectorGroup so that all columns are similarly partitioned.
   * @return Created Frame.
   */
  public static Frame emptyFrame(long nrow, int ncol, int minHint, int exactHint, Vec.VectorGroup vg) {
    int chunkSz = chunkSize(nrow, minHint, exactHint);
    int nchunks = (int)((nrow - 1) / chunkSz) + 1; // Final number of Chunks per Vec
    long espc[] = new long[nchunks + 1];

    for (int i = 0; i < nchunks; i++) {
      espc[i] = i * chunkSz;
    }
    espc[nchunks] = nrow;
    // Create a vector template for new vectors
    Vec vtemplate = new Vec(vg.addVec(), espc);
    // Make ncol-numeric vectors
    Vec[] vecs = vtemplate.makeCons(ncol, 0, null, null);

    return new Frame(vecs);
  }


  /**
   * The following two methods: vecChunkLen and makeEmptyStrVec
   * are h2o-0.1.25 specific.
   */
  public static Vec makeEmptyStrVec(final Vec template) {
    final int nChunks = template.nChunks();
    Key<Vec> key = template.group().addVec();
    final Vec emptystr = new Vec(key, template._espc, null, Vec.T_NUM);

    new MRTask() {
      @Override protected void setupLocal() {
        for (int i = 0; i < nChunks; i++) {
          Key k = emptystr.chunkKey(i);
          int chklen = vecChunkLen(template, i);
          int stridx[] = new int[chklen];
          byte b[] = new byte[1]; b[0] = 0;
          for (int j = 0; j < chklen; j++) stridx[j] = -1;
          if (k.home()) DKV.put(k, new CStrChunk(1, b, chklen, stridx), _fs);
        }
        if (emptystr._key.home()) DKV.put(emptystr._key, emptystr, _fs);
      }
    }.doAllNodes();
    return emptystr;
  }

  public static int vecChunkLen(Vec template, int chunk) {
    return (int) (template._espc[chunk + 1] - template._espc[chunk]);
  }

  /**
   * Create an empty (zero-filled) H2O DRM.
   *
   * Create a zero filled DRM with specified cardinality.
   * Use the efficient emptyFrame() method internally.
   *
   * @param nrow Number of rows in the Frame.
   * @param ncol Number of columns in the Frame.
   * @param minHint Hint for minimum number of chunks per column in created Frame.
   * @param exactHint Hint for exact number of chunks per column in created Frame.
   * @return Created DRM.
   */
  public static H2ODrm emptyDrm(long nrow, int ncol, int minHint, int exactHint) {
    return new H2ODrm(emptyFrame(nrow, ncol, minHint, exactHint));
  }

  public static Matrix allreduceBlock(H2ODrm drmA, Object bmfn, Object rfn) {
    class MRTaskMR extends MRTask<MRTaskMR> {
      H2OBCast<Matrix> bmf_out;
      Serializable bmf;
      Serializable rf;

      public MRTaskMR(Object _bmf, Object _rf) {
        bmf = (Serializable) _bmf;
        rf = (Serializable) _rf;
      }

      @Override
      public void map(Chunk chks[]) {
        Function1 f = (Function1) bmf;
        bmf_out = new H2OBCast((Matrix)f.apply(new scala.Tuple2(null, new H2OBlockMatrix(chks))));
      }

      @Override
      public void reduce(MRTaskMR that) {
        Function2 f = (Function2) rf;
        bmf_out = new H2OBCast((Matrix)f.apply(this.bmf_out.value(), that.bmf_out.value()));
      }
    }

    return new MRTaskMR(bmfn, rfn).doAll(drmA.frame).bmf_out.value();
  }
}
