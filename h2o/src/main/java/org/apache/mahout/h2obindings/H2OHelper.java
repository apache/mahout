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

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;

import org.apache.mahout.h2obindings.drm.H2ODrm;

public class H2OHelper {

  /*
    Is the matrix sparse? If the number of missing elements is
    32 x times the number of present elements, treat it as sparse
  */
  public static boolean is_sparse(Frame frame) {
    long rows = frame.numRows();
    long cols = frame.numCols();

    /* MRTask to aggregate precalculated per-chunk sparse lengths */
    class MRTaskNZ extends MRTask<MRTaskNZ> {
      long sparselen;
      public void map(Chunk chks[]) {
        for (Chunk chk : chks) {
          sparselen += chk.sparseLen();
        }
      }
      public void reduce(MRTaskNZ other) {
        sparselen += other.sparselen;
      }
    }

    long sparselen = new MRTaskNZ().doAll(frame).sparselen;

    return (((rows * cols) / (sparselen + 1)) > 32);
  }

  /*
    Extract a Matrix from a Frame. Create either Sparse or
    Dense Matrix depending on number of missing elements
    in Frame.
  */
  public static Matrix matrix_from_drm(H2ODrm drm) {
    Frame frame = drm.frame;
    Vec labels = drm.keys;
    Matrix m;

    if (is_sparse(frame)) {
      m = new SparseMatrix((int)frame.numRows(), frame.numCols());
    } else {
      m = new DenseMatrix((int)frame.numRows(), frame.numCols());
    }

    int c = 0;
    /* Fill matrix, column at a time */
    for (Vec v : frame.vecs()) {
      for (int r = 0; r < frame.numRows(); r++) {
        double d = 0.0;
        if (!v.isNA(r) && ((d = v.at(r)) != 0.0)) {
          m.setQuick(r, c, d);
        }
      }
      c++;
    }

    /* If string keyed, set the stings as rowlabels */
    if (labels != null) {
      HashMap<String,Integer> map = new HashMap<String,Integer>();
      ValueString vstr = new ValueString();
      for (long i = 0; i < labels.length(); i++) {
        map.put(labels.atStr(vstr, i).toString(), (int)i);
      }
      m.setRowLabelBindings(map);
    }
    return m;
  }

  /* Calculate Means of elements in a column, and return
     as a vector.

     H2O precalculates means in a Vec, and a Vec corresponds
     to a column.
  */
  public static Vector colMeans(Frame frame) {
    double means[] = new double[frame.numCols()];
    for (int i = 0; i < frame.numCols(); i++) {
      means[i] = frame.vecs()[i].mean();
    }
    return new DenseVector(means);
  }

  /* Calculate Sum of all elements in a column, and
     return as a Vector

     Run an MRTask Job to add up sums in @_sums

     WARNING: Vulnerable to overflow. No way around it.
  */
  public static Vector colSums(Frame frame) {
    class MRTaskSum extends MRTask<MRTaskSum> {
      public double sums[];
      public void map(Chunk chks[]) {
        sums = new double[chks.length];

        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[c].len(); r++) {
            sums[c] += chks[c].at0(r);
          }
        }
      }
      public void reduce(MRTaskSum other) {
        ArrayUtils.add(sums, other.sums);
      }
    }
    return new DenseVector(new MRTaskSum().doAll(frame).sums);
  }


  /* Calculate Sum of squares of all elements in the Matrix

     WARNING: Vulnerable to overflow. No way around it.
  */
  public static double sumSqr(Frame frame) {
    class MRTaskSumSqr extends MRTask<MRTaskSumSqr> {
      public double sumSqr;
      public void map(Chunk chks[]) {
        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[c].len(); r++) {
            sumSqr += (chks[c].at0(r) * chks[c].at0(r));
          }
        }
      }
      public void reduce(MRTaskSumSqr other) {
        sumSqr += other.sumSqr;
      }
    }
    return new MRTaskSumSqr().doAll(frame).sumSqr;
  }

  /* Calculate Sum of all elements in a column, and
     return as a Vector

     Run an MRTask Job to add up sums in @_sums

     WARNING: Vulnerable to overflow. No way around it.
  */
  public static Vector nonZeroCnt(Frame frame) {
    class MRTaskNonZero extends MRTask<MRTaskNonZero> {
      public double sums[];
      public void map(Chunk chks[]) {
        sums = new double[chks.length];

        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[c].len(); r++) {
            if ((long)chks[c].at0(r) != 0) {
              sums[c] ++;
            }
          }
        }
      }
      public void reduce(MRTaskNonZero other) {
        ArrayUtils.add(sums, other.sums);
      }
    }
    return new DenseVector(new MRTaskNonZero().doAll(frame).sums);
  }

  /* Convert String->Integer map to Integer->String map */
  private static Map<Integer,String> reverse_map(Map<String,Integer> map) {
    if (map == null) {
      return null;
    }

    Map<Integer,String> rmap = new HashMap<Integer,String>();

    for(Map.Entry<String,Integer> entry : map.entrySet()) {
      rmap.put(entry.getValue(),entry.getKey());
    }

    return rmap;
  }

  private static int chunk_size(long nrow, int ncol, int min, int exact) {
    int chunk_sz;
    int parts_hint = Math.max(min, exact);

    if (parts_hint < 1) {
      /* XXX: calculate based on cloud size and # of cpu */
      parts_hint = 4;
    }

    chunk_sz = (int)(((nrow - 1) / parts_hint) + 1);
    if (exact > 0) {
      return chunk_sz;
    }

    if (chunk_sz > 1e6) {
      chunk_sz = (int)1e6;
    }

    if (min > 0) {
      return chunk_sz;
    }

    if (chunk_sz < 1e3) {
      chunk_sz = (int)1e3;
    }

    return chunk_sz;
  }

  /* Ingest a Matrix into an H2O Frame. H2O Frame is the "backing"
     data structure behind CheckpointedDrm. Steps:
  */
  public static H2ODrm drm_from_matrix(Matrix m, int min_hint, int exact_hint) {
    /* First create an empty (0-filled) frame of the required dimensions */
    Frame frame = empty_frame(m.rowSize(), m.columnSize(), min_hint, exact_hint);
    Vec labels = null;
    Vec.Writer writers[] = new Vec.Writer[m.columnSize()];
    Futures closer = new Futures();

    /* "open" vectors for writing efficiently in bulk */
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

    /* If string labeled matrix, create aux Vec */
    Map<String,Integer> map = m.getRowLabelBindings();
    if (map != null) {
      /* label vector must be similarly partitioned like the Frame */
      labels = frame.anyVec().makeZero();
      Vec.Writer writer = labels.open();
      Map<Integer,String> rmap = reverse_map(map);

      for (long r = 0; r < m.rowSize(); r++) {
        writer.set(r, rmap.get(r));
      }

      writer.close(closer);
    }

    closer.blockForPending();

    return new H2ODrm(frame, labels);
  }

  public static Frame empty_frame(long nrow, int ncol, int min_hint, int exact_hint) {
    Vec.VectorGroup vg = new Vec.VectorGroup();

    return empty_frame(nrow, ncol, min_hint, exact_hint, vg);
  }

  public static Frame empty_frame(long nrow, int ncol, int min_hint, int exact_hint, Vec.VectorGroup vg) {
    int chunk_sz = chunk_size(nrow, ncol, min_hint, exact_hint);
    int nchunks = (int)((nrow - 1) / chunk_sz) + 1; /* Final number of Chunks per Vec */
    long espc[] = new long[nchunks + 1];
    final Vec[] vecs = new Vec[ncol];

    for (int i = 0; i < nchunks; i++) {
      espc[i] = i * chunk_sz;
    }
    espc[nchunks] = nrow;

    for (int i = 0; i < vecs.length; i++) {
      vecs[i] = Vec.makeCon(0, null, vg, espc);
    }

    return new Frame(vecs);
  }

  public static H2ODrm empty_drm(long nrow, int ncol, int min_hint, int exact_hint) {
    return new H2ODrm(empty_frame(nrow, ncol, min_hint, exact_hint));
  }
}
