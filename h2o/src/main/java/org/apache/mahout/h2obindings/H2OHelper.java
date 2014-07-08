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

import org.apache.mahout.math.*;

import water.*;
import water.fvec.*;
import water.util.FrameUtils;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;

import scala.Tuple2;

public class H2OHelper {

  /*
    Is the matrix sparse? If the number of missing elements is
    32 x times the number of present elements, treat it as sparse
  */
  private static boolean is_sparse (Frame frame) {
    long rows = frame.numRows();
    long cols = frame.numCols();


    class MRTaskNZ extends MRTask<MRTaskNZ> {
      long _sparselen;
      public void map(Chunk chks[]) {
        for (Chunk chk : chks) {
          _sparselen += chk.sparseLen();
        }
      }
      public void reduce(MRTaskNZ other) {
        _sparselen += other._sparselen;
      }
    }

    long sparselen = new MRTaskNZ().doAll(frame)._sparselen;

    return (((rows * cols) / (sparselen + 1)) > 32);
  }

  /*
    Extract a Matrix from a Frame. Create either Sparse or
    Dense Matrix depending on number of missing elements
    in Frame.
  */
  public static Matrix matrix_from_frame (Frame frame, Vec labels) {
    Matrix m;

    if (is_sparse (frame))
      m = new SparseMatrix ((int)frame.numRows(), frame.numCols());
    else
      m = new DenseMatrix ((int)frame.numRows(), frame.numCols());

    int c = 0;
    for (Vec v : frame.vecs()) {
      for (int r = 0; r < frame.numRows(); r++) {
        double d = 0.0;
        if (!v.isNA(r) && ((d = v.at(r)) != 0.0))
          m.setQuick(r, c, d);
      }
      c++;
    }

    if (labels != null) {
      HashMap<String,Integer> map = new HashMap<String,Integer>();
      for (long i = 0; i < labels.length(); i++) {
        map.put(labels.atStr(i), (int)i);
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
  public static Vector colMeans (Frame frame) {
    double means[] = new double[frame.numCols()];
    for (int i = 0; i < frame.numCols(); i++)
      means[i] = frame.vecs()[i].mean();
    return new DenseVector(means);
  }

  /* Calculate Sum of all elements in a column, and
     return as a Vector

     Run an MRTask Job to add up sums in @_sums

     WARNING: Vulnerable to overflow. No way around it.
  */
  public static Vector colSums (Frame frame) {
    class MRTaskSum extends MRTask<MRTaskSum> {
      public double _sums[];
      public void map(Chunk chks[]) {
        _sums = new double[chks.length];

        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[c].len(); r++) {
            _sums[c] += chks[c].at0(r);
          }
        }
      }
      public void reduce(MRTaskSum other) {
        for (int i = 0; i < _sums.length; i++)
          _sums[i] += other._sums[i];
      }
    }
    return new DenseVector(new MRTaskSum().doAll(frame)._sums);
  }


  /* Calculate Sum of squares of all elements in the Matrix

     WARNING: Vulnerable to overflow. No way around it.
  */
  public static double sumSqr (Frame frame) {
    class MRTaskSumSqr extends MRTask<MRTaskSumSqr> {
      public double _sumSqr;
      public void map(Chunk chks[]) {
        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[c].len(); r++) {
            _sumSqr += (chks[c].at0(r) * chks[c].at0(r));
          }
        }
      }
      public void reduce(MRTaskSumSqr other) {
        _sumSqr += other._sumSqr;
      }
    }
    return new MRTaskSumSqr().doAll(frame)._sumSqr;
  }

  /* Calculate Sum of all elements in a column, and
     return as a Vector

     Run an MRTask Job to add up sums in @_sums

     WARNING: Vulnerable to overflow. No way around it.
  */
  public static Vector nonZeroCnt (Frame frame) {
    class MRTaskNonZero extends MRTask<MRTaskNonZero> {
      public double _sums[];
      public void map(Chunk chks[]) {
        _sums = new double[chks.length];

        for (int c = 0; c < chks.length; c++) {
          for (int r = 0; r < chks[c].len(); r++) {
            if ((long)chks[c].at0(r) != 0)
              _sums[c] ++;
          }
        }
      }
      public void reduce(MRTaskNonZero other) {
        for (int i = 0; i < _sums.length; i++)
          _sums[i] += other._sums[i];
      }
    }
    return new DenseVector(new MRTaskNonZero().doAll(frame)._sums);
  }


  public static Frame frame_from_file (String path) throws IOException {
    return FrameUtils.parseFrame(null, new File(path));
  }

  private static Map<Integer,String> reverse_map(Map<String,Integer> map) {
    if (map == null)
      return null;

    Map<Integer,String> rmap = new HashMap<Integer,String>();

    for(Map.Entry<String,Integer> entry : map.entrySet()) {
      rmap.put(entry.getValue(),entry.getKey());
    }

    return rmap;
  }

  private static int chunk_size (long nrow, int ncol, int min, int exact) {
    int chunk_sz;
    int parts_hint = Math.max(min, exact);

    if (parts_hint < 1)
      /* XXX: calculate based on cloud size and # of cpu */
      parts_hint = 4;

    chunk_sz = (int) (((nrow - 1) / parts_hint) + 1);
    if (exact > 0)
      return chunk_sz;

    if (chunk_sz > 1e6)
      chunk_sz = (int)1e6;

    if (min > 0)
      return chunk_sz;

    if (chunk_sz < 1e3)
      chunk_sz = (int)1e3;

    return chunk_sz;
  }

  private static int next_chunks(NewChunk ncs[], int cidx, long r, int chunk_sz, AppendableVec avs[], Futures fs) {
    if ((r % chunk_sz) != 0)
      return cidx;
    for (int i = 0; i < ncs.length; i++) {
      if (ncs[i] != null)
        ncs[i].close(fs);
      ncs[i] = new NewChunk (avs[i], cidx);
    }
    return cidx + 1;
  }
  /* Ingest a Matrix into an H2O Frame. H2O Frame is the "backing"
     data structure behind CheckpointedDrm. Steps:

     - @cols is the number of columsn in the Matrix
     - An H2O Vec represents an H2O Column.
     - Create @cols number of Vec's.
     - Load data into Vecs by routing them through NewChunks
  */
  public static Tuple2<Frame,Vec> frame_from_matrix (Matrix m, int min_hint, int exact_hint) {
    Map<String,Integer> map = m.getRowLabelBindings();
    Map<Integer,String> rmap = reverse_map(map);
    int cols = m.columnSize();
    int nvecs = cols + (map != null ? 1 : 0);
    Vec.VectorGroup vg = new Vec.VectorGroup();
    Key keys[] = vg.addVecs(nvecs);
    AppendableVec avs[] = new AppendableVec[nvecs];
    Vec vecs[] = new Vec[nvecs];
    NewChunk ncs[] = new NewChunk[nvecs];
    int chunk_sz = chunk_size (m.rowSize(), m.columnSize(), min_hint, exact_hint);
    Futures fs = new Futures();
    int cidx = 0;

    for (int c = 0; c < nvecs; c++)
      avs[c] = new AppendableVec(keys[c]);

    long r = 0;
    for (MatrixSlice row : m) {
      cidx = next_chunks (ncs, cidx, r, chunk_sz, avs, fs);
      /* Detect entire sparse rows */
      while (r < row.index()) {
        for (int i = 0; i < cols; i++)
          ncs[i].addNum(0.0);
        if (nvecs != cols)
          ncs[nvecs-1].addStr(null);
        r++;
        cidx = next_chunks (ncs, cidx, r, chunk_sz, avs, fs);
      }
      int c = 0;
      for (Vector.Element element : row.nonZeroes()) {
        while (c < element.index())
          /* Detect sparse column elements within a row */
          ncs[c++].addNum(0.0);
        ncs[c++].addNum(element.get());
      }
      if (rmap != null)
        ncs[nvecs-1].addStr(rmap.get(r));
      r++;
    }

    for (int c = 0; c < nvecs; c++) {
      ncs[c].close(fs);
      vecs[c] = avs[c].close(fs);
    }
    fs.blockForPending();

    Frame fr = new Frame(Arrays.copyOfRange(vecs,0,cols));
    Vec labels = (rmap != null) ? vecs[nvecs-1] : null;
    return new Tuple2<Frame,Vec>(fr,labels);
  }

  public static Frame empty_frame (long nrow, int ncol, int min_hint, int exact_hint) {
    int chunk_sz = chunk_size (nrow, ncol, min_hint, exact_hint);
    int nchunks = (int) ((nrow - 1) / chunk_sz) + 1; /* Final number of Chunks per Vec */
    Futures fs = new Futures();
    Vec.VectorGroup vg = new Vec.VectorGroup();
    Key keys[] = vg.addVecs(ncol);
    long espc[] = new long[nchunks+1];
    for (int i = 0; i < nchunks; i++)
      espc[i] = i * chunk_sz;
    espc[nchunks] = nrow;
    final Vec[] vecs = new Vec[ncol];
    for (int i = 0; i < vecs.length; i++)
      vecs[i] = new Vec(keys[i], espc);
    new MRTask() {
      protected void setupLocal() {
        for (Vec v : vecs) {
          for (int i = 0; i < v.nChunks(); i++) {
            Key k = v.chunkKey(i);
            if (k.home()) DKV.put(k, new C0LChunk(0L, v.chunkLen(i)), _fs);
          }
        }
        for(Vec v : vecs) if(v._key.home()) DKV.put(v._key, v, _fs);
      }
    }.doAllNodes();
    return new Frame(vecs);
  }
}
