package org.apache.mahout.cf.taste.hadoop.item;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class PrefAndSimilarityColumnWritable implements Writable {

  private float prefValue;
  private Vector similarityColumn;

  public PrefAndSimilarityColumnWritable() {
    super();
  }

  public PrefAndSimilarityColumnWritable(float prefValue, Vector similarityColumn) {
    super();
    set(prefValue, similarityColumn);
  }

  public void set(float prefValue, Vector similarityColumn) {
    this.prefValue = prefValue;
    this.similarityColumn = similarityColumn;
  }

  public float getPrefValue() {
    return prefValue;
  }

  public Vector getSimilarityColumn() {
    return similarityColumn;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    prefValue = in.readFloat();
    VectorWritable vw = new VectorWritable();
    vw.readFields(in);
    similarityColumn = vw.get();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeFloat(prefValue);
    VectorWritable vw = new VectorWritable(similarityColumn);
    vw.setWritesLaxPrecision(true);
    vw.write(out);
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof PrefAndSimilarityColumnWritable) {
      PrefAndSimilarityColumnWritable other = (PrefAndSimilarityColumnWritable) obj;
      return prefValue == other.prefValue && similarityColumn.equals(other.similarityColumn);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return RandomUtils.hashFloat(prefValue) + 31 * similarityColumn.hashCode();
  }


}
