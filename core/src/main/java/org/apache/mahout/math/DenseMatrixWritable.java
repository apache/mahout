package org.apache.mahout.math;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;


public class DenseMatrixWritable extends DenseMatrix implements Writable {

  public void readFields(DataInput in) throws IOException {
    columnLabelBindings = new HashMap<String, Integer>();
    rowLabelBindings = new HashMap<String, Integer>();
    MatrixWritable.readLabels(in, columnLabelBindings, rowLabelBindings);
    int rows = in.readInt();
    int columns = in.readInt();
    this.values = new double[rows][columns];
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        this.values[row][column] = in.readDouble();
      }
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    MatrixWritable.writeLabelBindings(out, columnLabelBindings, rowLabelBindings);
    out.writeInt(rowSize());
    out.writeInt(columnSize());
    for (double[] row : values) {
      for (double value : row) {
        out.writeDouble(value);
      }
    }
  }

}
