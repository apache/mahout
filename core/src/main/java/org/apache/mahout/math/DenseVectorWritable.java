package org.apache.mahout.math;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

public class DenseVectorWritable extends DenseVector implements Writable {

  public DenseVectorWritable() {
    
  }

  public DenseVectorWritable(DenseVector v) {
    setName(v.getName());
    values = v.values;
    lengthSquared = v.lengthSquared;
  }

  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeUTF(getClass().getName());
    dataOutput.writeUTF(this.getName() == null ? "" : this.getName());
    dataOutput.writeInt(size());
    dataOutput.writeDouble(lengthSquared);
    Iterator<Vector.Element> iter = iterateAll();
    while (iter.hasNext()) {
      Vector.Element element = iter.next();
      dataOutput.writeDouble(element.get());
    }
  }

  public void readFields(DataInput dataInput) throws IOException {
    this.setName(dataInput.readUTF());
    double[] values = new double[dataInput.readInt()];
    lengthSquared = dataInput.readDouble();
    for (int i = 0; i < values.length; i++) {
      values[i] = dataInput.readDouble();
    }
    this.values = values;
  }
  
}
