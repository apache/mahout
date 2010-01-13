package org.apache.mahout.math;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;


public class SparseVectorWritable extends SparseVector implements Writable {

  public SparseVectorWritable(SparseVector vector) {
    setName(vector.getName());
    cardinality = vector.cardinality;
    values = vector.values;
  }

  public SparseVectorWritable() {
    
  }

  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeUTF(getClass().getName());
    dataOutput.writeUTF(this.getName() == null ? "" : this.getName());
    dataOutput.writeInt(size());
    int nde = getNumNondefaultElements();
    dataOutput.writeInt(nde);
    Iterator<Vector.Element> iter = iterateNonZero();
    int count = 0;
    while (iter.hasNext()) {
      Vector.Element element = iter.next();
      dataOutput.writeInt(element.index());
      dataOutput.writeDouble(element.get());
      count++;
    }
    assert (nde == count);
  }

  public void readFields(DataInput dataInput) throws IOException {
    this.setName(dataInput.readUTF());
    this.cardinality = dataInput.readInt();
    int size = dataInput.readInt();
    OpenIntDoubleHashMap values = new OpenIntDoubleHashMap((int) (size * 1.5));
    int i = 0;
    while (i < size) {
      int index = dataInput.readInt();
      double value = dataInput.readDouble();
      values.put(index, value);
      i++;
    }
    assert (i == size);
    this.values = values;
    this.lengthSquared = -1.0;
  }
}
