package org.apache.mahout.utils.vectors;

import junit.framework.TestCase;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

public class VectorHelperTest extends TestCase {

  public void testJsonFormatting() throws Exception {
    Vector v = new SequentialAccessSparseVector(10);
    v.set(2, 3.1);
    v.set(4, 1.0);
    v.set(6, 8.1);
    v.set(7, -100);
    v.set(9, 12.2);
    String UNUSED = "UNUSED";
    String[] dictionary = {
        UNUSED, UNUSED, "two", UNUSED, "four", UNUSED, "six", "seven", UNUSED, "nine"
    };

    assertEquals("sorted json form incorrect: ", "{nine:12.2,six:8.1,two:3.1}",
        VectorHelper.vectorToJson(v, dictionary, 3, true));
    assertEquals("unsorted form incorrect: ", "{two:3.1,four:1.0}",
        VectorHelper.vectorToJson(v, dictionary, 2, false));
  }

}
