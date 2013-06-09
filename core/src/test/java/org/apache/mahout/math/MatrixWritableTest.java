/**
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

package org.apache.mahout.math;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;

import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.hadoop.io.Writable;
import org.junit.Test;

public final class MatrixWritableTest extends MahoutTestCase {

	@Test
	public void testSparseMatrixWritable() throws Exception {
		Matrix m = new SparseMatrix(5, 5);
		m.set(1, 2, 3.0);
		m.set(3, 4, 5.0);
		Map<String, Integer> bindings = Maps.newHashMap();
		bindings.put("A", 0);
		bindings.put("B", 1);
		bindings.put("C", 2);
		bindings.put("D", 3);
		bindings.put("default", 4);
		m.setRowLabelBindings(bindings);
    m.setColumnLabelBindings(bindings);
		doTestMatrixWritableEquals(m);
	}

	@Test
	public void testDenseMatrixWritable() throws Exception {
		Matrix m = new DenseMatrix(5,5);
		m.set(1, 2, 3.0);
		m.set(3, 4, 5.0);
		Map<String, Integer> bindings = Maps.newHashMap();
		bindings.put("A", 0);
		bindings.put("B", 1);
		bindings.put("C", 2);
		bindings.put("D", 3);
		bindings.put("default", 4);
    m.setRowLabelBindings(bindings);
		m.setColumnLabelBindings(bindings);
		doTestMatrixWritableEquals(m);
	}

	private static void doTestMatrixWritableEquals(Matrix m) throws IOException {
		Writable matrixWritable = new MatrixWritable(m);
		MatrixWritable matrixWritable2 = new MatrixWritable();
		writeAndRead(matrixWritable, matrixWritable2);
		Matrix m2 = matrixWritable2.get();
		compareMatrices(m, m2); 
    doCheckBindings(m2.getRowLabelBindings());
    doCheckBindings(m2.getColumnLabelBindings());    
	}

	private static void compareMatrices(Matrix m, Matrix m2) {
		assertEquals(m.numRows(), m2.numRows());
		assertEquals(m.numCols(), m2.numCols());
		for (int r = 0; r < m.numRows(); r++) {
			for (int c = 0; c < m.numCols(); c++) {
				assertEquals(m.get(r, c), m2.get(r, c), EPSILON);
			}
		}
		Map<String,Integer> bindings = m.getRowLabelBindings();
		Map<String, Integer> bindings2 = m2.getRowLabelBindings();
		assertEquals(bindings == null, bindings2 == null);
		if (bindings != null) {
			assertEquals(bindings.size(), m.numRows());
			assertEquals(bindings.size(), bindings2.size());
			for (Map.Entry<String,Integer> entry : bindings.entrySet()) {
				assertEquals(entry.getValue(), bindings2.get(entry.getKey()));
			}
		}
		bindings = m.getColumnLabelBindings();
		bindings2 = m2.getColumnLabelBindings();
		assertEquals(bindings == null, bindings2 == null);
		if (bindings != null) {
			assertEquals(bindings.size(), bindings2.size());
			for (Map.Entry<String,Integer> entry : bindings.entrySet()) {
				assertEquals(entry.getValue(), bindings2.get(entry.getKey()));
			}
		}
	}

  private static void doCheckBindings(Map<String,Integer> labels) {
    assertTrue("Missing label", labels.keySet().contains("A"));
    assertTrue("Missing label", labels.keySet().contains("B"));
    assertTrue("Missing label", labels.keySet().contains("C"));
    assertTrue("Missing label", labels.keySet().contains("D"));
    assertTrue("Missing label", labels.keySet().contains("default"));
  }

	private static void writeAndRead(Writable toWrite, Writable toRead) throws IOException {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(baos);
		try {
			toWrite.write(dos);
		} finally {
			Closeables.close(dos, false);
		}

		ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
		DataInputStream dis = new DataInputStream(bais);
		try {
			toRead.readFields(dis);
		} finally {
			Closeables.close(dis, true);
		}
	}


}
