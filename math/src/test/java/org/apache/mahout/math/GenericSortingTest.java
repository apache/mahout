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

import org.apache.mahout.math.function.IntComparator;
import org.junit.Assert;
import org.junit.Test;

public class GenericSortingTest extends Assert {

	private static class SomethingToSort implements Swapper, IntComparator {
		private final int[] data;

		private SomethingToSort(int[] data) {
			this.data = data;
		}

		@Override
		public void swap(int a, int b) {
			int temp = data[a];
			data[a] = data[b];
			data[b] = temp;
		}

		@Override
		public int compare(int o1, int o2) {
			if (data[o1] < data[o2]) {
				return -1;
			} else if (data[o1] > data[o2]) {
				return 1;
			} else {
				return 0;
			}
		}
	}

	@Test
	public void testQuickSort() {
		int[] td = new int[20];
		for (int x = 0; x < 20; x ++) {
			td[x] = 20 - x;
		}
		SomethingToSort sts = new SomethingToSort(td);
		Sorting.quickSort(0, 20, sts, sts);
		for (int x = 0; x < 20; x ++) {
			assertEquals(x+1, td[x]);
		}
	}

	private static class SomethingToSortStable implements Swapper, IntComparator {
		private final String[] data;

		private SomethingToSortStable(String[] data) {
			this.data = data;
		}

		@Override
		public void swap(int a, int b) {
			String temp = data[a];
			data[a] = data[b];
			data[b] = temp;
		}

		@Override
		public int compare(int o1, int o2) {
			return data[o1].compareTo(data[o2]);
		}
	}

	@Test
	public void testMergeSort() {
		String[] sd = {new String("z"), new String("a"), new String("a"), new String("q"), new String("1")};
		String[] correct = {sd[4], sd[1], sd[2], sd[3], sd[0]};

		SomethingToSortStable sts = new SomethingToSortStable(sd);
		GenericSorting.mergeSort(0, 5, sts, sts);

		for (int x = 0; x < 5; x ++) {
      assertSame(correct[x], sd[x]);
		}
	}
}
