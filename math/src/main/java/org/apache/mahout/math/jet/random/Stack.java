/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

/** Not yet commented. */
class Stack {

  private final int N;                      /* max number of elts on stack */
  private final int[] v;                     /* array of values on the stack */
  private int i;                      /* index of top of stack */

  /** Constructs a new stack with the given capacity. */
  Stack(int capacity) {
    this.N = capacity;
    this.i = -1; // indicates stack is empty
    this.v = new int[N];
  }

  /** Returns the topmost element. */
  public int pop() {
    if (this.i < 0) {
      throw new InternalError("Cannot pop stack!");
    }
    this.i--;
    return this.v[this.i + 1];
  }

  /** Places the given value on top of the stack. */
  public void push(int value) {
    this.i++;
    if (this.i >= this.N) {
      throw new InternalError("Cannot push stack!");
    }
    this.v[this.i] = value;
  }

  /** Returns the number of elements contained. */
  public int size() {
    return i + 1;
  }
}
