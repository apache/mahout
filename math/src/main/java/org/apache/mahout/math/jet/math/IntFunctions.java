/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.math;

import org.apache.mahout.math.function.IntFunction;
import org.apache.mahout.math.function.IntIntFunction;
import org.apache.mahout.math.jet.random.engine.MersenneTwister;

import java.util.Date;

/**
 Integer Function objects to be passed to generic methods.
 Same as {@link Functions} except operating on integers.
 <p>
 For aliasing see {@link #intFunctions}.

 @author wolfgang.hoschek@cern.ch
 @version 1.0, 09/24/99
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class IntFunctions {

  /*****************************
   * <H3>Unary functions</H3>
   *****************************/
  /** Function that returns <tt>Math.abs(a) == (a < 0) ? -a : a</tt>. */
  public static final IntFunction abs = new IntFunction() {
    @Override
    public int apply(int a) {
      return (a < 0) ? -a : a;
    }
  };

  /** Function that returns its argument. */
  public static final IntFunction identity = new IntFunction() {
    @Override
    public int apply(int a) {
      return a;
    }
  };

  /** Function that returns <tt>~a</tt>. */
  public static final IntFunction not = new IntFunction() {
    @Override
    public int apply(int a) {
      return ~a;
    }
  };

  /** Function that returns <tt>a < 0 ? -1 : a > 0 ? 1 : 0</tt>. */
  public static final IntFunction sign = new IntFunction() {
    @Override
    public int apply(int a) {
      return a < 0 ? -1 : a > 0 ? 1 : 0;
    }
  };

  /** Function that returns <tt>a * a</tt>. */
  public static final IntFunction square = new IntFunction() {
    @Override
    public int apply(int a) {
      return a * a;
    }
  };


  /*****************************
   * <H3>Binary functions</H3>
   *****************************/

  /** Function that returns <tt>a & b</tt>. */
  public static final IntIntFunction and = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return a & b;
    }
  };

  /** Function that returns <tt>a < b ? -1 : a > b ? 1 : 0</tt>. */
  public static final IntIntFunction compare = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return a < b ? -1 : a > b ? 1 : 0;
    }
  };

  /** Function that returns <tt>a / b</tt>. */
  public static final IntIntFunction div = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return a / b;
    }
  };

  /** Function that returns <tt>a == b ? 1 : 0</tt>. */
  public static final IntIntFunction equals = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return a == b ? 1 : 0;
    }
  };

  /** Function that returns <tt>Math.max(a,b)</tt>. */
  public static final IntIntFunction max = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return (a >= b) ? a : b;
    }
  };

  /** Function that returns <tt>Math.min(a,b)</tt>. */
  public static final IntIntFunction min = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return (a <= b) ? a : b;
    }
  };

  /** Function that returns <tt>a - b</tt>. */
  public static final IntIntFunction minus = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return a - b;
    }
  };

  /** Function that returns <tt>a * b</tt>. */
  public static final IntIntFunction mult = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return a * b;
    }
  };

  /** Function that returns <tt>a | b</tt>. */
  public static final IntIntFunction or = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return a | b;
    }
  };

  /** Function that returns <tt>a + b</tt>. */
  public static final IntIntFunction plus = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return a + b;
    }
  };

  /** Function that returns <tt>(int) Math.pow(a,b)</tt>. */
  public static final IntIntFunction pow = new IntIntFunction() {
    @Override
    public int apply(int a, int b) {
      return (int) Math.pow(a, b);
    }
  };

  private IntFunctions() {
  }

  /** Constructs a function that returns <tt>a & b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  public static IntFunction and(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return a & b;
      }
    };
  }

  /**
   * Constructs a function that returns <tt>(from<=a && a<=to) ? 1 : 0</tt>. <tt>a</tt> is a variable, <tt>from</tt> and
   * <tt>to</tt> are fixed.
   */
  public static IntFunction between(final int from, final int to) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return (from <= a && a <= to) ? 1 : 0;
      }
    };
  }

  /**
   * Constructs the function <tt>g( h(a) )</tt>.
   *
   * @param g a unary function.
   * @param h a unary function.
   * @return the unary function <tt>g( h(a) )</tt>.
   */
  public static IntFunction chain(final IntFunction g, final IntFunction h) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return g.apply(h.apply(a));
      }
    };
  }

  /**
   * Constructs the function <tt>g( h(a,b) )</tt>.
   *
   * @param g a unary function.
   * @param h a binary function.
   * @return the unary function <tt>g( h(a,b) )</tt>.
   */
  public static IntIntFunction chain(final IntFunction g, final IntIntFunction h) {
    return new IntIntFunction() {
      @Override
      public int apply(int a, int b) {
        return g.apply(h.apply(a, b));
      }
    };
  }

  /**
   * Constructs the function <tt>f( g(a), h(b) )</tt>.
   *
   * @param f a binary function.
   * @param g a unary function.
   * @param h a unary function.
   * @return the binary function <tt>f( g(a), h(b) )</tt>.
   */
  public static IntIntFunction chain(final IntIntFunction f, final IntFunction g, final IntFunction h) {
    return new IntIntFunction() {
      @Override
      public int apply(int a, int b) {
        return f.apply(g.apply(a), h.apply(b));
      }
    };
  }

  /**
   * Constructs a function that returns <tt>a < b ? -1 : a > b ? 1 : 0</tt>. <tt>a</tt> is a variable, <tt>b</tt> is
   * fixed.
   */
  public static IntFunction compare(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return a < b ? -1 : a > b ? 1 : 0;
      }
    };
  }

  /** Constructs a function that returns the constant <tt>c</tt>. */
  public static IntFunction constant(final int c) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return c;
      }
    };
  }

  /** Constructs a function that returns <tt>a / b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  public static IntFunction div(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return a / b;
      }
    };
  }

  /** Constructs a function that returns <tt>a == b ? 1 : 0</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  public static IntFunction equals(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return a == b ? 1 : 0;
      }
    };
  }

  /** Constructs a function that returns <tt>Math.max(a,b)</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  public static IntFunction max(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return (a >= b) ? a : b;
      }
    };
  }

  /** Constructs a function that returns <tt>Math.min(a,b)</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  public static IntFunction min(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return (a <= b) ? a : b;
      }
    };
  }

  /** Constructs a function that returns <tt>a - b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  public static IntFunction minus(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return a - b;
      }
    };
  }

  /** Constructs a function that returns <tt>a * b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  public static IntFunction mult(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return a * b;
      }
    };
  }

  /** Constructs a function that returns <tt>a | b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  public static IntFunction or(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return a | b;
      }
    };
  }

  /** Constructs a function that returns <tt>a + b</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  public static IntFunction plus(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return a + b;
      }
    };
  }

  /** Constructs a function that returns <tt>(int) Math.pow(a,b)</tt>. <tt>a</tt> is a variable, <tt>b</tt> is fixed. */
  public static IntFunction pow(final int b) {
    return new IntFunction() {
      @Override
      public int apply(int a) {
        return (int) Math.pow(a, b);
      }
    };
  }

  /**
   * Constructs a function that returns a 32 bit uniformly distributed random number in the closed interval
   * <tt>[Integer.MIN_VALUE,Integer.MAX_VALUE]</tt> (including <tt>Integer.MIN_VALUE</tt> and
   * <tt>Integer.MAX_VALUE</tt>). Currently the engine is {@link org.apache.mahout.math.jet.random.engine.MersenneTwister}
   * and is seeded with the current time. <p> Note that any random engine derived from {@link
   * org.apache.mahout.math.jet.random.engine.RandomEngine} and any random distribution derived from {@link
   * org.apache.mahout.math.jet.random.AbstractDistribution} are function objects, because they implement the proper
   * interfaces. Thus, if you are not happy with the default, just pass your favourite random generator to function
   * evaluating methods.
   */
  public static IntFunction random() {
    return new MersenneTwister(new Date());
  }

}
