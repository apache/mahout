/**
 * <HTML>
 * <BODY>
 * Resizable lists holding objects or primitive data types such as <tt>int</tt>,
 * <tt>double</tt>, etc. For non-resizable lists (1-dimensional matrices) see
 * package <code>org.apache.mahout.math.matrix</code>.<p></p>
 * <h1><a name="Overview"></a>Getting Started</h1>
 * <h2>1. Overview</h2>
 * <p>The list package offers flexible object oriented abstractions modelling dynamically
 * resizing lists holding objects or primitive data types such as <tt>int</tt>,
 * <tt>double</tt>, etc. It is designed to be scalable in terms of performance
 * and memory requirements.</p>
 * <p>Features include: </p>
 * <p></p>
 * <ul>
 * <li>Lists operating on objects as well as all primitive data types such as <tt>int</tt>,
 * <tt>double</tt>, etc.
 * </li>
 * <li>Compact representations</li>
 * <li>A number of general purpose list operations including: adding, inserting,
 * removing, iterating, searching, sorting, extracting ranges and copying. All
 * operations are designed to perform well on mass data.
 * </li>
 * <li>Support for quick access to list elements. This is achieved by bounds-checking
 * and non-bounds-checking accessor methods as well as zero-copy transformations
 * to primitive arrays such as <tt>int[]</tt>, <tt>double[]</tt>, etc.
 * </li>
 * <li>Allows to use high level algorithms on primitive data types without any
 * space and time overhead. Operations on primitive arrays, Colt lists and JAL
 * algorithms can freely be mixed at zero copy overhead.
 * </li>
 * </ul>
 * <p>File-based I/O can be achieved through the standard Java built-in serialization
 * mechanism. All classes implement the {@link java.io.Serializable} interface.
 * However, the toolkit is entirely decoupled from advanced I/O. It provides data
 * structures and algorithms only.
 * <p> This toolkit borrows concepts and terminology from the Javasoft <a
 * href="http://www.javasoft.com/products/jdk/1.2/docs/guide/collections/index.html">
 * Collections framework</a> written by Josh Bloch and introduced in JDK 1.2.
 * <h2>2. Introduction</h2>
 * <p>Lists are fundamental to virtually any application. Large scale resizable lists
 * are, for example, used in scientific computations, simulations database management
 * systems, to name just a few.</p>
 * <h2></h2>
 * <p>A list is a container holding elements that can be accessed via zero-based
 * indexes. Lists may be implemented in different ways (most commonly with arrays).
 * A resizable list automatically grows as elements are added. The lists of this
 * package do not automatically shrink. Shrinking needs to be triggered by explicitly
 * calling <tt>trimToSize()</tt> methods.</p>
 * <p><i>Growing policy</i>: A list implemented with arrays initially has a certain
 * <tt>initialCapacity</tt> - per default 10 elements, but customizable upon instance
 * construction. As elements are added, this capacity may nomore be sufficient.
 * When a list is automatically grown, its capacity is expanded to <tt>1.5*currentCapacity</tt>.
 * Thus, excessive resizing (involving copying) is avoided.</p>
 * <h4>Copying</h4>
 * <p>
 * <p>Any list can be copied. A copy is <i>equal</i> to the original but entirely
 * independent of the original. So changes in the copy are not reflected in the
 * original, and vice-versa.
 * <h2>3. Organization of this package</h2>
 * <p>Class naming follows the schema <tt>&lt;ElementType&gt;&lt;ImplementationTechnique&gt;List</tt>.
 * For example, we have a {@link org.apache.mahout.math.list.DoubleArrayList}, which is a list
 * holding <tt>double</tt> elements implemented with <tt>double</tt>[] arrays.
 * </p>
 * <p>The classes for lists of a given value type are derived from a common abstract
 * base class tagged <tt>Abstract&lt;ElementType&gt;</tt><tt>List</tt>. For example,
 * all lists operating on <tt>double</tt> elements are derived from
 * {@link org.apache.mahout.math.list.AbstractDoubleList},
 * which in turn is derived from an abstract base class tying together all lists
 * regardless of value type, {@link org.apache.mahout.math.list.AbstractList}. The abstract
 * base classes provide skeleton implementations for all but few methods. Experimental
 * data layouts (such as compressed, sparse, linked, etc.) can easily be implemented
 * and inherit a rich set of functionality. Have a look at the javadoc <a href="package-tree.html">tree
 * view</a> to get the broad picture.</p>
 * <h2>4. Example usage</h2>
 * <p>The following snippet fills a list, randomizes it, extracts the first half
 * of the elements, sums them up and prints the result. It is implemented entirely
 * with accessor methods.</p>
 * <table>
 * <td class="PRE">
 * <pre>
 * int s = 1000000;<br>AbstractDoubleList list = new DoubleArrayList();
 * for (int i=0; i&lt;s; i++) { list.add((double)i); }
 * list.shuffle();
 * AbstractDoubleList part = list.partFromTo(0,list.size()/2 - 1);
 * double sum = 0.0;
 * for (int i=0; i&lt;part.size(); i++) { sum += part.get(i); }
 * log.info(sum);
 * </pre>
 * </td>
 * </table>
 * <p> For efficiency, all classes provide back doors to enable getting/setting the
 * backing array directly. In this way, the high level operations of these classes
 * can be used where appropriate, and one can switch to <tt>[]</tt>-array index
 * notations where necessary. The key methods for this are <tt>public &lt;ElementType&gt;[]
 * elements()</tt> and <tt>public void elements(&lt;ElementType&gt;[])</tt>. The
 * former trustingly returns the array it internally keeps to store the elements.
 * Holding this array in hand, we can use the <tt>[]</tt>-array operator to
 * perform iteration over large lists without needing to copy the array or paying
 * the performance penalty introduced by accessor methods. Alternatively any JAL
 * algorithm (or other algorithm) can operate on the returned primitive array.
 * The latter method forces a list to internally hold a user provided array. Using
 * this approach one can avoid needing to copy the elements into the list.
 * <p>As a consequence, operations on primitive arrays, Colt lists and JAL algorithms
 * can freely be mixed at zero-copy overhead.
 * <p> Note that such special treatment certainly breaks encapsulation. This functionality
 * is provided for performance reasons only and should only be used when absolutely
 * necessary. Here is the above example in mixed notation:
 * <table>
 * <td class="PRE">
 * <pre>
 * int s = 1000000;<br>DoubleArrayList list = new DoubleArrayList(s); // list.size()==0, capacity==s
 * list.setSize(s); // list.size()==s<br>double[] values = list.elements();
 * // zero copy, values.length==s<br>for (int i=0; i&lt;s; i++) { values[i]=(double)i; }
 * list.shuffle();
 * double sum = 0.0;
 * int limit = values.length/2;
 * for (int i=0; i&lt;limit; i++) { sum += values[i]; }
 * log.info(sum);
 * </pre>
 * </td>
 * </table>
 * <p> Or even more compact using lists as algorithm objects:
 * <table>
 * <td class="PRE">
 * <pre>
 * int s = 1000000;<br>double[] values = new double[s];
 * for (int i=0; i&lt;s; i++) { values[i]=(double)i; }
 * new DoubleArrayList(values).shuffle(); // zero-copy, shuffle via back door
 * double sum = 0.0;
 * int limit = values.length/2;
 * for (int i=0; i&lt;limit; i++) { sum += values[i]; }
 * log.info(sum);
 * </pre>
 * </td>
 * </table>
 * <p>
 * <h2>5. Notes </h2>
 * <p>The quicksorts and mergesorts are the JDK 1.2 V1.26 algorithms, modified as
 * necessary to operate on the given data types.
 * </BODY>
 * </HTML>
 */
package org.apache.mahout.math.list;
