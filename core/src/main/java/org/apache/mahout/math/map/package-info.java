/**
 * <HTML>
 * <BODY>
 * Automatically growing and shrinking maps holding objects or primitive
 * data types such as <tt>int</tt>, <tt>double</tt>, etc. Currently all maps are
 * based upon hashing.
 * <h2><a name="Overview"></a>1. Overview</h2>
 * <p>The map package offers flexible object oriented abstractions modelling automatically
 * resizing maps. It is designed to be scalable in terms of performance and memory
 * requirements.</p>
 * <p>Features include: </p>
 * <p></p>
 * <ul>
 * <li>Maps operating on objects as well as all primitive data types such as <code>int</code>,
 * <code>double</code>, etc.
 * </li>
 * <li>Compact representations</li>
 * <li>Support for quick access to associations</li>
 * <li>A number of general purpose map operations</li>
 * </ul>
 * <p>File-based I/O can be achieved through the standard Java built-in serialization
 * mechanism. All classes implement the {@link java.io.Serializable} interface.
 * However, the toolkit is entirely decoupled from advanced I/O. It provides data
 * structures and algorithms only.
 * <p> This toolkit borrows some terminology from the Javasoft <a
 * href="http://www.javasoft.com/products/jdk/1.2/docs/guide/collections/index.html">
 * Collections framework</a> written by Josh Bloch and introduced in JDK 1.2.
 * <h2>2. Introduction</h2>
 * <p>A map is an associative container that manages a set of (key,value) pairs.
 * It is useful for implementing a collection of one-to-one mappings. A (key,value)
 * pair is called an <i>association</i>. A value can be looked up up via its key.
 * Associations can quickly be set, removed and retrieved. They are stored in a
 * hashing structure based on the hash code of their keys, which is obtained by
 * using a hash function. </p>
 * <p> A map can, for example, contain <tt>Name-->Location</tt> associations like
 * <tt>{("Pete", "Geneva"), ("Steve", "Paris"), ("Robert", "New York")}</tt> used
 * in address books or <tt>Index-->Value</tt> mappings like <tt>{(0, 100), (3,
 * 1000), (100000, 70)}</tt> representing sparse lists or matrices. For example
 * this could mean at index 0 we have a value of 100, at index 3 we have a value
 * of 1000, at index 1000000 we have a value of 70, and at all other indexes we
 * have a value of, say, zero. Another example is a map of IP addresses to domain
 * names (DNS). Maps can also be useful to represent<i> multi sets</i>, that is,
 * sets where elements can occur more than once. For multi sets one would have
 * <tt>Value-->Frequency</tt> mappings like <tt>{(100, 1), (50, 1000), (101, 3))}</tt>
 * meaning element 100 occurs 1 time, element 50 occurs 1000 times, element 101
 * occurs 3 times. Further, maps can also manage <tt>ObjectIdentifier-->Object</tt>
 * mappings like <tt>{(12, obj1), (7, obj2), (10000, obj3), (9, obj4)}</tt> used
 * in Object Databases.
 * <p> A map cannot contain two or more <i>equal</i> keys; a key can map to at most
 * one value. However, more than one key can map to identical values. For primitive
 * data types "equality" of keys is defined as identity (operator <tt>==</tt>).
 * For maps using <tt>Object</tt> keys, the meaning of "equality" can be specified
 * by the user upon instance construction. It can either be defined to be identity
 * (operator <tt>==</tt>) or to be given by the method {@link java.lang.Object#equals(Object)}.
 * Associations of kind <tt>(AnyType,Object)</tt> can be of the form <tt>(AnyKey,null)
 * </tt>, i.e. values can be <tt>null</tt>.
 * <p> The classes of this package make no guarantees as to the order of the elements
 * returned by iterators; in particular, they do not guarantee that the order will
 * remain constant over time.
 * <h2></h2>
 * <h4>Copying</h4>
 * <p>
 * <p>Any map can be copied. A copy is <i>equal</i> to the original but entirely
 * independent of the original. So changes in the copy are not reflected in the
 * original, and vice-versa.
 * <h2>3. Package organization</h2>
 * <p>For most primitive data types and for objects there exists a separate map version.
 * All versions are just the same, except that they operate on different data types.
 * Colt includes two kinds of implementations for maps: The two different implementations
 * are tagged <b>Chained</b> and <b>Open</b>.
 * Note: Chained is no more included. Wherever it is mentioned it is of historic interest only.</p>
 * <ul>
 * <li><b>Chained</b> uses extendible separate chaining with chains holding unsorted
 * dynamically linked collision lists.
 * <li><b>Open</b> uses extendible open addressing with double hashing.
 * </ul>
 * <p>Class naming follows the schema <tt>&lt;Implementation&gt;&lt;KeyType&gt;&lt;ValueType&gt;HashMap</tt>.
 * For example, a {@link org.apache.mahout.math.map.OpenIntDoubleHashMap} holds <tt>(int-->double)</tt>
 * associations and is implemented with open addressing. A {@link org.apache.mahout.math.map.OpenIntObjectHashMap}
 * holds <tt>(int-->Object)</tt> associations and is implemented with open addressing.
 * </p>
 * <p>The classes for maps of a given (key,value) type are derived from a common
 * abstract base class tagged <tt>Abstract&lt;KeyType&gt;&lt;ValueType&gt;</tt><tt>Map</tt>.
 * For example, all maps operating on <tt>(int-->double)</tt> associations are
 * derived from {@link org.apache.mahout.math.map.AbstractIntDoubleMap}, which in turn is derived
 * from an abstract base class tying together all maps regardless of assocation
 * type, {@link org.apache.mahout.math.set.AbstractSet}. The abstract base classes provide skeleton
 * implementations for all but few methods. Experimental layouts (such as chaining,
 * open addressing, extensible hashing, red-black-trees, etc.) can easily be implemented
 * and inherit a rich set of functionality. Have a look at the javadoc <a href="package-tree.html">tree
 * view</a> to get the broad picture.</p>
 * <h2>4. Example usage</h2>
 * <TABLE>
 * <TD CLASS="PRE">
 * <PRE>
 * int[]    keys   = {0    , 3     , 100000, 9   };
 * double[] values = {100.0, 1000.0, 70.0  , 71.0};
 * AbstractIntDoubleMap map = new OpenIntDoubleHashMap();
 * // add several associations
 * for (int i=0; i &lt; keys.length; i++) map.put(keys[i], values[i]);
 * log.info("map="+map);
 * log.info("size="+map.size());
 * log.info(map.containsKey(3));
 * log.info("get(3)="+map.get(3));
 * log.info(map.containsKey(4));
 * log.info("get(4)="+map.get(4));
 * log.info(map.containsValue(71.0));
 * log.info("keyOf(71.0)="+map.keyOf(71.0));
 * // remove one association
 * map.removeKey(3);
 * log.info("\nmap="+map);
 * log.info(map.containsKey(3));
 * log.info("get(3)="+map.get(3));
 * log.info(map.containsValue(1000.0));
 * log.info("keyOf(1000.0)="+map.keyOf(1000.0));
 * // clear
 * map.clear();
 * log.info("\nmap="+map);
 * log.info("size="+map.size());
 * </PRE>
 * </TD>
 * </TABLE>
 * yields the following output
 * <TABLE>
 * <TD CLASS="PRE">
 * <PRE>
 * map=[0->100.0, 3->1000.0, 9->71.0, 100000->70.0]
 * size=4
 * true
 * get(3)=1000.0
 * false
 * get(4)=0.0
 * true
 * keyOf(71.0)=9
 * map=[0->100.0, 9->71.0, 100000->70.0]
 * false
 * get(3)=0.0
 * false
 * keyOf(1000.0)=-2147483648
 * map=[]
 * size=0
 * </PRE>
 * </TD>
 * </TABLE>
 * <h2> 5. Notes </h2>
 * <p>
 * Note that implementations are not synchronized.
 * <p>
 * Choosing efficient parameters for hash maps is not always easy.
 * However, since parameters determine efficiency and memory requirements, here is a quick guide how to choose them.
 * If your use case does not heavily operate on hash maps but uses them just because they provide
 * convenient functionality, you can safely skip this section.
 * For those of you who care, read on.
 * <p>
 * There are three parameters that can be customized upon map construction: <tt>initialCapacity</tt>,
 * <tt>minLoadFactor</tt> and <tt>maxLoadFactor</tt>.
 * The more memory one can afford, the faster a hash map.
 * The hash map's capacity is the maximum number of associations that can be added without needing to allocate new
 * internal memory.
 * A larger capacity means faster adding, searching and removing.
 * The <tt>initialCapacity</tt> corresponds to the capacity used upon instance construction.
 * <p>
 * The <tt>loadFactor</tt> of a hash map measures the degree of "fullness".
 * It is given by the number of assocations (<tt>size()</tt>)
 * divided by the hash map capacity <tt>(0.0 &lt;= loadFactor &lt;= 1.0)</tt>.
 * The more associations are added, the larger the loadFactor and the more hash map performance degrades.
 * Therefore, when the loadFactor exceeds a customizable threshold (<tt>maxLoadFactor</tt>), the hash map is
 * automatically grown.
 * In such a way performance degradation can be avoided.
 * Similarly, when the loadFactor falls below a customizable threshold (<tt>minLoadFactor</tt>), the hash map is
 * automatically shrinked.
 * In such a way excessive memory consumption can be avoided.
 * Automatic resizing (both growing and shrinking) obeys the following invariant:
 * <p>
 * <tt>capacity * minLoadFactor <= size() <= capacity * maxLoadFactor</tt>
 * <p> The term <tt>capacity * minLoadFactor</tt> is called the <i>low water mark</i>,
 * <tt>capacity * maxLoadFactor</tt> is called the <i>high water mark</i>. In other
 * words, the number of associations may vary within the water mark constraints.
 * When it goes out of range, the map is automatically resized and memory consumption
 * changes proportionally.
 * <ul>
 * <li>To tune for memory at the expense of performance, both increase <tt>minLoadFactor</tt> and
 * <tt>maxLoadFactor</tt>.
 * <li>To tune for performance at the expense of memory, both decrease <tt>minLoadFactor</tt> and
 * <tt>maxLoadFactor</tt>.
 * As as special case set <tt>minLoadFactor=0</tt> to avoid any automatic shrinking.
 * </ul>
 * Resizing large hash maps can be time consuming, <tt>O(size())</tt>, and should be avoided if possible (maintaining
 * primes is not the reason).
 * Unnecessary growing operations can be avoided if the number of associations is known before they are added, or can be
 * estimated.<p>
 * In such a case good parameters are as follows:
 * <p>
 * <i>For chaining:</i>
 * <br>Set the <tt>initialCapacity = 1.4*expectedSize</tt> or greater.
 * <br>Set the <tt>maxLoadFactor = 0.8</tt> or greater.
 * <p>
 * <i>For open addressing:</i>
 * <br>Set the <tt>initialCapacity = 2*expectedSize</tt> or greater. Alternatively call <tt>ensureCapacity(...)</tt>.
 * <br>Set the <tt>maxLoadFactor = 0.5</tt>.
 * <br>Never set <tt>maxLoadFactor &gt; 0.55</tt>; open addressing exponentially slows down beyond that point.
 * <p>
 * In this way the hash map will never need to grow and still stay fast.
 * It is never a good idea to set <tt>maxLoadFactor &lt; 0.1</tt>,
 * because the hash map would grow too often.
 * If it is entirelly unknown how many associations the application will use,
 * the default constructor should be used. The map will grow and shrink as needed.
 * <p>
 * <b>Comparision of chaining and open addressing</b>
 * <p> Chaining is faster than open addressing, when assuming unconstrained memory
 * consumption. Open addressing is more space efficient than chaining, because
 * it does not create entry objects but uses primitive arrays which are considerably
 * smaller. Entry objects consume significant amounts of memory compared to the
 * information they actually hold. Open addressing also poses no problems to the
 * garbage collector. In contrast, chaining can create millions of entry objects
 * which are linked; a nightmare for any garbage collector. In addition, entry
 * object creation is a bit slow. <br>
 * Therefore, with the same amount of memory, or even less memory, hash maps with
 * larger capacity can be maintained under open addressing, which yields smaller
 * loadFactors, which in turn keeps performance competitive with chaining. In our
 * benchmarks, using significantly less memory, open addressing usually is not
 * more than 1.2-1.5 times slower than chaining.
 * <p><b>Further readings</b>:
 * <br>Knuth D., The Art of Computer Programming: Searching and Sorting, 3rd ed.
 * <br>Griswold W., Townsend G., The Design and Implementation of Dynamic Hashing for Sets and Tables in Icon,
 * Software - Practice and Experience, Vol. 23(4), 351-367 (April 1993).
 * <br>Larson P., Dynamic hash tables, Comm. of the ACM, 31, (4), 1988.
 * <p>
 * <b>Performance:</b>
 * <p>
 * Time complexity:
 * <br>The classes offer <i>expected</i> time complexity <tt>O(1)</tt> (i.e. constant time) for the basic operations
 * <tt>put</tt>, <tt>get</tt>, <tt>removeKey</tt>, <tt>containsKey</tt> and <tt>size</tt>,
 * assuming the hash function disperses the elements properly among the buckets.
 * Otherwise, pathological cases, although highly improbable, can occur, degrading performance to <tt>O(N)</tt> in the
 * worst case.
 * Operations <tt>containsValue</tt> and <tt>keyOf</tt> are <tt>O(N)</tt>.
 * <p>
 * Memory requirements for <i>open addressing</i>:
 * <br>worst case: <tt>memory [bytes] = (1/minLoadFactor) * size() * (1 + sizeOf(key) + sizeOf(value))</tt>.
 * <br>best case: <tt>memory [bytes] = (1/maxLoadFactor) * size() * (1 + sizeOf(key) + sizeOf(value))</tt>.
 * Where <tt>sizeOf(int) = 4</tt>, <tt>sizeOf(double) = 8</tt>, <tt>sizeOf(Object) = 4</tt>, etc.
 * Thus, an <tt>OpenIntIntHashMap</tt> with minLoadFactor=0.25 and maxLoadFactor=0.5 and 1000000 associations uses
 * between 17 MB and 34 MB.
 * The same map with 1000 associations uses between 17 and 34 KB.
 * <p>
 * </BODY>
 * </HTML>
 */
package org.apache.mahout.math.map;
