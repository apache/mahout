/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.list.adapter;

import org.apache.mahout.math.list.AbstractIntList;

import java.util.AbstractList;
/**
 * Adapter that permits an {@link org.apache.mahout.math.list.AbstractIntList} to be viewed and treated as a JDK 1.2 {@link java.util.AbstractList}.
 * Makes the contained list compatible with the JDK 1.2 Collections Framework.
 * <p>
 * Any attempt to pass elements other than <tt>java.lang.Number</tt> to setter methods will throw a <tt>java.lang.ClassCastException</tt>.
 * <tt>java.lang.Number.intValue()</tt> is used to convert objects into primitive values which are then stored in the backing templated list.
 * Getter methods return <tt>java.lang.Integer</tt> objects.
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class IntListAdapter extends AbstractList<Integer> {

  private final AbstractIntList content;

  /** Constructs a list backed by the specified content list. */
  public IntListAdapter(AbstractIntList content) {
    this.content = content;
  }

  /**
   * Inserts the specified element at the specified position in this list (optional operation).  Shifts the element
   * currently at that position (if any) and any subsequent elements to the right (adds one to their indices).<p>
   *
   * @param index   index at which the specified element is to be inserted.
   * @param element element to be inserted.
   * @throws ClassCastException        if the class of the specified element prevents it from being added to this list.
   * @throws IllegalArgumentException  if some aspect of the specified element prevents it from being added to this
   *                                   list.
   * @throws IndexOutOfBoundsException index is out of range (<tt>index &lt; 0 || index &gt; size()</tt>).
   */
  @Override
  public void add(int index, Integer element) {
    content.beforeInsert(index, element);
    modCount++;
  }

  /**
   * Returns the element at the specified position in this list.
   *
   * @param index index of element to return.
   * @return the element at the specified position in this list.
   * @throws IndexOutOfBoundsException if the given index is out of range (<tt>index &lt; 0 || index &gt;=
   *                                   size()</tt>).
   */
  @Override
  public Integer get(int index) {
    return content.get(index);
  }

  /**
   * Removes the element at the specified position in this list (optional operation).  Shifts any subsequent elements to
   * the left (subtracts one from their indices).  Returns the element that was removed from the list.<p>
   *
   * @param index the index of the element to remove.
   * @return the element previously at the specified position.
   * @throws IndexOutOfBoundsException if the specified index is out of range (<tt>index &lt; 0 || index &gt;=
   *                                   size()</tt>).
   */
  @Override
  public Integer remove(int index) {
    Integer old = get(index);
    content.remove(index);
    modCount++;
    return old;
  }

  /**
   * Replaces the element at the specified position in this list with the specified element (optional operation). <p>
   *
   * @param index   index of element to replace.
   * @param element element to be stored at the specified position.
   * @return the element previously at the specified position.
   * @throws ClassCastException        if the class of the specified element prevents it from being added to this list.
   * @throws IllegalArgumentException  if some aspect of the specified element prevents it from being added to this
   *                                   list.
   * @throws IndexOutOfBoundsException if the specified index is out of range (<tt>index &lt; 0 || index &gt;=
   *                                   size()</tt>).
   */

  @Override
  public Integer set(int index, Integer element) {
    Integer old = get(index);
    content.set(index, element);
    return old;
  }

  /**
   * Returns the number of elements in this list.
   *
   * @return the number of elements in this list.
   */
  @Override
  public int size() {
    return content.size();
  }

}
