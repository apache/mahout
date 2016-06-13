#ifndef VIENNACL_MISC_CUTHILL_MCKEE_HPP
#define VIENNACL_MISC_CUTHILL_MCKEE_HPP

/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */


/** @file viennacl/misc/cuthill_mckee.hpp
*    @brief Implementation of several flavors of the Cuthill-McKee algorithm.  Experimental.
*
*   Contributed by Philipp Grabenweger, interface adjustments and performance tweaks by Karl Rupp.
*/

#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <deque>
#include <cmath>

#include "viennacl/forwards.h"

namespace viennacl
{
namespace detail
{

  // Calculate the bandwidth of a reordered matrix
  template<typename IndexT, typename ValueT>
  IndexT calc_reordered_bw(std::vector< std::map<IndexT, ValueT> > const & matrix,
                           std::vector<bool> & dof_assigned_to_node,
                           std::vector<IndexT> const & permutation)
  {
    IndexT bw = 0;

    for (vcl_size_t i = 0; i < permutation.size(); i++)
    {
      if (!dof_assigned_to_node[i])
        continue;

      IndexT min_index = static_cast<IndexT>(matrix.size());
      IndexT max_index = 0;
      for (typename std::map<IndexT, ValueT>::const_iterator it = matrix[i].begin(); it != matrix[i].end(); it++)
      {
        vcl_size_t col_index = static_cast<vcl_size_t>(it->first);
        if (!dof_assigned_to_node[col_index])
          continue;

        if (permutation[col_index] > max_index)
          max_index = permutation[col_index];
        if (permutation[col_index] < min_index)
          min_index = permutation[col_index];
      }
      if (max_index > min_index)
        bw = std::max(bw, max_index - min_index);
    }

    return bw;
  }



  // function to calculate the increment of combination comb.
  // parameters:
  // comb: pointer to vector<int> of size m, m <= n
  //       1 <= comb[i] <= n for 0 <= i < m
  //       comb[i] < comb[i+1] for 0 <= i < m - 1
  //       comb represents an ordered selection of m values out of n
  // n: int
  //    total number of values out of which comb is taken as selection
  template<typename IndexT>
  bool comb_inc(std::vector<IndexT> & comb, vcl_size_t n)
  {
    vcl_size_t m;
    vcl_size_t k;

    m = static_cast<vcl_size_t>(comb.size());
    // calculate k as highest possible index such that (*comb)[k-1] can be incremented
    k = m;
    while ( (k > 0) && ( ((k == m) && (comb[k-1] == static_cast<IndexT>(n)-1)) ||
                         ((k <  m) && (comb[k-1] == comb[k] - 1) )) )
    {
      k--;
    }

    if (k == 0) // no further increment of comb possible -> return false
      return false;

    comb[k-1] += 1;

    // and all higher index positions of comb are calculated just as directly following integer values
    // Example (1, 4, 7) -> (1, 5, 6) -> (1, 5, 7) -> (1, 6, 7) -> done   for n=7
    for (vcl_size_t i = k; i < m; i++)
      comb[i] = comb[k-1] + IndexT(i - k);
    return true;
  }


  /** @brief Function to generate a node layering as a tree structure
    *
    *
    */
  // node s
  template<typename MatrixT, typename IndexT>
  void generate_layering(MatrixT const & matrix,
                         std::vector< std::vector<IndexT> > & layer_list)
  {
    std::vector<bool> node_visited_already(matrix.size(), false);

    //
    // Step 1: Set root nodes to visited
    //
    for (vcl_size_t i=0; i<layer_list.size(); ++i)
    {
      for (typename std::vector<IndexT>::iterator it  = layer_list[i].begin();
           it != layer_list[i].end();
           it++)
        node_visited_already[*it] = true;
    }

    //
    // Step 2: Fill next layers
    //
    while (layer_list.back().size() > 0)
    {
      vcl_size_t layer_index = layer_list.size();  //parent nodes are at layer 0
      layer_list.push_back(std::vector<IndexT>());

      for (typename std::vector<IndexT>::iterator it  = layer_list[layer_index].begin();
           it != layer_list[layer_index].end();
           it++)
      {
        for (typename MatrixT::value_type::const_iterator it2  = matrix[*it].begin();
             it2 != matrix[*it].end();
             it2++)
        {
          if (it2->first == *it) continue;
          if (node_visited_already[it2->first]) continue;

          layer_list.back().push_back(it2->first);
          node_visited_already[it2->first] = true;
        }
      }
    }

    // remove last (empty) nodelist:
    layer_list.resize(layer_list.size()-1);
  }


  // function to generate a node layering as a tree structure rooted at node s
  template<typename MatrixType>
  void generate_layering(MatrixType const & matrix,
                         std::vector< std::vector<int> > & l,
                         int s)
  {
    vcl_size_t n = matrix.size();
    //std::vector< std::vector<int> > l;
    std::vector<bool> inr(n, false);
    std::vector<int> nlist;

    nlist.push_back(s);
    inr[static_cast<vcl_size_t>(s)] = true;
    l.push_back(nlist);

    for (;;)
    {
      nlist.clear();
      for (std::vector<int>::iterator it  = l.back().begin();
           it != l.back().end();
           it++)
      {
        for (typename MatrixType::value_type::const_iterator it2  = matrix[static_cast<vcl_size_t>(*it)].begin();
             it2 != matrix[static_cast<vcl_size_t>(*it)].end();
             it2++)
        {
          if (it2->first == *it) continue;
          if (inr[static_cast<vcl_size_t>(it2->first)]) continue;

          nlist.push_back(it2->first);
          inr[static_cast<vcl_size_t>(it2->first)] = true;
        }
      }

      if (nlist.size() == 0)
        break;

      l.push_back(nlist);
    }

  }

  /** @brief Fills the provided nodelist with all nodes of the same strongly connected component as the nodes in the node_list
    *
    *  If more than one node is provided, all nodes should be from the same strongly connected component.
    */
  template<typename MatrixT, typename IndexT>
  void nodes_of_strongly_connected_component(MatrixT const & matrix,
                                             std::vector<IndexT> & node_list)
  {
    std::vector<bool> node_visited_already(matrix.size(), false);
    std::deque<IndexT> node_queue;

    //
    // Step 1: Push root nodes to queue:
    //
    for (typename std::vector<IndexT>::iterator it  = node_list.begin();
         it != node_list.end();
         it++)
    {
      node_queue.push_back(*it);
    }
    node_list.resize(0);

    //
    // Step 2: Fill with remaining nodes of strongly connected compontent
    //
    while (!node_queue.empty())
    {
      vcl_size_t node_id = static_cast<vcl_size_t>(node_queue.front());
      node_queue.pop_front();

      if (!node_visited_already[node_id])
      {
        node_list.push_back(IndexT(node_id));
        node_visited_already[node_id] = true;

        for (typename MatrixT::value_type::const_iterator it  = matrix[node_id].begin();
             it != matrix[node_id].end();
             it++)
        {
          vcl_size_t neighbor_node_id = static_cast<vcl_size_t>(it->first);
          if (neighbor_node_id == node_id) continue;
          if (node_visited_already[neighbor_node_id]) continue;

          node_queue.push_back(IndexT(neighbor_node_id));
        }
      }
    }

  }


  // comparison function for comparing two vector<int> values by their
  // [1]-element
  inline bool cuthill_mckee_comp_func(std::vector<int> const & a,
                                      std::vector<int> const & b)
  {
    return (a[1] < b[1]);
  }

  template<typename IndexT>
  bool cuthill_mckee_comp_func_pair(std::pair<IndexT, IndexT> const & a,
                                    std::pair<IndexT, IndexT> const & b)
  {
    return (a.second < b.second);
  }

  /** @brief Runs the Cuthill-McKee algorithm on a strongly connected component of a graph
    *
    * @param matrix                  The matrix describing the full graph
    * @param node_assignment_queue   A queue prepopulated with the root nodes
    * @param dof_assigned_to_node    Boolean flag array indicating whether a dof got assigned to a certain node
    * @param permutation             The permutation array to write the result to
    * @param current_dof             The first dof to be used for assignment
    *
    * @return The next free dof available
    */
  template<typename IndexT, typename ValueT>
  vcl_size_t cuthill_mckee_on_strongly_connected_component(std::vector< std::map<IndexT, ValueT> > const & matrix,
                                                           std::deque<IndexT> & node_assignment_queue,
                                                           std::vector<bool>  & dof_assigned_to_node,
                                                           std::vector<IndexT> & permutation,
                                                           vcl_size_t current_dof)
  {
    typedef std::pair<IndexT, IndexT> NodeIdDegreePair; //first member is the node ID, second member is the node degree

    std::vector< NodeIdDegreePair > local_neighbor_nodes(matrix.size());

    while (!node_assignment_queue.empty())
    {
      // Grab first node from queue
      vcl_size_t node_id = static_cast<vcl_size_t>(node_assignment_queue.front());
      node_assignment_queue.pop_front();

      // Assign dof if a new dof hasn't been assigned yet
      if (!dof_assigned_to_node[node_id])
      {
        permutation[node_id] = static_cast<IndexT>(current_dof);  //TODO: Invert this!
        ++current_dof;
        dof_assigned_to_node[node_id] = true;

        //
        // Get all neighbors of that node:
        //
        vcl_size_t num_neighbors = 0;
        for (typename std::map<IndexT, ValueT>::const_iterator neighbor_it  = matrix[node_id].begin();
             neighbor_it != matrix[node_id].end();
             ++neighbor_it)
        {
          vcl_size_t neighbor_node_index = static_cast<vcl_size_t>(neighbor_it->first);
          if (!dof_assigned_to_node[neighbor_node_index])
          {
            local_neighbor_nodes[num_neighbors] = NodeIdDegreePair(neighbor_it->first, static_cast<IndexT>(matrix[neighbor_node_index].size()));
            ++num_neighbors;
          }
        }

        // Sort neighbors by increasing node degree
        std::sort(local_neighbor_nodes.begin(),
                  local_neighbor_nodes.begin() + static_cast<typename std::vector< NodeIdDegreePair >::difference_type>(num_neighbors),
                  detail::cuthill_mckee_comp_func_pair<IndexT>);

        // Push neighbors to queue
        for (vcl_size_t i=0; i<num_neighbors; ++i)
          node_assignment_queue.push_back(local_neighbor_nodes[i].first);

      } // if node doesn't have a new dof yet

    } // while nodes in queue

    return current_dof;

  }

} //namespace detail

//
// Part 1: The original Cuthill-McKee algorithm
//

/** @brief A tag class for selecting the Cuthill-McKee algorithm for reducing the bandwidth of a sparse matrix. */
struct cuthill_mckee_tag {};

/** @brief Function for the calculation of a node number permutation to reduce the bandwidth of an incidence matrix by the Cuthill-McKee algorithm
 *
 * references:
 *    Algorithm was implemented similary as described in
 *      "Tutorial: Bandwidth Reduction - The CutHill-
 *      McKee Algorithm" posted by Ciprian Zavoianu as weblog at
 *    http://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html
 *    on January 15, 2009
 *    (URL taken on June 14, 2011)
 *
 * @param matrix  vector of n matrix rows, where each row is a map<int, double> containing only the nonzero elements
 * @return permutation vector r. r[l] = i means that the new label of node i will be l.
 *
 */
template<typename IndexT, typename ValueT>
std::vector<IndexT> reorder(std::vector< std::map<IndexT, ValueT> > const & matrix, cuthill_mckee_tag)
{
  std::vector<IndexT> permutation(matrix.size());
  std::vector<bool>   dof_assigned_to_node(matrix.size(), false);   //flag vector indicating whether node i has received a new dof
  std::deque<IndexT>  node_assignment_queue;

  vcl_size_t current_dof = 0;  //the dof to be assigned

  while (current_dof < matrix.size()) //outer loop for each strongly connected component (there may be more than one)
  {
    //
    // preprocessing: Determine node degrees for nodes which have not been assigned
    //
    vcl_size_t current_min_degree = matrix.size();
    vcl_size_t node_with_minimum_degree = 0;
    bool found_unassigned_node = false;
    for (vcl_size_t i=0; i<matrix.size(); ++i)
    {
      if (!dof_assigned_to_node[i])
      {
        if (matrix[i].size() == 1)  //This is an isolated node, so assign DOF right away
        {
          permutation[i] = static_cast<IndexT>(current_dof);
          dof_assigned_to_node[i] = true;
          ++current_dof;
          continue;
        }

        if (!found_unassigned_node) //initialize minimum degree on first node without new dof
        {
          current_min_degree = matrix[i].size();
          node_with_minimum_degree = i;
          found_unassigned_node = true;
        }

        if (matrix[i].size() < current_min_degree) //found a node with smaller degree
        {
          current_min_degree = matrix[i].size();
          node_with_minimum_degree = i;
        }
      }
    }

    //
    // Stage 2: Distribute dofs on this closely connected (sub-)graph in a breath-first manner using one root node
    //
    if (found_unassigned_node) // there's work to be done
    {
      node_assignment_queue.push_back(static_cast<IndexT>(node_with_minimum_degree));
      current_dof = detail::cuthill_mckee_on_strongly_connected_component(matrix, node_assignment_queue, dof_assigned_to_node, permutation, current_dof);
    }
  }

  return permutation;
}


//
// Part 2: Advanced Cuthill McKee
//

/** @brief Tag for the advanced Cuthill-McKee algorithm (i.e. running the 'standard' Cuthill-McKee algorithm for a couple of different seeds). */
class advanced_cuthill_mckee_tag
{
public:
  /** @brief CTOR which may take the additional parameters for the advanced algorithm.
      *
      * additional parameters for CTOR:
      *   a:  0 <= a <= 1
      *     parameter which specifies which nodes are tried as starting nodes
      *     of generated node layering (tree structure whith one ore more
      *     starting nodes).
      *     the relation deg_min <= deg <= deg_min + a * (deg_max - deg_min)
      *     must hold for node degree deg for a starting node, where deg_min/
      *     deg_max is the minimal/maximal node degree of all yet unnumbered
      *     nodes.
      *    gmax:
      *      integer which specifies maximum number of nodes in the root
      *      layer of the tree structure (gmax = 0 means no limit)
      *
      * @return permutation vector r. r[l] = i means that the new label of node i will be l.
      *
     */
  advanced_cuthill_mckee_tag(double a = 0.0, vcl_size_t gmax = 1) : starting_node_param_(a), max_root_nodes_(gmax) {}

  double starting_node_param() const { return starting_node_param_;}
  void starting_node_param(double a) { if (a >= 0) starting_node_param_ = a; }

  vcl_size_t max_root_nodes() const { return max_root_nodes_; }
  void max_root_nodes(vcl_size_t gmax) { max_root_nodes_ = gmax; }

private:
  double starting_node_param_;
  vcl_size_t max_root_nodes_;
};



/** @brief Function for the calculation of a node number permutation to reduce the bandwidth of an incidence matrix by the advanced Cuthill-McKee algorithm
 *
 *
 *  references:
 *    see description of original Cuthill McKee implementation, and
 *    E. Cuthill and J. McKee: "Reducing the Bandwidth of sparse symmetric Matrices".
 *    Naval Ship Research and Development Center, Washington, D. C., 20007
 */
template<typename IndexT, typename ValueT>
std::vector<IndexT> reorder(std::vector< std::map<IndexT, ValueT> > const & matrix,
                            advanced_cuthill_mckee_tag const & tag)
{
  vcl_size_t n = matrix.size();
  double a = tag.starting_node_param();
  vcl_size_t gmax = tag.max_root_nodes();
  std::vector<IndexT> permutation(n);
  std::vector<bool>   dof_assigned_to_node(n, false);
  std::vector<IndexT> nodes_in_strongly_connected_component;
  std::vector<IndexT> parent_nodes;
  vcl_size_t deg_min;
  vcl_size_t deg_max;
  vcl_size_t deg_a;
  vcl_size_t deg;
  std::vector<IndexT> comb;

  nodes_in_strongly_connected_component.reserve(n);
  parent_nodes.reserve(n);
  comb.reserve(n);

  vcl_size_t current_dof = 0;

  while (current_dof < matrix.size()) // for all strongly connected components
  {
    // get all nodes of the strongly connected component:
    nodes_in_strongly_connected_component.resize(0);
    for (vcl_size_t i = 0; i < n; i++)
    {
      if (!dof_assigned_to_node[i])
      {
        nodes_in_strongly_connected_component.push_back(static_cast<IndexT>(i));
        detail::nodes_of_strongly_connected_component(matrix, nodes_in_strongly_connected_component);
        break;
      }
    }

    // determine minimum and maximum node degree
    deg_min = 0;
    deg_max = 0;
    for (typename std::vector<IndexT>::iterator it  = nodes_in_strongly_connected_component.begin();
         it != nodes_in_strongly_connected_component.end();
         it++)
    {
      deg = matrix[static_cast<vcl_size_t>(*it)].size();
      if (deg_min == 0 || deg < deg_min)
        deg_min = deg;
      if (deg_max == 0 || deg > deg_max)
        deg_max = deg;
    }
    deg_a = deg_min + static_cast<vcl_size_t>(a * double(deg_max - deg_min));

    // fill array of parent nodes:
    parent_nodes.resize(0);
    for (typename std::vector<IndexT>::iterator it  = nodes_in_strongly_connected_component.begin();
         it != nodes_in_strongly_connected_component.end();
         it++)
    {
      if (matrix[static_cast<vcl_size_t>(*it)].size() <= deg_a)
        parent_nodes.push_back(*it);
    }

    //
    // backup current state in order to restore for every new combination of parent nodes below
    //
    std::vector<bool> dof_assigned_to_node_backup = dof_assigned_to_node;
    std::vector<bool> dof_assigned_to_node_best;

    std::vector<IndexT> permutation_backup = permutation;
    std::vector<IndexT> permutation_best = permutation;

    vcl_size_t current_dof_backup = current_dof;

    vcl_size_t g = 1;
    comb.resize(1);
    comb[0] = 0;

    IndexT bw_best = 0;

    //
    // Loop over all combinations of g <= gmax root nodes
    //

    for (;;)
    {
      dof_assigned_to_node = dof_assigned_to_node_backup;
      permutation          = permutation_backup;
      current_dof          = current_dof_backup;

      std::deque<IndexT>  node_queue;

      // add the selected root nodes according to actual combination comb to q
      for (typename std::vector<IndexT>::iterator it = comb.begin(); it != comb.end(); it++)
        node_queue.push_back(parent_nodes[static_cast<vcl_size_t>(*it)]);

      current_dof = detail::cuthill_mckee_on_strongly_connected_component(matrix, node_queue, dof_assigned_to_node, permutation, current_dof);

      // calculate resulting bandwith for root node combination
      // comb for current numbered component of the node graph
      IndexT bw = detail::calc_reordered_bw(matrix, dof_assigned_to_node, permutation);

      // remember best ordering:
      if (bw_best == 0 || bw < bw_best)
      {
        permutation_best = permutation;
        bw_best = bw;
        dof_assigned_to_node_best = dof_assigned_to_node;
      }

      // calculate next combination comb, if not existing
      // increment g if g stays <= gmax, or else terminate loop
      if (!detail::comb_inc(comb, parent_nodes.size()))
      {
        ++g;
        if ( (gmax > 0 && g > gmax) || g > parent_nodes.size())
          break;

        comb.resize(g);
        for (vcl_size_t i = 0; i < g; i++)
          comb[i] = static_cast<IndexT>(i);
      }
    }

    //
    // restore best permutation
    //
    permutation = permutation_best;
    dof_assigned_to_node = dof_assigned_to_node_best;

  }

  return permutation;
}


} //namespace viennacl


#endif
