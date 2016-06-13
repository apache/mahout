#ifndef VIENNACL_MISC_GIBBS_POOLE_STOCKMEYER_HPP
#define VIENNACL_MISC_GIBBS_POOLE_STOCKMEYER_HPP

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


/** @file viennacl/misc/gibbs_poole_stockmeyer.hpp
 *  @brief Implementation of the Gibbs-Poole-Stockmeyer algorithm.  Experimental.
 *
 *  Contributed by Philipp Grabenweger, interface adjustments by Karl Rupp.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <deque>
#include <cmath>

#include "viennacl/forwards.h"

#include "viennacl/misc/cuthill_mckee.hpp"

namespace viennacl
{
namespace detail
{

  // calculates width of a node layering
  inline int calc_layering_width(std::vector< std::vector<int> > const & l)
  {
    int w;

    w = 0;
    for (vcl_size_t i = 0; i < l.size(); i++)
    {
      w = std::max(w, static_cast<int>(l[i].size()));
    }

    return w;
  }

  // function to decompose a list of nodes rg into connected components
  // sorted by decreasing number of nodes per component
  template<typename MatrixType>
  std::vector< std::vector<int> > gps_rg_components(MatrixType const & matrix, int n,
                                                    std::vector<int> const & rg)
  {
    std::vector< std::vector<int> > rgc;
    std::vector< std::vector<int> > rgc_sorted;
    std::vector< std::vector<int> > sort_ind;
    std::vector<int> ind(2);
    std::vector<int> tmp;
    int c;
    std::vector<bool> inr(static_cast<vcl_size_t>(n), true);
    std::deque<int> q;

    for (vcl_size_t i = 0; i < rg.size(); i++)
    {
      inr[static_cast<vcl_size_t>(rg[i])] = false;
    }

    do
    {
      for (int i = 0; i < n; i++)
      {
        if (!inr[static_cast<vcl_size_t>(i)])
        {
          q.push_front(i);
          break;
        }
      }
      if (q.size() == 0)
        break;

      tmp.resize(0);
      while (q.size() > 0)
      {
        c = q.front();
        q.pop_front();

        if (!inr[static_cast<vcl_size_t>(c)])
        {
          tmp.push_back(c);
          inr[static_cast<vcl_size_t>(c)] = true;

          for (typename MatrixType::value_type::const_iterator it = matrix[static_cast<vcl_size_t>(c)].begin(); it != matrix[static_cast<vcl_size_t>(c)].end(); it++)
          {
            if (it->first == c) continue;
            if (inr[static_cast<vcl_size_t>(it->first)]) continue;

            q.push_back(it->first);
          }
        }
      }
      rgc.push_back(tmp);
    } while (true);

    for (vcl_size_t i = 0; i < rgc.size(); i++)
    {
      ind[0] = static_cast<int>(i);
      ind[1] = static_cast<int>(rgc[i].size());
      sort_ind.push_back(ind);
    }
    std::sort(sort_ind.begin(), sort_ind.end(), detail::cuthill_mckee_comp_func);
    for (vcl_size_t i = 0; i < rgc.size(); i++)
      rgc_sorted.push_back(rgc[static_cast<vcl_size_t>(sort_ind[rgc.size()-1-i][0])]);

    return rgc_sorted;
  }

} // namespace detail


/** @brief Tag class for identifying the Gibbs-Poole-Stockmeyer algorithm for reducing the bandwidth of a sparse matrix. */
struct gibbs_poole_stockmeyer_tag {};


/** @brief Function for the calculation of a node numbering permutation vector to reduce the bandwidth of a incidence matrix by the Gibbs-Poole-Stockmeyer algorithm
 *
 * references:
 *   Werner Neudorf: "Bandbreitenreduktion - Teil 3. Algorithmus von
 *   Gibbs-Poole-Stockmeyer. Testbeispiele mit CM und GPS", Preprint No.
 *   M 08/02, September 2002. Technische Universität Ilmenau, Fakultät
 *   für Mathematik und Naturwissenschaften, Institut für Mathematik.
 *   http://www.db-thueringen.de/servlets/DerivateServlet/Derivate-8673/IfM_Preprint_M_02_08.pdf
 *   (URL taken on June 14, 2011)
 *
 * @param matrix  vector of n matrix rows, where each row is a map<int, double> containing only the nonzero elements
 * @return permutation vector r. r[l] = i means that the new label of node i will be l.
 */
template<typename MatrixType>
std::vector<int> reorder(MatrixType const & matrix,
                         gibbs_poole_stockmeyer_tag)
{
  vcl_size_t n = matrix.size();
  std::vector<int> r(n);
  std::vector< std::vector<int> > rl;
  vcl_size_t l = 0;
  int state;
  bool state_end;
  std::vector< std::vector<int> > nodes;
  std::vector<bool> inr(n, false);
  std::vector<bool> isn(n, false);
  std::vector<int> tmp(2);
  int g = 0;
  int h = 0;
  std::vector< std::vector<int> > lg;
  std::vector< std::vector<int> > lh;
  std::vector< std::vector<int> > ls;
  std::map< int, std::vector<int> > lap;
  std::vector<int> rg;
  std::vector< std::vector<int> > rgc;
  int m;
  int m_min;
  bool new_g = true;
  int k1, k2, k3, k4;
  std::vector<int> wvs;
  std::vector<int> wvsg;
  std::vector<int> wvsh;
  int deg_min;
  int deg;
  int ind_min = 0;

  nodes.reserve(n);

  int current_dof = 0;

  while (current_dof < static_cast<int>(n)) // for all components of the graph apply GPS algorithm
  {
    // determine node g with mimimal degree among all nodes which
    // are not yet in result array r
    deg_min = -1;
    for (vcl_size_t i = 0; i < n; i++)
    {
      if (!inr[i])
      {
        deg = static_cast<int>(matrix[i].size() - 1); // node degree
        if (deg_min < 0 || deg < deg_min)
        {
          g = static_cast<int>(i); // node number
          deg_min = deg;
        }
      }
    }

    // algorithm for determining nodes g, h as endpoints of a pseudo graph diameter
    while (new_g)
    {
      lg.clear();
      detail::generate_layering(matrix, lg, g);

      nodes.resize(0);
      for (vcl_size_t i = 0; i < lg.back().size(); i++)
      {
        tmp[0] = lg.back()[i];
        tmp[1] = static_cast<int>(matrix[static_cast<vcl_size_t>(lg.back()[i])].size() - 1);
        nodes.push_back(tmp);
      }
      std::sort(nodes.begin(), nodes.end(), detail::cuthill_mckee_comp_func);
      for (vcl_size_t i = 0; i < nodes.size(); i++)
      {
        lg.back()[i] = nodes[i][0];
      }

      m_min = -1;
      new_g = false;
      for (vcl_size_t i = 0; i < lg.back().size(); i++)
      {
        lh.clear();
        detail::generate_layering(matrix, lh, lg.back()[i]);
        if (lh.size() > lg.size())
        {
          g = lg.back()[i];
          new_g = true;
          break;
        }
        m = detail::calc_layering_width(lh);
        if (m_min < 0 || m < m_min)
        {
          m_min = m;
          h = lg.back()[i];
        }
      }
    }

    lh.clear();
    detail::generate_layering(matrix, lh, h);

    // calculate ls as layering intersection and rg as remaining
    // graph
    lap.clear();
    for (vcl_size_t i = 0; i < lg.size(); i++)
    {
      for (vcl_size_t j = 0; j < lg[i].size(); j++)
      {
        lap[lg[i][j]].resize(2);
        lap[lg[i][j]][0] = static_cast<int>(i);
      }
    }
    for (vcl_size_t i = 0; i < lh.size(); i++)
      for (vcl_size_t j = 0; j < lh[i].size(); j++)
        lap[lh[i][j]][1] = static_cast<int>(lg.size() - 1 - i);

    rg.clear();
    ls.clear();
    ls.resize(lg.size());
    for (std::map< int, std::vector<int> >::iterator it = lap.begin();
         it != lap.end(); it++)
    {
      if ((it->second)[0] == (it->second)[1])
        ls[static_cast<vcl_size_t>((it->second)[0])].push_back(it->first);
      else
        rg.push_back(it->first);
    }
    // partition remaining graph in connected components
    rgc = detail::gps_rg_components(matrix, static_cast<int>(n), rg);

    // insert nodes of each component of rgc
    k1 = detail::calc_layering_width(lg);
    k2 = detail::calc_layering_width(lh);
    wvs.resize(ls.size());
    wvsg.resize(ls.size());
    wvsh.resize(ls.size());
    for (vcl_size_t i = 0; i < rgc.size(); i++)
    {
      for (vcl_size_t j = 0; j < ls.size(); j++)
      {
        wvs[j]  = static_cast<int>(ls[j].size());
        wvsg[j] = static_cast<int>(ls[j].size());
        wvsh[j] = static_cast<int>(ls[j].size());
      }
      for (vcl_size_t j = 0; j < rgc[i].size(); j++)
      {
        (wvsg[static_cast<vcl_size_t>(lap[rgc[i][j]][0])])++;
        (wvsh[static_cast<vcl_size_t>(lap[rgc[i][j]][1])])++;
      }
      k3 = 0;
      k4 = 0;
      for (vcl_size_t j = 0; j < ls.size(); j++)
      {
        if (wvsg[j] > wvs[j])
          k3 = std::max(k3, wvsg[j]);
        if (wvsh[j] > wvs[j])
          k4 = std::max(k4, wvsh[j]);
      }
      if (k3 < k4 || (k3 == k4 && k1 <= k2) )
        for (vcl_size_t j = 0; j < rgc[i].size(); j++)
          ls[static_cast<vcl_size_t>(lap[rgc[i][j]][0])].push_back(rgc[i][j]);
      else
        for (vcl_size_t j = 0; j < rgc[i].size(); j++)
          ls[static_cast<vcl_size_t>(lap[rgc[i][j]][1])].push_back(rgc[i][j]);
    }

    // renumber nodes in ls
    rl.clear();
    rl.resize(ls.size());
    state = 1;
    state_end = false;
    while (!state_end)
    {
      switch (state)
      {
      case 1:
        l = 0;
        state = 4;
        break;

      case 2:
        for (vcl_size_t i = 0; i < rl[l-1].size(); i++)
        {
          isn.assign(n, false);
          for (std::map<int, double>::const_iterator it  = matrix[static_cast<vcl_size_t>(rl[l-1][i])].begin();
               it != matrix[static_cast<vcl_size_t>(rl[l-1][i])].end();
               it++)
          {
            if (it->first == rl[l-1][i]) continue;
            isn[static_cast<vcl_size_t>(it->first)] = true;
          }
          nodes.resize(0);
          for (vcl_size_t j = 0; j < ls[l].size(); j++)
          {
            if (inr[static_cast<vcl_size_t>(ls[l][j])]) continue;
            if (!isn[static_cast<vcl_size_t>(ls[l][j])]) continue;
            tmp[0] = ls[l][j];
            tmp[1] = static_cast<int>(matrix[static_cast<vcl_size_t>(ls[l][j])].size() - 1);
            nodes.push_back(tmp);
          }
          std::sort(nodes.begin(), nodes.end(), detail::cuthill_mckee_comp_func);
          for (vcl_size_t j = 0; j < nodes.size(); j++)
          {
            rl[l].push_back(nodes[j][0]);
            r[static_cast<vcl_size_t>(nodes[j][0])] = current_dof++;
            inr[static_cast<vcl_size_t>(nodes[j][0])] = true;
          }
        }

      case 3:
        for (vcl_size_t i = 0; i < rl[l].size(); i++)
        {
          isn.assign(n, false);
          for (std::map<int, double>::const_iterator it = matrix[static_cast<vcl_size_t>(rl[l][i])].begin();
               it != matrix[static_cast<vcl_size_t>(rl[l][i])].end();
               it++)
          {
            if (it->first == rl[l][i]) continue;
            isn[static_cast<vcl_size_t>(it->first)] = true;
          }
          nodes.resize(0);
          for (vcl_size_t j = 0; j < ls[l].size(); j++)
          {
            if (inr[static_cast<vcl_size_t>(ls[l][j])]) continue;
            if (!isn[static_cast<vcl_size_t>(ls[l][j])]) continue;
            tmp[0] = ls[l][j];
            tmp[1] = static_cast<int>(matrix[static_cast<vcl_size_t>(ls[l][j])].size() - 1);
            nodes.push_back(tmp);
          }
          std::sort(nodes.begin(), nodes.end(), detail::cuthill_mckee_comp_func);
          for (vcl_size_t j = 0; j < nodes.size(); j++)
          {
            rl[l].push_back(nodes[j][0]);
            r[static_cast<vcl_size_t>(nodes[j][0])] = current_dof++;
            inr[static_cast<vcl_size_t>(nodes[j][0])] = true;
          }
        }

      case 4:
        if (rl[l].size() < ls[l].size())
        {
          deg_min = -1;
          for (vcl_size_t j = 0; j < ls[l].size(); j++)
          {
            if (inr[static_cast<vcl_size_t>(ls[l][j])]) continue;
            deg = static_cast<int>(matrix[static_cast<vcl_size_t>(ls[l][j])].size() - 1);
            if (deg_min < 0 || deg < deg_min)
            {
              ind_min = ls[l][j];
              deg_min = deg;
            }
          }
          rl[l].push_back(ind_min);
          r[static_cast<vcl_size_t>(ind_min)] = current_dof++;
          inr[static_cast<vcl_size_t>(ind_min)] = true;
          state = 3;
          break;
        }

      case 5:
        l++;
        if (l < ls.size())
          state = 2;
        else
          state_end = true;
        break;

      default:
        break;
      }
    }

  }

  return r;
}

} //namespace viennacl


#endif
