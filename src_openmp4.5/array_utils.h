// this file contains a large extract from <utility> from GNU libstdc++
// git commit e4bb5efbffbbc8a617365b789933a42b66f9acfa

// Copyright (C) 2001-2017 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 *
 * Copyright (c) 1996,1997
 * Silicon Graphics Computer Systems, Inc.
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Silicon Graphics makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 */


#ifndef VLP4D_ARRAY__UTILS_H
#define VLP4D_ARRAY__UTILS_H

#include <array>
#include <utility>

namespace {

namespace implem {

#ifdef OWN_INDEX_SEQUENCE

// Stores a tuple of indices.  Used by tuple and pair, and by bind() to
// extract the elements in a tuple.
template <size_t... _Indexes>
struct _Index_tuple
{};

// Concatenates two _Index_tuples.
template <typename _Itup1, typename _Itup2>
struct _Itup_cat;

template <size_t... _Ind1, size_t... _Ind2>
struct _Itup_cat<_Index_tuple<_Ind1...>, _Index_tuple<_Ind2...>>
{
    using __type = _Index_tuple<_Ind1..., (_Ind2 + sizeof...(_Ind1))...>;
};

// Builds an _Index_tuple<0, 1, 2, ..., _Num-1>.
template <size_t _Num>
struct _Build_index_tuple : _Itup_cat<typename _Build_index_tuple<_Num / 2>::__type, typename _Build_index_tuple<_Num - _Num / 2>::__type>
{};

template <>
struct _Build_index_tuple<1>
{
    typedef _Index_tuple<0> __type;
};

template <>
struct _Build_index_tuple<0>
{
    typedef _Index_tuple<> __type;
};

/// Class template integer_sequence
template <typename _Tp, _Tp... _Idx>
struct integer_sequence
{
    typedef _Tp value_type;
    static constexpr size_t size() { return sizeof...(_Idx); }
};

template <typename _Tp, _Tp _Num, typename _ISeq = typename _Build_index_tuple<_Num>::__type>
struct _Make_integer_sequence;

template <typename _Tp, _Tp _Num, size_t... _Idx>
struct _Make_integer_sequence<_Tp, _Num, _Index_tuple<_Idx...>>
{
    static_assert(_Num >= 0, "Cannot make integer sequence of negative length");

    typedef integer_sequence<_Tp, static_cast<_Tp>(_Idx)...> __type;
};

/// Alias template make_integer_sequence
template <typename _Tp, _Tp _Num>
using make_integer_sequence = typename _Make_integer_sequence<_Tp, _Num>::__type;

/// Alias template index_sequence
template <size_t... _Idx>
using index_sequence = integer_sequence<size_t, _Idx...>;

/// Alias template make_index_sequence
template <size_t _Num>
using make_index_sequence = make_integer_sequence<size_t, _Num>;

/// Alias template index_sequence_for
template <typename... _Types>
using index_sequence_for = make_index_sequence<sizeof...(_Types)>;

#else
using std::index_sequence;
using std::make_index_sequence;
#endif
} // namespace implem

template <typename T, std::size_t SZ, std::size_t... II>
constexpr std::array<T, sizeof...(II) + 1> sub_array_append(const std::array<T, SZ>& a, implem::index_sequence<II...>, T ii)
{
    return {a[II]..., ii};
}

template <typename T, size_t S>
constexpr std::array<T, S + 1> array_append(const std::array<T, S>& a, T ii)
{
    return sub_array_append(a, implem::make_index_sequence<S>(), ii);
}

template <typename T, size_t S, size_t SS>
constexpr std::array<size_t, SS> subarray(const std::array<T, S>& a)
{
    return sub_array_append(a, implem::make_index_sequence<SS - 1>(), a[SS - 1]);
}

} // namespace

#endif // VLP4D_ARRAY__UTILS_H
