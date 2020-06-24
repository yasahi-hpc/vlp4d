#ifndef VLP4D_VIEW__ND_H
#define VLP4D_VIEW__ND_H

#include <cassert>
#include <vector>

#include "array_utils.h"

template <size_t ND>
using coord_t = std::array<size_t, ND>;

template <size_t ND>
using shape_t = std::array<size_t, ND>;

template <typename T, size_t ND>
class view_nd;

template <typename T, size_t ND, size_t SND>
class subview_nd
{
public:
    using type = subview_nd<T, ND, SND>;

    using view_t = view_nd<T, ND>;

    using subcoord_t = coord_t<ND - SND>;

private:
    const view_t& m_parent;

    const subcoord_t m_coord;

    inline constexpr subview_nd(const view_t& p, subcoord_t c) noexcept : m_parent(p), m_coord(c) {}

public:
    static inline constexpr subview_nd<T, ND, SND> view(const view_t& parent, subcoord_t coord) noexcept;

    inline constexpr typename subview_nd<T, ND, SND - 1>::type operator[](size_t ii) const noexcept;
};

template <typename T, size_t ND>
class subview_nd<T, ND, 0>
{
public:
    using type = T&;

    using view_t = view_nd<T, ND>;

    using subcoord_t = coord_t<ND>;

    static inline constexpr T& view(const view_t& parent, subcoord_t coord) noexcept;
};

template <typename T, size_t ND>
class view_nd
{
private:
    /// The raw data
    T* m_data;

    /** The shape of m_dat 
     * from m_shape[0] number of points in the largest stride dimension
     * to m_shape[ND-1] number of points in the contiguous dimension
     */
    shape_t<ND> m_shape;

public:
    inline constexpr view_nd() noexcept : m_data(nullptr), m_shape{0} {}

    inline constexpr view_nd(shape_t<ND> shape, T* data) noexcept : m_data(data), m_shape(shape) {}

    template <typename OT>
    inline constexpr view_nd(const view_nd<OT, ND>& o) noexcept : m_data(o.raw()), m_shape(o.shape())
    {}

    inline view_nd operator=(const view_nd& o) noexcept
    {
        m_data = o.raw();
        m_shape = o.shape();
        return *this;
    }

    template <typename OT>

    inline view_nd operator=(const view_nd<OT, ND>& o) noexcept
    {
        m_data = o.raw();
        m_shape = o.shape();
    }

    template <typename... S>
    inline constexpr T& operator()(S... ii) const noexcept;

    inline constexpr T& operator[](coord_t<ND>) const noexcept;

    inline constexpr typename subview_nd<T, ND, ND - 1>::type operator[](size_t ii) const noexcept { return subview_nd<T, ND, ND - 1>::view(*this, {ii}); }

    inline constexpr T* raw() const noexcept { return m_data; }

    inline constexpr const shape_t<ND>& shape() const noexcept { return m_shape; }

    inline constexpr size_t shape(size_t ii) const noexcept { return m_shape[ii]; }
};

template <typename T, size_t ND, size_t SND>
inline constexpr subview_nd<T, ND, SND> subview_nd<T, ND, SND>::view(const view_t& parent, subcoord_t coord) noexcept
{
    return subview_nd<T, ND, SND>{parent, coord};
}

template <typename T, size_t ND>
inline constexpr T& subview_nd<T, ND, 0>::view(const view_t& parent, subcoord_t coord) noexcept
{
    return parent[coord];
}

template <typename T, size_t ND, size_t SND>
inline constexpr typename subview_nd<T, ND, SND - 1>::type subview_nd<T, ND, SND>::operator[](size_t ii) const noexcept
{
    return subview_nd<T, ND, SND - 1>::view(m_parent, array_append(m_coord, ii));
}

template <typename T, size_t ND>
template <typename... S>
inline constexpr T& view_nd<T, ND>::operator()(S... ii) const noexcept
{
    return (*this)[coord_t<sizeof...(ii)>{{static_cast<size_t>(ii)...}}];
}

template <size_t ND, size_t IT = ND - 1>
struct coord
{
    static inline constexpr int linearize(const coord_t<ND>& c, const shape_t<ND>& s) noexcept
    {
#ifndef NO_ASSERT_IN_CONSTEXPR
        assert(c[IT] < s[ND - IT - 1]);
#endif
        return c[IT] + s[ND - IT - 1] * coord<ND, IT - 1>::linearize(c, s);
    }
};

template <size_t ND>
struct coord<ND, 0>
{
    static inline constexpr int linearize(const coord_t<ND>& c, const shape_t<ND>& s) noexcept
    {
#ifndef NO_ASSERT_IN_CONSTEXPR
        assert(c[0] < s[ND - 1]);
#endif
        return c[0];
    }
};

template <typename T, size_t ND>
inline constexpr T& view_nd<T, ND>::operator[](coord_t<ND> ii) const noexcept
{
    return m_data[coord<ND>::linearize(ii, m_shape)];
}

template <typename T, size_t ND>
void allocate(view_nd<T, ND>& v, shape_t<ND> shape)
{
    size_t sz = 1;
    for(auto&& dim : shape)
        sz *= dim;
    v = view_nd<T, ND>(shape, new T[sz]);
}

template <typename T, size_t ND>
void deallocate(view_nd<T, ND> v)
{
    delete[] v.raw();
}

typedef view_nd<double, 1> view_1d;
typedef view_nd<double, 2> view_2d;
typedef view_nd<double, 4> view_4d;
typedef const view_nd<const double, 1> cview_1d;
typedef const view_nd<const double, 2> cview_2d;
typedef const view_nd<const double, 4> cview_4d;

#endif // VLP4D_VIEW__ND_H
