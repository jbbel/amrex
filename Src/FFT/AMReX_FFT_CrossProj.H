#ifndef AMREX_FFT_CROSSPROJ_H_
#define AMREX_FFT_CROSSPROJ_H_

#include <AMReX_FFT.H>
#include <AMReX_Geometry.H>

namespace amrex::FFT
{

/**
 * \file AMReX_FFT_CrossProj.H
 *
 * FFT-based CrossProj solvers (periodic, mixed BC, hybrid, and open BC variants).
 */

/// \cond DOXYGEN_IGNORE
namespace detail {
template <typename MF>
void fill_physbc (MF& mf, Geometry const& geom,
                  Array<std::pair<Boundary,Boundary>,AMREX_SPACEDIM> const& bc);

}
/// \endcond

/**
 * \ingroup amrex_fft
 * \brief CrossProj solver for periodic, Dirichlet & Neumann boundaries using
 * FFT.
 */
template <typename MF = MultiFab>
class CrossProj
{
public:

    /**
     * \brief Construct a CrossProj solver with explicit boundary types.
     *
     * \param geom Geometry describing the domain and metric information.
     * \param bc   Pair of boundary descriptors (low/high) per coordinate direction.
     */
    CrossProj (Geometry const& geom,
             Array<std::pair<Boundary,Boundary>,AMREX_SPACEDIM> const& bc)
        requires (IsFabArray_v<MF>)
        : m_domain_lo(geom.Domain().smallEnd()),
          m_geom(detail::shift_geom(geom)),
          m_bc(bc)
    {
        bool all_periodic = true;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            all_periodic = all_periodic
                && (bc[idim].first == Boundary::periodic)
                && (bc[idim].second == Boundary::periodic);
        }
        if (all_periodic) {
            Info info = Info{};
            info.setBatchSize(2);
            m_r2c = std::make_unique<R2C<typename MF::value_type>>(m_geom.Domain(), info);
        } else {
            m_r2x = std::make_unique<R2X<typename MF::value_type>> (m_geom.Domain(), m_bc);
        }
    }

    /**
     * \brief Construct a purely periodic CrossProj solver.
     *
     * \param geom Periodic geometry (all directions must be periodic).
     */
    explicit CrossProj (Geometry const& geom)
        requires (IsFabArray_v<MF>)
        : m_domain_lo(geom.Domain().smallEnd()),
          m_geom(detail::shift_geom(geom)),
          m_bc{AMREX_D_DECL(std::make_pair(Boundary::periodic,Boundary::periodic),
                            std::make_pair(Boundary::periodic,Boundary::periodic),
                            std::make_pair(Boundary::periodic,Boundary::periodic))}
    {
        if (m_geom.isAllPeriodic()) {
            Info info = Info{};
            info.setBatchSize(2);
            m_r2c = std::make_unique<R2C<typename MF::value_type>>(m_geom.Domain());
        } else {
            amrex::Abort("FFT::CrossProj: wrong BC");
        }
    }

    /**
     * \brief Solve del dot grad soln = rhs.
     *
     * If \p soln has ghost cells, one layer of ghost cells will be filled
     * except for the corners of the physical domain where they are not used
     * in the cross stencil of the operator. The two MFs could be the same MF.
     *
     * \param a_soln Solution MultiFab (updated in place).
     * \param a_rhs  Right-hand side MultiFab.
     */
    void solve (MF& a_soln, MF const& a_rhs);

private:
    IntVect m_domain_lo;
    Geometry m_geom;
    Array<std::pair<Boundary,Boundary>,AMREX_SPACEDIM> m_bc;
    std::unique_ptr<R2X<typename MF::value_type>> m_r2x;
    std::unique_ptr<R2C<typename MF::value_type>> m_r2c;
};

template <typename MF>
void CrossProj<MF>::solve (MF& a_soln, MF const& a_rhs)
{
    BL_PROFILE("FFT::CrossProj::solve");

    MF* soln = &a_soln;
    MF const* rhs = &a_rhs;
    MF solntmp, rhstmp;
    if (m_domain_lo != 0) {
        detail::shift_mfs(m_domain_lo, a_soln, a_rhs, solntmp, rhstmp);
        soln = &solntmp;
        rhs = &rhstmp;
    }

    AMREX_ASSERT(soln->is_cell_centered() && rhs->is_cell_centered());

    using T = typename MF::value_type;

    GpuArray<T,AMREX_SPACEDIM> fac
        {AMREX_D_DECL(Math::pi<T>()/(T(m_geom.Domain().length(0))),
                      Math::pi<T>()/(T(m_geom.Domain().length(1))),
                      Math::pi<T>()/(T(m_geom.Domain().length(2))))};
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (m_bc[idim].first == Boundary::periodic) {
            fac[idim] *= T(1);
        }
    }
    GpuArray<T,AMREX_SPACEDIM> dxfac
        {AMREX_D_DECL(T(1)/T(m_geom.CellSize(0)),
                      T(1)/T(m_geom.CellSize(1)),
                      T(1)/T(m_geom.CellSize(2)))};
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (m_geom.Domain().length(idim) == 1) {
            dxfac[idim] = 0;
        }
    }
    auto scale = (m_r2x) ? m_r2x->scalingFactor() : m_r2c->scalingFactor();

    GpuArray<T,AMREX_SPACEDIM> offset{AMREX_D_DECL(T(0),T(0),T(0))};
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (m_bc[idim].first == Boundary::odd &&
            m_bc[idim].second == Boundary::odd)
        {
            offset[idim] = T(1);
        }
        else if ((m_bc[idim].first == Boundary::odd &&
                  m_bc[idim].second == Boundary::even) ||
                 (m_bc[idim].first == Boundary::even &&
                  m_bc[idim].second == Boundary::odd))
        {
            offset[idim] = T(0.5);
        }
    }

    auto f = [=] AMREX_GPU_DEVICE (int i, int j, int k, CellData < GpuComplex < T >> spectral_data)
    {
        amrex::ignore_unused(j,k);
        AMREX_D_TERM(T a = fac[0]*(i+offset[0]);,
                     T b = fac[1]*(j+offset[1]);,
                     T c = fac[2]*(k+offset[2]));

        T cdiffx = (std::sin(a+b)+std::sin(a-b))*dxfac[0];
        T cdiffy = (std::sin(a+b)-std::sin(a-b))*dxfac[1];

        auto cdiv = cdiffx * spectral_data[0] +cdiffy * spectral_data[1];;

        T dginv = cdiffx*cdiffx + cdiffy*cdiffy;
        if (std::abs(dginv) > T(1.e-10)) {
            cdiv /= dginv;;
        }
        spectral_data[0] -=  cdiffx * cdiv;
        spectral_data[1] -=  cdiffy * cdiv;
        spectral_data[0] *= scale;
        spectral_data[1] *= scale;
    };

    IntVect const& ng = amrex::elemwiseMin(soln->nGrowVect(), IntVect(1));

        m_r2c->forwardThenBackward(*rhs, *soln, f);
//        m_r2c->forward(*rhs);
//        m_r2c->post_forward_doit_0(f);
//        m_r2c->backward_doit(*soln, ng, m_geom.periodicity());
}


/// \cond DOXYGEN_IGNORE
namespace detail {

template <class T>
struct FFTPhysBCTag {
    Array4<T>   dfab;
    Box         dbox;
    Boundary    bc;
    Orientation face;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Box const& box () const noexcept { return dbox; }
};

template <typename MF>
void fill_physbc (MF& mf, Geometry const& geom,
                  Array<std::pair<Boundary,Boundary>,AMREX_SPACEDIM> const& bc)
{
    using T = typename MF::value_type;
    using Tag = FFTPhysBCTag<T>;
    Vector<Tag> tags;

    for (MFIter mfi(mf, MFItInfo{}.DisableDeviceSync()); mfi.isValid(); ++mfi)
    {
        auto const& box = mfi.fabbox();
        auto const& arr = mf.array(mfi);
        for (OrientationIter oit; oit; ++oit) {
            Orientation face = oit();
            int idim = face.coordDir();
            Box b = geom.Domain();
            Boundary fbc;
            if (face.isLow()) {
                b.setRange(idim,geom.Domain().smallEnd(idim)-1);
                fbc = bc[idim].first;
            } else {
                b.setRange(idim,geom.Domain().bigEnd(idim)+1);
                fbc = bc[idim].second;
            }
            b &= box;
            if (b.ok() && fbc != Boundary::periodic) {
                tags.push_back(Tag{.dfab = arr, .dbox = b, .bc = fbc, .face = face});
            }
        }
    }

#if defined(AMREX_USE_GPU)
    amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE (int i, int j, int k,
                                                   Tag const& tag) noexcept
#else
    auto ntags = int(tags.size());
#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
    for (int itag = 0; itag < ntags; ++itag) {
        Tag const& tag = tags[itag];
        amrex::LoopOnCpu(tag.dbox, [&] (int i, int j, int k)
#endif
        {
            int sgn = tag.face.isLow() ? 1 : -1;
            IntVect siv = IntVect(AMREX_D_DECL(i,j,k))
                +  sgn * IntVect::TheDimensionVector(tag.face.coordDir());
            if (tag.bc == Boundary::odd) {
                tag.dfab(i,j,k) = -tag.dfab(siv);
            } else { // even
                tag.dfab(i,j,k) = tag.dfab(siv);
            }
        });
#if !defined(AMREX_USE_GPU)
    }
#endif
}
}
/// \endcond

}

#endif
