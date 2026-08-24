#include <AMReX_FFT_CrossProj.H>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main");

        AMREX_D_TERM(int n_cell_x = 128;,
                     int n_cell_y = 128;,
                     int n_cell_z = 64);

        AMREX_D_TERM(int max_grid_size_x = 16;,
                     int max_grid_size_y = 16;,
                     int max_grid_size_z = 16);

        AMREX_D_TERM(Real prob_lo_x = 0.;,
                     Real prob_lo_y = 0.;,
                     Real prob_lo_z = 0.);
        AMREX_D_TERM(Real prob_hi_x = 1.;,
                     Real prob_hi_y = 1.;,
                     Real prob_hi_z = 1.);

        {
            ParmParse pp;
            AMREX_D_TERM(pp.query("n_cell_x", n_cell_x);,
                         pp.query("n_cell_y", n_cell_y);,
                         pp.query("n_cell_z", n_cell_z));
            AMREX_D_TERM(pp.query("max_grid_size_x", max_grid_size_x);,
                         pp.query("max_grid_size_y", max_grid_size_y);,
                         pp.query("max_grid_size_z", max_grid_size_z));
        }

        auto bcs = std::pair<FFT::Boundary,FFT::Boundary>{FFT::Boundary::periodic,
                                                          FFT::Boundary::periodic};

        Array<std::pair<FFT::Boundary,FFT::Boundary>,AMREX_SPACEDIM>
                fft_bc{AMREX_D_DECL(bcs,bcs,bcs)};

        Box domain(IntVect(0),IntVect(AMREX_D_DECL(n_cell_x-1,n_cell_y-1,n_cell_z-1)));
        BoxArray ba(domain);
        ba.maxSize(IntVect(AMREX_D_DECL(max_grid_size_x,
                                        max_grid_size_y,
                                        max_grid_size_z)));
        DistributionMapping dm(ba);

        Geometry geom;
        {
            geom.define(domain,
                        RealBox(AMREX_D_DECL(prob_lo_x,prob_lo_y,prob_lo_z),
                                AMREX_D_DECL(prob_hi_x,prob_hi_y,prob_hi_z)),
                        CoordSys::cartesian, {AMREX_D_DECL(1,1,1)});
        }
        auto const& dx = geom.CellSizeArray();

        MultiFab vel(ba,dm,2,1);
        auto const& u = vel.arrays();
        ParallelFor(vel, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            AMREX_D_TERM(Real x = (i+0.5_rt) * dx[0] - 0.5_rt;,
                         Real y = (j+0.5_rt) * dx[1] - 0.5_rt;,
                         Real z = (k+0.5_rt) * dx[2] - 0.5_rt);
            u[b](i,j,k,0) = std::cos(Real(2.)*Math::pi<Real>()*x)
                         + std::cos(Real(2.)*Math::pi<Real>()*y);
            u[b](i,j,k,1) = std::cos(Real(2.)*Math::pi<Real>()*x+.2)
                         + std::cos(Real(2.)*Math::pi<Real>()*y+.7);
        });

        vel.FillBoundary(geom.periodicity());

        MultiFab vel_return(ba,dm,2,1);

        {
            FFT::CrossProj crossproj(geom, fft_bc);
            crossproj.solve(vel_return,  vel );

            vel_return.FillBoundary(geom.periodicity());

            MultiFab div_orig(ba, dm, 1, 0);
            MultiFab div_err(ba, dm, 1, 0);
            auto const& u = vel.arrays();
            auto const& ud = vel_return.arrays();
            auto const& divinit = div_orig.arrays();
            auto const& div = div_err.arrays();
            ParallelFor(div_err, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
            {
             
                divinit[b](i,j,k) = u[b](i+1,j+1,k,0) + u[b](i+1,j,k,0) - u[b](i,j+1,k,0)  - u[b](i,j,k,0) 
                               +     u[b](i+1,j+1,k,1) + u[b](i,j+1,k,1) - u[b](i+1,j,k,1)  - u[b](i,j,k,1) ;
                div[b](i,j,k) = ud[b](i+1,j+1,k,0) + ud[b](i+1,j,k,0) - ud[b](i,j+1,k,0)  - ud[b](i,j,k,0) 
                               +     ud[b](i+1,j+1,k,1) + ud[b](i,j+1,k,1) - ud[b](i+1,j,k,1)  - ud[b](i,j,k,1) ;
             });


            Real div_initial = div_orig.norminf();
            Real div_error = div_err.norminf();
            amrex::Print() << "  original divergence " << div_initial << "\n";
            amrex::Print() << "  divegence expected to be close to zero: " << div_error << "\n";
        }
    }
    amrex::Finalize();
}
