
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

#include "CheckPair.H"

#include "MDParticleContainer.H"

namespace amrex {
template <typename T, typename S> 
std::ostream& operator<<(std::ostream& os, const std::pair<T, S>& v) 
{ 
    os << "("; 
    os << v.first << ", " 
       << v.second << ")"; 
    return os; 
} 
}

using namespace amrex;

struct TestParams
{
    IntVect size;
    int max_grid_size;
    int num_ppc;
    bool write_particles;
};

void main_main();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    main_main();
    amrex::Finalize();
}

void get_test_params(TestParams& params)
{
    ParmParse pp;
    pp.get("size", params.size);
    pp.get("max_grid_size", params.max_grid_size);
    pp.get("write_particles", params.write_particles);
    pp.get("num_ppc", params.num_ppc);
}

void main_main ()
{
    BL_PROFILE("main::main()");
    TestParams params;
    get_test_params(params);

    RealBox real_box;
    for (int n = 0; n < BL_SPACEDIM; n++)
    {
        real_box.setLo(n, 0.0);
        real_box.setHi(n, params.size[n]);
    }

    IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
    IntVect domain_hi(AMREX_D_DECL(params.size[0]-1,params.size[1]-1,params.size[2]-1));
    const Box domain(domain_lo, domain_hi);

    int coord = 0;
    int is_per[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; i++)
        is_per[i] = 1;
    Geometry geom(domain, &real_box, coord, is_per);
    
    BoxArray ba(domain);
    ba.maxSize(params.max_grid_size);
    DistributionMapping dm(ba);

    const int ncells = 1;
    MDParticleContainer pc(geom, dm, ba, ncells);

    int npc = params.num_ppc;
    IntVect nppc = IntVect(AMREX_D_DECL(npc, npc, npc));

    if (ParallelDescriptor::MyProc() == dm[0])
        amrex::Print() << "About to initialize particles \n";

    pc.InitParticles(nppc, 1.0, 0.0);

    if (ParallelDescriptor::MyProc() == dm[0])
        amrex::Print() << "Check neighbors after init ... \n";
    pc.checkNeighbors();

    pc.fillNeighbors();

    if (ParallelDescriptor::MyProc() == dm[0])
        amrex::Print() << "Check neighbors after fill ... \n";
    pc.checkNeighbors();

    pc.updateNeighbors();

    if (ParallelDescriptor::MyProc() == dm[0])
        amrex::Print() << "Check neighbors after update ... \n";
    pc.checkNeighbors();

    if (ParallelDescriptor::MyProc() == dm[0])
        amrex::Print() << "Now resetting the particle test_id values  \n";
    pc.reset_test_id();

    if (ParallelDescriptor::MyProc() == dm[0])
        amrex::Print() << "Check neighbors after reset ... \n";
    pc.checkNeighbors();

    if (ParallelDescriptor::MyProc() == dm[0])
        amrex::Print() << "Now updateNeighbors again ...  \n";
    pc.updateNeighbors();

    if (ParallelDescriptor::MyProc() == dm[0])
        amrex::Print() << "Check neighbors after update ... \n";
    pc.checkNeighbors();

    ParallelDescriptor::Barrier();

    amrex::Print() << "Testing neighbor particles after move \n";

    // so we can call minAndMaxDistance
    pc.buildNeighborList(CheckPair());

    amrex::Print() << "Min distance is " << pc.minAndMaxDistance() << ", should be (1, 1) \n"; 

    amrex::Print() << "Moving particles and updating neighbors \n";
    pc.moveParticles(0.1);
    pc.updateNeighbors();

    amrex::Print() << "Min distance is " << pc.minAndMaxDistance() << ", should be (1, 1) \n"; 

    amrex::Print() << "Moving particles and updating neighbors again \n";
    pc.moveParticles(0.1);
    pc.updateNeighbors();

    amrex::Print() << "Min distance is " << pc.minAndMaxDistance() << ", should be (1, 1) \n"; 

    amrex::Print() << "Moving particles and updating neighbors yet again \n";
    pc.moveParticles(0.1);
    pc.updateNeighbors();

    amrex::Print() << "Min distance is " << pc.minAndMaxDistance() << ", should be (1, 1) \n"; 
    
    if (params.write_particles) 
        pc.writeParticles(0);
}