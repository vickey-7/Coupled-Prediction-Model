/*---------------------------------------------------------------------------*\

License
    This file is part of GeoChemFoam, an Open source software using OpenFOAM
    for multiphase multicomponent reactive transport simulation in pore-scale
    geological domain.

    GeoChemFoam is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version. See <http://www.gnu.org/licenses/>.

    The code was developed by Dr Julien Maes as part of his research work for
    the GeoChemFoam Group at Heriot-Watt University. Please visit our
    website for more information <https://github.com/GeoChemFoam>.

Application
    reactiveTransportDBSFoam

Description
    Solves reactive transport equation with microcontinuum DBS for multi-species flow

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "reactiveMixture.H"
#include "multiComponentTransportMixture.H"
#include "steadyStateControl.H"
#include "dynamicFvMesh.H"
#include "fvOptions.H"
#include "cpuTime.H"
#include <hdf5.h>

// ---------------- H5 Predictor ----------------
class H5Predictor
{
public:
    H5Predictor() {}

    void readH5ToFields(const word& h5File,
                        volScalarField& eps,
                        volScalarField& C)
    {
        label nCells = eps.internalField().size();
        hid_t file_id = H5Fopen(h5File.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t dataset_id = H5Dopen(file_id, "y_pred", H5P_DEFAULT);
        hsize_t dims[5];
        hid_t dataspace_id = H5Dget_space(dataset_id);
        H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
        std::vector<float> buffer(nCells * 2, 0.0);
        H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());

        scalarField& epsF = eps.ref();
        scalarField& CF   = C.ref();

        for (label i=0; i<nCells; i++)
        {
            CF[i]   = buffer[i];        
            epsF[i] = buffer[nCells+i]; 
        }

        eps.correctBoundaryConditions();
        C.correctBoundaryConditions();

        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);

    }
};

// ---------------- Main Solver ----------------
int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Steady-state solver for reactive transport with dissolving solid surface."
    );

    #include "postProcess.H"
    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"

    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"

    const speciesTable& solutionSpecies = speciesMixture.species();
    const wordList& kineticPhases = speciesMixture.kineticPhases();
    const wordList& kineticPhaseReactions = speciesMixture.kineticPhaseReactions();

    turbulence->validate();
    #include "deltaEpsMax.H"
    #include "setInitialDeltaT.H"
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    H5Predictor predictor;

    Info << "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readDyMControls.H"
        #include "CourantNo.H"
        #include "deltaEpsMax.H"
        #include "setDeltaT.H"

        ++runTime;
        Info << "Time = " << runTime.timeName() << nl << endl;
        mesh.controlledUpdate();

        if (mesh.changing())
        {
            MRF.update();

            phi = mesh.Sf() & fvc::interpolate(U);
        }

        #include "epsEqn.H"

        // permeability
        Kinv = kf*pow(1-eps,2)/pow(eps,3);

        volVectorField gradEps = fvc::grad(eps);
        surfaceVectorField gradEpsf = fvc::interpolate(gradEps);
        surfaceVectorField nEpsv = -gradEpsf/(mag(gradEpsf) + deltaN);
        nEpsf = nEpsv & mesh.Sf();

        volScalarField a = mag(fvc::grad(eps));

        scalar lambda = psiCoeff;

        if (VoS=="VoS-psi")
        {
            scalar As = a.weightedAverage(mesh.V()).value();

            a = a*(1-eps)*(1e-3+eps);

            if (adaptPsiCoeff)
                lambda = As/a.weightedAverage(mesh.V()).value();

            a = lambda*a;

            Info << "psiCoeff=" << lambda << endl;
        }
      steadyStateControl steadyState(mesh);
        cpuTime physTimer; 
        while (steadyState.loop())
        {
            #include "UEqn.H"
            #include "pEqn.H"

            laminarTransport.correct();
            turbulence->correct();

            #include "YiEqn.H"
        }

        scalar thisPhysTime = physTimer.elapsedCpuTime();

        if (runTime.timeIndex() % 4 == 0)  
        {
            cpuTime nnTimer;
            int ret = std::system("python gundongypredict.py");

           word h5File = runTime.path()/"predictions/predictions_results.h5";
            predictor.readH5ToFields(h5File, eps, C);

            scalar thisNNTime = nnTimer.elapsedCpuTime();
        }

        eps.correctBoundaryConditions();
        C.correctBoundaryConditions();
        U.correctBoundaryConditions();

        runTime.write();
        runTime.printExecutionTime(Info);
    }
    return 0;
}
