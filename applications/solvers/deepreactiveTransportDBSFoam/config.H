#include "fileName.H"

IOdictionary cnn_config
(
    IOobject
    (
        "cnn_config",
        runTime.constant(),   
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

const std::string cnn_model_path = runTime.constantPath() + "/" + word(cnn_config.lookup("cnnModelPath"));
const std::string device_type    = word(cnn_config.lookup("device"));


// ************************************************************************* //
