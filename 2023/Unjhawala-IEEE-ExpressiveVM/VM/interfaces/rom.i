%module rom

%include "std_string.i"
%include "std_vector.i"


%{
#include "../utils.h"
#include "Eightdof.h"
using namespace EightDOF;
%}

// initiate our vector of entries and doubles
%template(vector_entry) std::vector <Entry>;
%template(vector_mapEntry) std::vector <MapEntry>;
%template(vector_double) std::vector <double>;

%include "../utils.h"
%include "Eightdof.h"

