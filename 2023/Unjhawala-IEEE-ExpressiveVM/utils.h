#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <functional>


static const double C_PI = 3.141592653589793238462643383279;
static const double C_2PI = 6.283185307179586476925286766559;
static const double G = 9.81; // gravity constant
static const double rpm2rad = C_PI / 30;

/// Structure for storing driver input - Similar to Chrono
struct Entry{
    Entry() {} // alias
    // constructor
    Entry(double time, double steering, double throttle, double braking)
        : m_time(time), m_steering(steering), m_throttle(throttle), m_braking(braking) {}

    double m_time;
    double m_steering;
    double m_throttle;
    double m_braking;
};

/// Structure for entries of any kind of map (anything with x and y)
struct MapEntry{
    MapEntry() {} // alias
    //constructor
    MapEntry(double x, double y)
        : _x(x), _y(y) {}
    

    double _x; // x coordinate of the map
    double _y; // y coordinate of the map
};

/// Structure for entries of any kind of map (anything with x and y)
struct TiRainMapEntry{
    TiRainMapEntry() {} // alias
    //constructor
    TiRainMapEntry(double sinkage, double vel, double omega, double fz, double fx, double tau)
        : _sinkage(sinkage), _vel(vel), _omega(omega), _fz(fz), _fx(fx), _tau(tau) {}
    
    // Inputs (x coordinates)
    double _sinkage; // Sinkage - which is one of the x coordinates of the Map
    double _vel; // Linear Velocity - another x coordinate of the map
    double _omega; // Angular velocity - final input x coordinate of the map

    // Outputs (y coordinates)
    double _fz; // Vertical force
    double _fx; // Longitudinal force
    double _tau; // Resistive torque
};


/// Driver inputs from data file.
void driverInput(std::vector <Entry>& m_data ,const std::string& filename);

/// function needed to compare times for driver input
inline bool compareTime(const Entry& a, const Entry& b){ return a.m_time < b.m_time; };

/// function needed to compare RPM while interpolating torque map
inline bool compareRPM(const MapEntry& a, const MapEntry& b){return a._x < b._x; };

/// fucntion to get the Y of the map by linear interpolation
double getMapY(std::vector<MapEntry>& map, const double x);

/// Function to get the vehicle controls at a given time
/// need to pass the data as well
void getControls(std::vector <double>& controls, std::vector<Entry>& m_data, const double time);

// linear interpolation function
inline double InterpL(double fz, double w1, double w2, double pn) { return w1 + (w2 - w1) * (fz / pn - 1.); }

// quadratic interpolation function
inline double InterpQ(double fz, double w1, double w2, double pn) { return (fz/pn) * (2. * w1 - 0.5 * w2 - (w1 - 0.5 * w2) * (fz/pn)); }

// temlate safe signum function 
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


double sineStep(double x, double x1, double y1, double x2, double y2);

// clamp function from chrono
template <typename T>
T clamp(T value, T limitMin, T limitMax) {
    if (value < limitMin)
        return limitMin;
    if (value > limitMax)
        return limitMax;

    return value;
};

class CSV_writer {
  public:
    explicit CSV_writer(const std::string& delim = ",") : m_delim(delim) {}

    CSV_writer(const CSV_writer& source) : m_delim(source.m_delim) {
        // Note that we do not copy the stream buffer (as then it would be shared!)
        m_ss.copyfmt(source.m_ss);          // copy all data
        m_ss.clear(source.m_ss.rdstate());  // copy the error state
    }

    ~CSV_writer() {}

    void write_to_file(const std::string& filename, const std::string& header = "") const {
        std::ofstream ofile(filename.c_str());
        ofile << header;
        ofile << m_ss.str();
        ofile.close();
    }

    const std::string& delim() const { return m_delim; }
    std::ostringstream& stream() { return m_ss; }

    template <typename T>
    CSV_writer& operator<<(const T& t) {
        m_ss << t << m_delim;
        return *this;
    }

    CSV_writer& operator<<(std::ostream& (*t)(std::ostream&)) {
        m_ss << t;
        return *this;
    }
    CSV_writer& operator<<(std::ios& (*t)(std::ios&)) {
        m_ss << t;
        return *this;
    }
    CSV_writer& operator<<(std::ios_base& (*t)(std::ios_base&)) {
        m_ss << t;
        return *this;
    }

  private:
    std::string m_delim;
    std::ostringstream m_ss;
};


template <typename T>
inline CSV_writer& operator<<(CSV_writer& out, const std::vector<T>& vec) {
    for (const auto& v : vec)
        out << v;
    return out;
}


#endif