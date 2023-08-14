#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <stdint.h>
#include "utils.h"



/// Driver inputs from data file.
/// A driver model based on user inputs provided as time series. If provided as a
/// text file, each line in the file must contain 4 values:
///   time steering throttle braking
/// It is assumed that the time values are unique and soted from 0 to T
void driverInput(std::vector <Entry>& m_data ,const std::string& filename){

    std::ifstream ifile(filename.c_str());
    std::string line;

    
    // get each line
    while(std::getline(ifile,line)){
        std::istringstream iss(line);
        
        double time, steering, throttle, braking;

        // put the stream into our varaibles
        iss >> time >> steering >> throttle >> braking;

        if (iss.fail())
            break;

        // push into our structure
        m_data.push_back(Entry(time,steering,throttle,braking));
    }

    ifile.close();
}



/// Function to get the vehicle controls at a given time
/// need to pass the data as well

void getControls(std::vector <double>& controls, std::vector<Entry>& m_data, const double time){
    
    // if its before time or after time
    if(time <= m_data[0].m_time){
        controls[0] = time;
        controls[1] = m_data[0].m_steering;
        controls[2] = m_data[0].m_throttle;
        controls[3] = m_data[0].m_braking;
        return;
    } else if(time >= m_data.back().m_time){
        controls[0] = time;
        controls[1] = m_data.back().m_steering;
        controls[2] = m_data.back().m_throttle;
        controls[3] = m_data.back().m_braking;
        return;
    }

    // if its within the time, get an iterator and do linear interpolation
    // use compare fucntion earlier defined

    std::vector<Entry>::iterator right =
        std::lower_bound(m_data.begin(), m_data.end(), Entry(time,0,0,0), compareTime); // return first value after Entry
    
    std::vector<Entry>::iterator left = right - 1;

    // linear interplolation
    
    double tbar = (time - left->m_time) / (right->m_time - left->m_time);

    controls[0] = time;
    controls[1] = left->m_steering + tbar * (right->m_steering - left->m_steering);
    controls[2] = left->m_throttle + tbar * (right->m_throttle - left->m_throttle);
    controls[3] = left->m_braking + tbar * (right->m_braking - left->m_braking);
}

// sine step function for some smoothing operations
double sineStep(double x, double x1, double y1, double x2, double y2){
    if (x <= x1)
        return y1;
    if (x >= x2)
        return y2;
    
    double dx = x2 - x1;
    double dy = y2 - y1;
    double y = y1 + dy * (x - x1) / dx - (dy / C_2PI) * std::sin(C_2PI * (x - x1) / dx);
    return y;
}


double getMapY(std::vector<MapEntry>& map, const double x){
    // if speed is less than or more than the min max, return the min max
    if(x <= map[0]._x){
        return map[0]._y;
    } else if(x >= map.back()._x){
        return map.back()._y;
    }

    // if its within the time, get an iterator and do linear interpolation
    // use compare fucntion earlier defined

    std::vector<MapEntry>::iterator right =
        std::lower_bound(map.begin(), map.end(), MapEntry(x,0), compareRPM); // return first value after Entry
    
    std::vector<MapEntry>::iterator left = right - 1;

    // linear interplolation
    
    double mbar = (x - left->_x) / (right->_x - left->_x);

    return (left->_y + mbar * (right->_y - left->_y));
}





