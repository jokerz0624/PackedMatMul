//
// Created by cc 
// 

#pragma once

#include <chrono>
#include <ctime>
#include <ratio>
#include <iomanip>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <numeric>

class Timer {
public:
    Timer() : begin_(std::chrono::high_resolution_clock::now()) {
    }

    void reset() {
        begin_ = std::chrono::high_resolution_clock::now();
    }

    double cost() const {   //ms
        auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - begin_).count();
        return time_cost / 1000.0;
    }
   
private:
    std::chrono::high_resolution_clock::time_point begin_;
};
