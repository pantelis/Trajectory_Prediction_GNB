//
// Created by Pantelis Monogioudis on 11/22/17.
//

#ifndef TRAJECTORY_PREDICTION_GNB_CLASSIFIER_H
#define TRAJECTORY_PREDICTION_GNB_CLASSIFIER_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:

    vector<string> possible_labels = {"left","keep","right"};


    /**
      * Constructor
      */
    GNB();

    /**
     * Destructor
     */
    virtual ~GNB();

    void train(vector<vector<double> > data, vector<string>  labels);

    string predict(vector<double>);

private:
    // number of features
    int F;

    // number of classes
    int K;

    vector<double> mu_left;
    vector<double> var_left;

    vector<double> mu_right;
    vector<double> var_right;

    vector<double> mu_keep;
    vector<double> var_keep;

    // class prior probabilities
    double p_left_prior = 0.0;
    double p_right_prior = 0.0;
    double p_keep_prior = 0.0;

    vector<double> p_prior;
};





#endif //TRAJECTORY_PREDICTION_GNB_CLASSIFIER_H
