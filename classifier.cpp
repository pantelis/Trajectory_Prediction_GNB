//
// Created by Pantelis Monogioudis on 11/22/17.
//

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <numeric>
#include <boost/range/irange.hpp>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector< vector<double> > data, vector<string> labels)
{

    /*
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d,
            s_dot and d_dot (features).
          - Example :
                  3.5, 0.1, 5.9, -0.02],
                  8.0, -0.3, 3.0, 2.2],
                  ...


        labels - array of N labels
          - Each label is one of "left", "keep", or "right".
    */

    // number of training samples
    const int n = labels.size();

    // number of features
    F = data[0].size();

    // number of classes (left, keep right)
    K = 3;

    // Initialize means and variances of the Gaussian model
    // across features f and classes (left, right, keep)
    for (int i : boost::irange(0, F)) {

        mu_left.push_back(0.0);
        var_left.push_back(0.0);
        mu_right.push_back(0.0);
        var_right.push_back(0.0);
        mu_keep.push_back(0.0);
        var_keep.push_back(0.0);
    }

    // store all the indeces of the data matrix associated with each classes
    vector<long> left_indeces;
    vector<long> right_indeces;
    vector<long> keep_indeces;

    auto n_left(0);
    auto n_right(0);
    auto n_keep(0);

    for (vector<string>::iterator iter = labels.begin(); iter != labels.end(); iter++)
    {
        if ((*iter) == "left"){
            p_left_prior++;
            n_left++;
            left_indeces.push_back(std::distance(labels.begin(), iter));
        }
        else if ((*iter) == "right"){
            p_right_prior++;
            n_right++;
            right_indeces.push_back(std::distance(labels.begin(), iter));
        }
        else if ((*iter) == "keep"){
            p_keep_prior++;
            n_keep++;
            keep_indeces.push_back(std::distance(labels.begin(), iter));
        }
    }

    // prior probability is normalized across **all** number of data samples
    p_left_prior /= n;
    p_right_prior /= n;
    p_keep_prior /= n;
    p_prior.push_back(p_left_prior);
    p_prior.push_back(p_keep_prior);
    p_prior.push_back(p_right_prior);


    // Segment the data matrix into indivisual matrices left, right and keep.
    vector<vector<double>> p_left_data(n_left);
    vector<vector<double>> p_right_data(n_right);
    vector<vector<double>> p_keep_data(n_keep);

    auto i(0);
    for (auto& l : left_indeces)
    {
        for (auto& f : {0, 1, 2, 3})
        {
            p_left_data[i].push_back(data[l][f]);
        }
        i++;
    }

    i = 0;
    for (auto& r : right_indeces)
    {
        for (auto& f : {0, 1, 2, 3})
        {
            p_right_data[i].push_back(data[r][f]);
        }
        i++;
    }

    i = 0;
    for (auto& k : keep_indeces)
    {
        for (auto& f : {0, 1, 2, 3})
        {
            p_keep_data[i].push_back(data[k][f]);
        }
        i++;
    }

    // Estimate the Likelihood function p(x | C_k) assuming that its
    // distributed according to a N(mu_k, sigma_k^2). For this we need
    // the estimates of the mean and variance per feature per class.

    // mean of each feature per class
    for (auto& l : p_left_data)
    {
        for (auto f : {0,1,2,3}) {
            mu_left[f] += l[f] / n_left;
        }
    }

    for (auto& r : p_right_data)
    {
        for (auto f : {0,1,2,3}) {
            mu_right[f] += r[f] / n_right;
        }
    }

    for (auto& k : p_keep_data)
    {
        for (auto f : {0,1,2,3}) {
            mu_keep[f] += k[f] / n_keep;
        }
    }


    // var of each feature per class
    for (auto& l : p_left_data)
    {
        for (auto f : {0,1,2,3}) {
            var_left[f] += ((l[f] - mu_left[f]) * (l[f] - mu_left[f])) / n_left;
        }
    }

    for (auto& r : p_right_data)
    {
        for (auto f : {0,1,2,3}) {
            var_right[f] += ((r[f] - mu_right[0]) * (r[f] - mu_right[f])) / n_right;
        }
    }

    for (auto& k : p_keep_data)
    {
        for (auto f : {0,1,2,3}) {
            var_keep[f] += ((k[f] - mu_keep[0]) * (k[f] - mu_keep[f])) / n_keep;
        }
    }
}



string GNB::predict(vector<double> sample)
{
    /*
        Once trained, this method is called and expected to return
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
        """

    */

    // Estimate the likelihood function
    vector<double> likelihood_func_left;
    vector<double> likelihood_func_right;
    vector<double> likelihood_func_keep;

    for (auto& f : {0,1,2,3})
    {

        likelihood_func_left.push_back( (1./sqrt(2. * M_PI * var_left[f])) *
                                           exp(-(sample[f] - mu_left[f]) * (sample[f] - mu_left[f])  /
                                                       (2. * var_left[f])) );

        likelihood_func_right.push_back( (1./sqrt(2. * M_PI * var_right[f])) *
                                        exp(-(sample[f] - mu_right[f]) * (sample[f] - mu_right[f])  /
                                            (2. * var_right[f])) );

        likelihood_func_keep.push_back( (1./sqrt(2. * M_PI * var_keep[f])) *
                                        exp(-(sample[f] - mu_keep[f]) * (sample[f] - mu_keep[f])  /
                                            (2. * var_keep[f])) );

    }


    // Bayes rule assuming the "naive" conditional independence assumption:
    // each feature is conditionally independent of every other feature given the class.
    vector<double> p_class;
    for (auto k : {0, 1, 2})
    {
        p_class.push_back(p_prior[k]);
    }
    for (auto& f : {0, 1, 2, 3})
    {
        p_class[0] *= likelihood_func_left[f];
        p_class[1] *= likelihood_func_keep[f];
        p_class[2] *= likelihood_func_right[f];
    }

    double normalization = std::accumulate(p_class.begin(),p_class.end(),0.0);

    for (auto p : boost::irange(0, K))
    {
        p_class[p] /= normalization;
    }

    // arg max over the classes
    std::vector<double>::iterator max_result;
    max_result = std::max_element(p_class.begin(), p_class.end());
    long index = std::distance(p_class.begin(), max_result);

    switch (index)
    {
        case 0:
            return "left";
        case 1:
            return "keep";
        case 2:
            return "right";
        default:
            cout << "Not valid classification: must be left, keep or right" << endl;
    }
}