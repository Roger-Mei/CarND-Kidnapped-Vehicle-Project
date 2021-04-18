/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  // Number of particles is chose in order to run the algorithm in real time
  num_particles = 30;  // TODO: Set the number of particles

  std::default_random_engine gen;

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    // Initialize each particle
    Particle current_particle;
    current_particle.id = i;
    current_particle.x = dist_x(gen);
    current_particle.y = dist_y(gen);
    current_particle.theta = dist_theta(gen);
    current_particle.weight = 1.0;

    // construct particles vector and weight vector
    particles.push_back(current_particle);
    weights.push_back(current_particle.weight);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; ++i)
  {
    // Extract currrent state information
    double curr_x = particles[i].x;
    double curr_y = particles[i].y;
    double curr_theta = particles[i].theta;

    // Initialize predicted state
    double pred_x;
    double pred_y;
    double pred_theta;

    // Different kinematic model based on yaw rate scale
    if (fabs(yaw_rate) < 0.0001)
    {
      pred_x = curr_x + velocity * cos(curr_theta) * delta_t;
      pred_y = curr_y + velocity * sin(curr_theta) * delta_t;
      pred_theta = curr_theta;
    }
    else
    {
      pred_x = curr_x + (velocity/yaw_rate) * (sin(curr_theta + (yaw_rate * delta_t)) - sin(curr_theta));
      pred_y = curr_y + (velocity/yaw_rate) * (cos(curr_theta) - cos(curr_theta + (yaw_rate * delta_t)));
      pred_theta = curr_theta + (yaw_rate * delta_t);
    }

    std::normal_distribution<double> dist_x(pred_x, std_pos[0]);
    std::normal_distribution<double> dist_y(pred_y, std_pos[1]);
    std::normal_distribution<double> dist_theta(pred_theta, std_pos[2]);

    // Update each particle with the predicted states
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < observations.size(); ++i)
  {
    // Initialization
    double min_distance = 500.0; // Randomly initialize with a large number
    int closest_landmark_id = -1;
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;

    for (int j = 0; j < predicted.size(); ++j)
    {
      double pred_x = predicted[j].x;
      double pred_y = predicted[j].y;
      int pred_id = predicted[j].id;
      double curr_dist = dist(obs_x, obs_y, pred_x, pred_y);

      // Update nearest neighbor
      if (curr_dist < min_distance)
      {
        min_distance = curr_dist;
        closest_landmark_id = pred_id;
      }
    }
    observations[i].id = closest_landmark_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  // Initialize weight normalizer
  double weight_normalizer = 0.0;

  for (int i = 0; i < num_particles; i++)
  {
    double curr_x = particles[i].x;
    double curr_y = particles[i].y;
    double curr_theta = particles[i].theta;

    /* Step 1: Transform observation from vehicle frame to world fram*/
    vector<LandmarkObs> transformed_observations;

    for (int j = 0; j < observations.size(); ++j)
    {
      LandmarkObs transformed_observation;
      transformed_observation.id = j;
      transformed_observation.x = curr_x + (cos(curr_theta) * observations[j].x) - (sin(curr_theta) * observations[j].y);
      transformed_observation.y = curr_y + (sin(curr_theta) * observations[j].x) + (cos(curr_theta) * observations[j].y);
      transformed_observations.push_back(transformed_observation);
    } 

     /* Step 2: Exclude the landmarks in the map that are not in the sensor range and puish them to predictions vector*/
    vector<LandmarkObs> predicted_landmarks;
    for (int k = 0; k < map_landmarks.landmark_list.size(); ++k)
    {
      Map::single_landmark_s curr_landmark = map_landmarks.landmark_list[k];
      if ((fabs(curr_x - curr_landmark.x_f) <= sensor_range) && (fabs(curr_y - curr_landmark.y_f) <= sensor_range))
      {
        predicted_landmarks.push_back(LandmarkObs {curr_landmark.id_i, curr_landmark.x_f, curr_landmark.y_f});
      }
    }

    /* Step 3: Landmark Association*/
    dataAssociation(predicted_landmarks, transformed_observations);

    /* Step 4: Calculate the weight of each particle usingg Multivariate Gaussian Distribution*/
    particles[i].weight = 1.0;

    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double sigma_x_2 = pow(sigma_x, 2);
    double sigma_y_2 = pow(sigma_y, 2);
    double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));

    for (int k = 0; k < transformed_observations.size(); ++k)
    {
      double trans_obs_x = transformed_observations[k].x;
      double trans_obs_y = transformed_observations[k].y;
      double trans_obs_id = transformed_observations[k].id;
      double multi_prob = 1.0;

      for (int l = 0; l < predicted_landmarks.size(); ++l)
      {
        double pred_landmark_x = predicted_landmarks[l].x;
        double pred_landmark_y = predicted_landmarks[l].y;
        double pred_landmark_id = predicted_landmarks[l].id;

        if (trans_obs_id == pred_landmark_id)
        {
          multi_prob = normalizer * exp(-1.0 * ((pow((trans_obs_x - pred_landmark_x), 2)/(2.0 * sigma_x_2)) + 
                                                (pow((trans_obs_y - pred_landmark_y), 2)/(2.0 * sigma_y_2))));
          particles[i].weight *= multi_prob;
        }
      }
    }
    weight_normalizer += particles[i].weight;                                                                  
  }

  /* Step 5: Normalize the weight of all particles*/
  for (int i = 0; i < particles.size(); ++i)
  {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Initialization
  vector<Particle> resampled_particles;
  std::default_random_engine gen;

  // Generate random particle index
  std::uniform_int_distribution<int> particle_idx(0, num_particles - 1);

  int curr_idx = particle_idx(gen);

  double beta = 0.0;

  double max_weight_threshold = 2.0 * *std::max_element(weights.begin(), weights.end());

  for (int i = 0; i < particles.size(); ++i)
  {
    std::uniform_real_distribution<double> random_weight(0.0, max_weight_threshold);
    beta += random_weight(gen);

    while (beta > weights[curr_idx])
    {
      beta -= weights[curr_idx];
      curr_idx = (curr_idx + 1) % num_particles;
    }

    resampled_particles.push_back(particles[curr_idx]);
  }

  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}