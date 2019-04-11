/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

const double ZERO_THRESH = 0.00001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  if(is_initialized){
    return;
  }

  num_particles = 101;
  std::cout<<"init start!\n";

    
  // TODO: Set standard deviations for x, y, and theta
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);


  for(int i=0; i<num_particles; ++i){
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }

  is_initialized = true;
  
  std::cout<<"init good!\n";
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  std::cout<<"prediction start!\n";

  
  normal_distribution<double> dist_x_noise(0, std_pos[0]);
  normal_distribution<double> dist_y_noise(0, std_pos[1]);
  normal_distribution<double> dist_theta_noise(0, std_pos[2]);


  for(int i=0; i<num_particles; ++i){

    // calculate new state
    if (fabs(yaw_rate) < ZERO_THRESH) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else{
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += dist_x_noise(gen);
    particles[i].y += dist_y_noise(gen); 
    particles[i].theta += dist_theta_noise(gen);

  }

  std::cout<<"prediction good!\n";

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


  for(int i=0; i<int(observations.size()); i++){

    LandmarkObs& obs = observations[i];
    
    double minDist= numeric_limits<double>::max();
    obs.id = -1;
    for(int j=0; j<int(predicted.size()); j++){
      const LandmarkObs& pred = predicted[j];
      double currDist = dist_square(obs.x, obs.y, pred.x, pred.y);
      if(currDist<minDist){
        minDist = currDist;
        obs.id = pred.id;
      }
    }
  }

  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  std::cout<<"update weights starts!\n";

  for(int i=0; i<num_particles; i++){

    // get particles
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    //
    // for loop to get the predicted measurements
    vector<LandmarkObs> pred_landmarks;
    const auto& landmarks_in_map = map_landmarks.landmark_list;
    for(int j=0; j<int(landmarks_in_map.size()); ++j){
      int lm_map_id = landmarks_in_map[j].id_i;
      double lm_map_x = landmarks_in_map[j].x_f;
      double lm_map_y = landmarks_in_map[j].y_f;

      if(!(fabs(p_x - lm_map_x)>sensor_range) && !(fabs(p_y - lm_map_y)>sensor_range) && 
         dist(p_x, p_y, lm_map_x, lm_map_y) <= sensor_range ){
        
        pred_landmarks.push_back(LandmarkObs{lm_map_id, lm_map_x, lm_map_y});
      }

    }

    vector<LandmarkObs> transformed_os;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }

    dataAssociation(pred_landmarks, transformed_os);

    particles[i].weight = 1.0;

    for(int j=0; j<int(transformed_os.size()); j++){
      const LandmarkObs& ob = transformed_os[j];

      for(int k=0; k<int(pred_landmarks.size()); k++){
        const LandmarkObs& pred = pred_landmarks[k];
        if(ob.id == pred.id){
          double obs_diff_x = ob.x - pred.x;
          double obs_diff_y = ob.y - pred.y;

          double s_x = std_landmark[0];
          double s_y = std_landmark[1];
          double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(obs_diff_x, 2)/(2*pow(s_x, 2)) + (pow(obs_diff_y, 2)/(2*pow(s_y, 2))) ) );
          
          particles[i].weight *= obs_w; 

        }
      }
    }
  }
  

  std::cout<<"update weights good!\n";

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::cout<<"resample starts!\n";

  std::vector<Particle> new_particles;
  std::vector<double> weights;
  for(const auto& p: particles){
    weights.push_back(p.weight);
  }
  
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  int index = uniintdist(gen);

  double max_weight = *max_element(weights.begin(), weights.end());
  
  double beta = 0.0;

  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  for(int i=0; i<num_particles; ++i){
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
  std::cout<<"resample good!\n";
  
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
