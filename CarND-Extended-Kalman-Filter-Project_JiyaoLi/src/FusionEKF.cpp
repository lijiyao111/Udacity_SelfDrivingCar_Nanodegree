#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  // Hj_ is H matrix for EKF, it will be caculated using
  // Tools::CalculateJacobian(ekf_.x_), thus not really used.
  // Here is just to use it as a placeholder
  Hj_ = MatrixXd(3, 4);  

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  
  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // Acceleration noise
  noise_ax_ = 9.;
  noise_ay_ = 9.;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;

    VectorXd x(4);
    previous_timestamp_ = measurement_pack.timestamp_;

    MatrixXd P = MatrixXd::Identity(4, 4);

    P(2,2) = 1000;
    P(3,3) = 1000;

    MatrixXd F = MatrixXd::Identity(4, 4);
    MatrixXd Q(4, 4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double rho = measurement_pack.raw_measurements_(0);
      double phi = measurement_pack.raw_measurements_(1);
      x << rho*std::cos(phi), rho*std::sin(phi), 0, 0;
      ekf_.Init(x, P, F, Hj_, R_radar_, Q);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1), 0, 0;
      ekf_.Init(x, P, F, H_laser_, R_laser_, Q);
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  double dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  if(dt > 0){
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    double dt2 = dt*dt;
    double dt3 = dt2*dt;
    double dt4 = dt3*dt;

    ekf_.Q_ << dt4/4 * noise_ax_, 0, dt3/2*noise_ax_, 0,
      0, dt4/4*noise_ay_, 0, dt3/2*noise_ay_,
      dt3/2*noise_ax_, 0, dt2*noise_ax_, 0,
      0, dt3/2*noise_ay_, 0, dt2*noise_ay_;
        
    ekf_.Predict();
    cout<<"prediction\n";
  }
  

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;
    ekf_.H_ = Tools::CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    cout<<"radar update\n";
    
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
    cout<<"lidar update\n";
  }

  // print the output
  // cout << "x_ = " << ekf_.x_ << endl;
  // cout << "P_ = " << ekf_.P_ << endl;
}
