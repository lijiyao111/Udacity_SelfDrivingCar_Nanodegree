#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID(): p_error(0.0), i_error(0.0) {}

PID::~PID() {}

void PID::Init(double Kp_in, double Ki_in, double Kd_in) {
  Kp = Kp_in;
  Ki = Ki_in;
  Kd = Kd_in;
}

void PID::UpdateError(double cte) {
  d_error = (cte - p_error);
  p_error = cte;
  i_error += cte;
}

double PID::TotalError() {
  return -1*(Kp*p_error + Kd*d_error + Ki*i_error);
}

