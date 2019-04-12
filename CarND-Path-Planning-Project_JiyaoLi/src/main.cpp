#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int LaneNumber(float d) {
  int lane = floor(d / 4);
  return (lane >= 0 && lane <= 2) ? lane : -1;
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // car lane number, start from center lane
  int my_lane = 1;

  // Reference velocity.
  double ref_vel = 0.0;  // m/s

  h.onMessage([&map_waypoints_x, &map_waypoints_y, &map_waypoints_s,
               &map_waypoints_dx, &map_waypoints_dy, &my_lane,
               &ref_vel](uWS::WebSocket<uWS::SERVER> ws, char *data,
                         size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);

        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;

          // /**
          //  * TODO: define a path made up of (x,y) points that the car will
          //  visit
          //  *   sequentially every .02 seconds
          //  */

          // Provided previous path point size.
          int prev_path_size = previous_path_x.size();

          // Preventing collitions.
          if (prev_path_size > 0) {
            car_s = end_path_s;
          }

          std::set<double> left_cars_ahead_dist;
          std::set<double> right_cars_ahead_dist;

          const double Mph2Ms = 0.447;
          const double Ms2Mph = 2.237;
          const double kBufferFrontFar = 100;  // meter
          const double kFrontTargetDist = 30;  // meter
          const double kBufferEnd = 15;        // meter
          const double kTimeInterval = 0.02;   // second

          const double MAX_SPEED = 49.5 * Mph2Ms; // m/s
          // max acc is 10 m/s2, to 0.02s interval, max velocity
          // change is 0.02 m/s
          const double MAX_ACC = 0.15;

          /* Prediction of other cars */
          bool front_clear = true;
          bool left_clear = true;
          bool right_clear = true;

          for (int i = 0; i < sensor_fusion.size(); ++i) {
            float d = sensor_fusion[i][6];
            int other_car_lane = LaneNumber(d);
            if (other_car_lane < 0) {
              continue;
            }
            // Find car speed.
            double vx = sensor_fusion[i][3];
            double vy = sensor_fusion[i][4];
            double other_car_vel = sqrt(vx * vx + vy * vy);
            double other_car_s = sensor_fusion[i][5];
            // Estimate car s position after executing previous trajectory.
            other_car_s +=
                ((double)prev_path_size * kTimeInterval * other_car_vel);

            double car_distance = other_car_s - car_s;

            if (other_car_lane == my_lane) {
              // Car in our lane.
              if (car_distance > 0 && car_distance < kFrontTargetDist) {
                front_clear = false;
              }

            } else if (other_car_lane == my_lane - 1) {
              // Car left
              if (car_distance > -kBufferEnd &&
                  car_distance < kFrontTargetDist) {
                left_clear = false;
              }
              // find the distance of the car in front of us on left
              if (car_distance >= kFrontTargetDist) {
                left_cars_ahead_dist.insert(car_distance);
              }

            } else if (other_car_lane == my_lane + 1) {
              // Car right
              if (car_distance > -kBufferEnd &&
                  car_distance < kFrontTargetDist) {
                right_clear = false;
              }
              // find the distance of the car in front of us on right
              if (car_distance >= kFrontTargetDist) {
                right_cars_ahead_dist.insert(car_distance);
              }
            }
          }

          /* End of Prediction of other cars */

          /* Behavior Planning */
          double speed_diff = 0;

          if (!front_clear) {  //  there is car ahead
            // if both left and right lanes are clear, choose the one where the
            // car is farther away
            if ((left_clear && my_lane != 0) && (right_clear && my_lane != 2)) {
              if (left_cars_ahead_dist.empty()) {
                --my_lane;
              } else if (right_cars_ahead_dist.empty()) {
                ++my_lane;
              } else {
                double left_dist = *(left_cars_ahead_dist.begin());
                double right_dist = *(right_cars_ahead_dist.begin());
                if (left_dist > right_dist) {
                  --my_lane;
                } else {
                  ++my_lane;
                }
              }
            } else if (left_clear && my_lane != 0) {
              // if left lane is clear and available
              --my_lane;
            } else if (right_clear && my_lane != 2) {
              // if right lane is clear and available
              ++my_lane;
            } else {
              speed_diff -= MAX_ACC;
            }
          } else {
            if (my_lane != 1) {
              // since there are more choices to switch lanes if we are on the
              // center lane, if the center lane is empty or close to empty, go
              // to center lane
              if ((my_lane == 0 && right_clear &&
                   (right_cars_ahead_dist.empty() ||
                    *(right_cars_ahead_dist.begin()) > kBufferFrontFar)) ||
                  (my_lane == 2 && left_clear &&
                   (left_cars_ahead_dist.empty() ||
                    *(left_cars_ahead_dist.begin()) > kBufferFrontFar))) {
                my_lane = 1;
              }
            }
            if (ref_vel < MAX_SPEED) {
              speed_diff += MAX_ACC;
            }
          }
          /* End of Behavior Planning */

          // /* Generate Trajectory */

          vector<double> ptsx;
          vector<double> ptsy;

          double ref_x = car_x;
          double ref_y = car_y;
          double car_yaw_rad = deg2rad(car_yaw);

          // // Do I have have previous points
          if (prev_path_size < 2) {
            // There are not too many...
            double prev_car_x = car_x - cos(car_yaw_rad);
            double prev_car_y = car_y - sin(car_yaw_rad);

            ptsx.push_back(prev_car_x);
            ptsx.push_back(car_x);

            ptsy.push_back(prev_car_y);
            ptsy.push_back(car_y);
          } else {
            // Use the last two points.
            // for smoothness and continuity
            ref_x = previous_path_x[prev_path_size - 1];
            ref_y = previous_path_y[prev_path_size - 1];

            double ref_x_prev = previous_path_x[prev_path_size - 2];
            double ref_y_prev = previous_path_y[prev_path_size - 2];
            car_yaw_rad = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

            ptsx.push_back(ref_x_prev);
            ptsx.push_back(ref_x);

            ptsy.push_back(ref_y_prev);
            ptsy.push_back(ref_y);
          }

          // Setting up target points in the future.
          vector<double> next_wp0 =
              getXY(car_s + kFrontTargetDist, 2 + 4 * my_lane, map_waypoints_s,
                    map_waypoints_x, map_waypoints_y);
          vector<double> next_wp1 =
              getXY(car_s + kFrontTargetDist * 2, 2 + 4 * my_lane,
                    map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp2 =
              getXY(car_s + kFrontTargetDist * 3, 2 + 4 * my_lane,
                    map_waypoints_s, map_waypoints_x, map_waypoints_y);

          ptsx.push_back(next_wp0[0]);
          ptsx.push_back(next_wp1[0]);
          ptsx.push_back(next_wp2[0]);

          ptsy.push_back(next_wp0[1]);
          ptsy.push_back(next_wp1[1]);
          ptsy.push_back(next_wp2[1]);

          // Making coordinates to local car coordinates.
          for (int i = 0; i < ptsx.size(); i++) {
            double shift_x = ptsx[i] - ref_x;
            double shift_y = ptsy[i] - ref_y;

            ptsx[i] = shift_x * cos(car_yaw_rad) + shift_y * sin(car_yaw_rad);
            ptsy[i] = -shift_x * sin(car_yaw_rad) + shift_y * cos(car_yaw_rad);
          }

          // Create the spline, which is based on the coordinate created above
          tk::spline spline_line;
          spline_line.set_points(ptsx, ptsy);

          // Calculate distance y position on kFrontTargetDist ahead.
          double target_x = kFrontTargetDist;
          double target_y = spline_line(target_x);
          double target_dist = sqrt(target_x * target_x + target_y * target_y);

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          // Output path points from previous path for continuity and
          // smoothness.
          for (int i = 0; i < prev_path_size; i++) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          double x_add_on = 0;

          for (int i = 0; i < 50 - prev_path_size; i++) {
            ref_vel += speed_diff;
            if (ref_vel > MAX_SPEED) {
              ref_vel = MAX_SPEED;
            } else if (ref_vel < MAX_ACC) {
              ref_vel = MAX_ACC;
            }

            double N = target_dist / (kTimeInterval * ref_vel);
            double x_point = x_add_on + target_x / N;
            double y_point = spline_line(x_point);

            x_add_on = x_point;

            if (x_add_on > target_x) {
              break;
            }

            double x_ref = x_point;
            double y_ref = y_point;

            x_point =
                ref_x + x_ref * cos(car_yaw_rad) - y_ref * sin(car_yaw_rad);
            y_point =
                ref_y + x_ref * sin(car_yaw_rad) + y_ref * cos(car_yaw_rad);

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }

          // /* End of Generate Trajectory */

          // // End of my code

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\"," + msgJson.dump() + "]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  });  // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  h.run();
}