#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

double poly_derivative(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 1; i < coeffs.size(); i++) {
    result += i * coeffs[i] * pow(x, i-1);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px0 = j[1]["x"];
          double py0 = j[1]["y"];
          double psi0 = j[1]["psi"];
          double v0 = j[1]["speed"];
          double delta0 = j[1]["steering_angle"];
          double a0 = j[1]["throttle"];

          delta0 = - delta0;
          // fix latency
          // the latency value may be different in different computer due to the perfermence difference
          // it may be between [0.05, 1]
          double latency = 0.05; 
          double Lf = 2.67;
          // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
          // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
          // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
          // v_[t+1] = v[t] + a[t] * dt
          // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
          // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
          double px = px0 + v0 * cos(psi0) * latency;
          double py = py0 + v0 * sin(psi0) * latency;
          double psi = psi0 + v0 / Lf * delta0 * latency;
          double v = v0 + a0 * latency;

          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          cout << "transform ptsx, ptsy" << endl;
          // transform ptsx, ptsy
          vector<double> r_ptsx;
          vector<double> r_ptsy;
          for (int i = 0; i < ptsx.size(); ++i) {
            double r_x =    cos(psi) * (ptsx[i] - px) + sin(psi) * (ptsy[i] - py );
            double r_y =  - sin(psi) * (ptsx[i] - px) + cos(psi) * (ptsy[i] - py );
            r_ptsx.push_back(r_x);
            r_ptsy.push_back(r_y);
          }

          Eigen::VectorXd v_ptsx = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(r_ptsx.data(), r_ptsx.size());
          Eigen::VectorXd v_ptsy = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(r_ptsy.data(), r_ptsy.size());
          cout << "ptsx_relative: " << v_ptsx.transpose() << endl;
          cout << "ptsy_relative: " << v_ptsy.transpose() << endl;

          Eigen::VectorXd poly_coeffs = polyfit(v_ptsx, v_ptsy, 3);

          // calculate cte, epsi
          double cte = polyeval(poly_coeffs, 0.0);
          double epsi = - atan( poly_derivative(poly_coeffs, 0.0));

          Eigen::VectorXd state(6);
          state << 0, 0, 0, v, cte, epsi;
          cout<< "state: " << state.transpose() << endl;

          auto mpc_result = mpc.Solve(state, poly_coeffs);
          cout << "ret size: " << mpc_result.size() << endl;
          
          double steer_value = mpc_result[0];
          double throttle_value = mpc_result[1];
          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          // msgJson["steering_angle"] = 0;
          msgJson["steering_angle"] = - steer_value / deg2rad(25);
          // msgJson["throttle"] = 0.1;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
          
          for (int i = 0; i < 5; ++i) {
            // double r_x =    cos(psi) * (mpc_result[2+2*i] - px) + sin(psi) * (mpc_result[2+2*i+1] - py );
            // double r_y =  - sin(psi) * (mpc_result[2+2*i] - px) + cos(psi) * (mpc_result[2+2*i+1] - py );
            double r_x = mpc_result[2+2*i];
            double r_y = mpc_result[2+2*i+1];
            mpc_x_vals.push_back(r_x);
            mpc_y_vals.push_back(r_y);
          }
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          for (int i = 0; i < ptsx.size(); ++i){
            double r_x =    cos(psi) * (ptsx[i] - px) + sin(psi) * (ptsy[i] - py );
            next_x_vals.push_back(r_x);
            next_y_vals.push_back(polyeval(poly_coeffs, r_x));
          }

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

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
